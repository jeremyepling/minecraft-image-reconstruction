"""Image-only fragment joins and native-pixel, visible-only grading on the frozen views."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark import current_placement_viewer as viewer, facade_inset_round as suite
from benchmark.pavement_opening_eval import visible_faces
from benchmark.shared_face_error_budget import LATEST
from benchmark.texture_camera_eval import read, save
from benchmark.wall_texture_evaluate import metrics
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs
from engine.surface_perimeter import draw_outline

OUT = Path('benchmark/reviews/concrete-shadow-boundaries-01')
MODES = ('color', 'guarded')
FOCUS = ('shopfront-plaster-low-left', 'registry-plaster-high-right',
         'portico-plaster-low-left', 'courtyard-plaster-low-left')


@lru_cache(maxsize=1)
def baseline():
    return {row['id']:row for row in read(LATEST)['rows']}


def seam_features(rgb, masks, parts):
    # Sample the two observed sides directly. Do not bridge a third detected
    # surface. Mask overlaps are deliberately excluded from the interface.
    owner = np.zeros(rgb.shape[:2], np.int32)
    for i, mask in enumerate(masks, 1):
        owner[mask] = np.where(owner[mask] == 0, i, -1)
    n = len(masks) + 1
    contacts = defaultdict(list)
    for dy, dx in [(0,1), (1,0), (0,2), (2,0)]:
        a = (slice(None, -dy or None), slice(None, -dx or None))
        b = (slice(dy, None), slice(dx, None))
        left, right = owner[a], owner[b]
        valid = (left > 0) & (right > 0) & (left != right)
        if dx + dy == 2:
            mid = owner[dy//2:owner.shape[0]-dy//2, dx//2:owner.shape[1]-dx//2]
            valid &= (mid == 0) | (mid == left) | (mid == right)
        ids = np.minimum(left[valid], right[valid]) * n + np.maximum(left[valid], right[valid])
        ca, cb = rgb[a][valid], rgb[b][valid]
        delta = cb - ca
        luminance = np.abs(delta.mean(axis=1))
        # Residual after allowing a multiplicative brightness change.
        scaled = ca * ((cb.mean(axis=1) + 1) / (ca.mean(axis=1) + 1))[:,None]
        chroma = np.abs(cb - scaled).max(axis=1)
        for pair in np.unique(ids):
            hits = ids == pair
            contacts[int(pair)].append(np.column_stack((luminance[hits], chroma[hits])))
    proposals = []
    for pair, samples in contacts.items():
        i, j = divmod(pair, n)
        data = np.concatenate(samples)
        if len(data) < 10:
            continue
        med, upper = np.quantile(data, [.5,.8], axis=0)
        if med[0] > 27 or upper[0] > 38 or med[1] > 4 or upper[1] > 8:
            continue
        pa,pb = parts[i-1],parts[j-1]
        na,nb = [np.asarray(p.get('normal_camera') or [0,0,0]) for p in (pa,pb)]
        cosine = np.dot(na,nb) / max(1e-9, np.linalg.norm(na)*np.linalg.norm(nb))
        da,db = [p.get('relative_inverse_depth') for p in (pa,pb)]
        depth = abs(da-db)/max(abs(da),abs(db),1e-6) if da is not None and db is not None else 1.
        area = min(masks[i-1].sum(),masks[j-1].sum())
        guarded = cosine >= .97 and depth <= .045 and len(data) >= max(14, .8*np.sqrt(area)) and upper[0] <= 17
        proposals.append((float(med[1]+.2*med[0]-.005*min(200,len(data))), i-1, j-1, bool(guarded)))
    return sorted(proposals)


def apply_pairs(parts, proposals, size, mode):
    used = set()
    joins = []
    result = []
    for _, i, j, guarded in proposals:
        if (mode == 'guarded' and not guarded) or i in used or j in used:
            continue
        used.update((i,j))
        joins.append((i,j))
        union = native_mask(parts[i],size) | native_mask(parts[j],size)
        # This surface-only experiment scores the observable union. It does not
        # generate a chart for any downstream counter or fabricate gap pixels.
        result.append(dict(id='union-'+str(i)+'-'+str(j),
                           support=dict(size=list(size),runs=row_runs(union))))
    return [p for i,p in enumerate(parts) if i not in used] + result, joins


def label_maps(labels, rays):
    sid = np.zeros(len(labels['faces'])+1,np.int32)
    concrete = np.zeros_like(sid, bool)
    blocks = {b['id']:b['block_id'] for b in labels['blocks']}
    for face in labels['faces']:
        idx = face['id']
        sid[idx] = face['surface_id']
        concrete[idx] = blocks[face['block']].endswith('_concrete')
    return sid[rays], concrete


def shade_audit(image, rays, concrete):
    # A local brightness change on ONE visible concrete block face cannot be a
    # geometric boundary between blocks. Chroma-stable shifts are compatible
    # with lighting; resource texture and renderer effects remain possible.
    rgb = np.asarray(image)[::2,::2].astype(np.float32)
    faces = rays[::2,::2]
    counts = Counter()
    for a,b in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        valid = (faces[a] == faces[b]) & concrete[faces[a]]
        ca,cb = rgb[a][valid],rgb[b][valid]
        lum = np.abs(cb.mean(1) - ca.mean(1))
        residual = np.abs(cb - ca * ((cb.mean(1)+1)/(ca.mean(1)+1))[:,None]).max(1)
        counts['same_face_pairs'] += int(valid.sum())
        counts['luminance_step_ge_12'] += int((lum>=12).sum())
        counts['chroma_stable_steps'] += int(((lum>=12) & (residual<=4)).sum())
    return dict(counts)


def surface_score(m):
    return dict(predicted=m['predicted'], target=m['target'], matched=len(m['matched']),
                paired_pixels=m['paired_pixels'], background_pixels=m['background_pixels'])


def dominance(p):
    return p['dominant'] if p['correct_if_dominant'] / max(1,p['pixels']) >= .7 else 0


def picture(image, before, after, truth, name, comparison='Guarded joins (cyan)'):
    ys,xs = np.nonzero(truth)
    box = (max(0,int(xs.min())-14), max(0,int(ys.min())-14),
           min(image.width,int(xs.max())+15), min(image.height,int(ys.max())+15))
    ref = np.array(image)
    edge = np.zeros(truth.shape,bool)
    edge[1:] |= truth[1:] != truth[:-1]
    edge[:,1:] |= truth[:,1:] != truth[:,:-1]
    ref[edge] = [42,255,220]
    pictures = [image]
    previous_ids = {p['id'] for p in before}
    for parts in (before,after):
        layer = image.copy()
        for p in parts:
            draw_outline(layer,p,'#59ecff' if p['id'] not in previous_ids else '#ff952f', 2)
        pictures.append(layer)
    pictures.append(Image.fromarray(ref))
    width,height = box[2]-box[0], box[3]-box[1]
    panel = Image.new('RGB',(2*width,2*(height+25)),'#192730')
    pen = ImageDraw.Draw(panel)
    for i,(pic,label) in enumerate(zip(pictures,('Source image','Current surface outlines',comparison,'Reference visible boundaries'))):
        x,y = i%2*width,i//2*(height+25)
        panel.paste(pic.crop(box),(x,y+25))
        pen.text((x+6,y+5),label,fill='white')
    panel.save(OUT/(name+'.png'))


def frame(job):
    split, src, _, _ = job
    key = src['id']
    image, parts, _, _ = viewer.structure_stages(job,baseline()[key])
    rgb = np.asarray(image)[1::3,1::3].astype(np.float32)
    masks = [native_mask(p,image.size)[1::3,1::3].copy() for p in parts]
    candidates = seam_features(rgb,masks,parts)
    predicted = {m:apply_pairs(parts,candidates,image.size,m) for m in MODES}
    labels, rays = visible_faces(split,src)
    truth,concrete = label_maps(labels,rays)
    before = metrics(parts,truth,image.size)
    pieces = Counter(dominance(p) for p in before['parts'])
    pieces.pop(0,None)
    row = dict(id=key,split=split,rgb_sha256=src['files']['rgb.png'],baseline=surface_score(before),
               fragmented_surfaces=sum(n>1 for n in pieces.values()),
               extra_fragments=sum(n-1 for n in pieces.values()),shade=shade_audit(image,rays,concrete))
    for mode,(revised,joins) in predicted.items():
        tally = Counter()
        records = []
        after = metrics(revised,truth,image.size) if joins else before
        for i,j in joins:
            pa,pb = [before['parts'][k] for k in (i,j)]
            a,b = dominance(pa),dominance(pb)
            typ = ('same_surface' if a==b else 'across_surfaces') if a and b else 'mixed_or_background'
            tally[typ] += 1
            records.append([parts[i]['id'],parts[j]['id'],typ])
        row[mode] = dict(surface_score(after),join_types=dict(tally),joins=records)
    if 'legacy-light_gray_concrete-' in key and any(token in key for token in FOCUS):
        picture(image,parts,predicted['guarded'][0],truth,key.replace('legacy-light_gray_concrete-',''))
    return row


def summarize(rows):
    groups = {}
    for group, rr in [('fixed',[r for r in rows if r['split'] in suite.framed.EVALUATION]),
                      ('other',[r for r in rows if r['split'] not in suite.framed.EVALUATION])]:
        def total(items):
            counts = Counter()
            for d in items:
                counts.update(d)
            return dict(counts)
        data = dict(frames=len(rr),shade=total(r['shade'] for r in rr),
                    fragmented_surfaces=sum(r['fragmented_surfaces'] for r in rr),
                    extra_fragments=sum(r['extra_fragments'] for r in rr))
        for mode in ('baseline',*MODES):
            score = total({k:v for k,v in r[mode].items() if k in r['baseline']} for r in rr)
            score['f1'] = 2*score['matched'] / (score['predicted']+score['target'])
            if mode in MODES:
                score['join_types'] = total(r[mode]['join_types'] for r in rr)
                score['changed_views'] = sum(bool(r[mode]['joins']) for r in rr)
                for k in ('matched','paired_pixels'):
                    score[k+'_gain_views'] = sum(r[mode][k] > r['baseline'][k] for r in rr)
                    score[k+'_loss_views'] = sum(r[mode][k] < r['baseline'][k] for r in rr)
            data[mode] = score
        groups[group] = data
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--focus',action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    jobs = list(suite.jobs([*suite.framed.SIZES,'shelter']))
    if args.focus:
        jobs = [j for j in jobs if 'legacy-light_gray_concrete-' in j[1]['id'] and any(t in j[1]['id'] for t in FOCUS)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(frame,jobs,chunksize=1))
    name = 'focus-' if args.focus else ''
    save(OUT/(name+'cases.json'),rows)
    save(OUT/(name+'summary.json'),summarize(rows))


if __name__ == '__main__':
    main()
