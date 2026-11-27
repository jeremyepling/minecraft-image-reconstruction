"""Test a one-pass feedback link from final visible cube IDs to proposed surface joins."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from benchmark import concrete_shadow_boundaries as seams
from benchmark.texture_camera_eval import checked, read, save
from engine.projected_appearance import Scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmark/reviews/material-feedback-surfaces-01'
CHECKS = ('material_only', 'plane', 'plane_and_id')
MODES = (*CHECKS, 'guarded_plane_and_id')
ORACLES = ('true_material', 'true_surface')


@lru_cache(maxsize=1)
def sites():
    release = ROOT / 'benchmark/reviews/typed-structures-01'
    meta = read(release/'summary.json')
    return json.loads(gzip.decompress(checked(release/'sites.json.gz', meta['sites_sha256']).read_bytes()))


def dilation(mask):
    return np.array(Image.fromarray(mask.astype('uint8')).filter(ImageFilter.MaxFilter(9)), bool)


def dominant(values):
    unique, counts = np.unique(values, return_counts=True)
    best = counts.argmax()
    return int(unique[best]) if counts[best] >= .8 * len(values) else None


def observe(side, scene, material):
    # Read visible evidence within 12 source pixels of the shared interface.
    mask = np.flatnonzero(side.ravel())
    owner = scene.owner[mask]
    valid = (owner >= 0) & (material[owner] >= 0)
    if valid.sum() < max(12, .75*len(mask)) or len(np.unique(owner[valid])) < 2:
        return None
    cells = np.array(scene.cells)[owner[valid]]
    axes = scene.axes[mask[valid]]
    level = cells[np.arange(len(axes)), axes]
    level += scene.eye[axes] > level+1
    planes = level * 3 + axes
    return dominant(material[owner[valid]]), dominant(planes)


def decide(parts, color, scene, material, size):
    masks = [seams.native_mask(p, size)[::3, ::3] for p in parts]
    result = {m:[] for m in CHECKS}
    for i,j in color:
        a,b = masks[i], masks[j]
        left = observe(a & ~b & dilation(b), scene, material)
        right = observe(b & ~a & dilation(a), scene, material)
        if left is None or right is None:
            continue
        same_id = left[0] is not None and left[0] == right[0]
        coplanar = left[1] is not None and left[1] == right[1]
        for mode, accepted in zip(CHECKS, (same_id, coplanar, same_id and coplanar), strict=True):
            if accepted:
                result[mode].append((i,j))
    return result


def truth_seams(parts, size, pairs, labels, rays):
    # Deliberate oracle. Pixel ownership and material come from visible faces only.
    masks = [seams.native_mask(part,size)[::3,::3] for part in parts]
    faces = labels['faces']
    blocks = {b['id']:b['block_id'] for b in labels['blocks']}
    names = {name:i for i,name in enumerate(sorted(set(blocks.values())))}
    table = np.full((len(faces)+1,3),-1,dtype=int)
    for face in faces:
        b = face['block']
        table[face['id']] = b, names[blocks[b]], face['surface_id']
    evidence = table[rays[::3,::3]]
    accepted = {m:[] for m in ORACLES}
    def side(mask):
        data = evidence[mask]
        observed = data[data[:,0] >= 0]
        if len(observed) < max(12,.75*len(data)) or len(np.unique(observed[:,0])) < 2:
            return None
        return dominant(observed[:,1]), dominant(observed[:,2])
    for i,j in pairs:
        a,b = masks[i],masks[j]
        first = side(a & ~b & dilation(b))
        second = side(b & ~a & dilation(a))
        if first is None or second is None:
            continue
        for axis,mode in enumerate(ORACLES):
            if first[axis] is not None and first[axis] == second[axis]:
                accepted[mode].append((i,j))
    return accepted


def frame(task):
    job, output = task
    split, f, _, _ = job
    key = f['id']
    assigned = sites()[key]
    if assigned['source_hash'] != f['files']['rgb.png']:
        raise ValueError(f'Mismatched typed export for {key}')
    image, parts, _, camera = seams.viewer.structure_stages(job, seams.baseline()[key])
    rgb = np.asarray(image)[1::3, 1::3].astype(np.float32)
    masks = [seams.native_mask(p, image.size)[1::3, 1::3].copy() for p in parts]
    features = seams.seam_features(rgb, masks, parts)
    _, color = seams.apply_pairs(parts, features, image.size, 'color')
    _, guarded = seams.apply_pairs(parts, features, image.size, 'guarded')
    estimates = {tuple(v[:3]):v[3:] for v in assigned['rows']}
    proposed = {m:[] for m in MODES}
    if (color or guarded) and estimates and camera.get('P') is not None:
        scene = Scene(image, [], camera, sorted(estimates), step=3)
        codes = {v:n for n,v in enumerate(sorted({v for v,_ in estimates.values()}))}
        material = np.array([codes[estimates[c][0]] if estimates[c][1]!='fallback' else -1 for c in scene.cells])
        proposed.update(decide(parts,color,scene,material,image.size))
        if guarded:
            proposed['guarded_plane_and_id'] = decide(parts,guarded,scene,material,image.size)['plane_and_id']
    # The rule has finished before the reference face map is opened.
    labels, rays = seams.visible_faces(split, f)
    truth,_ = seams.label_maps(labels, rays)
    proposed.update(truth_seams(parts,image.size,color,labels,rays))
    before = seams.metrics(parts, truth, image.size)
    row = dict(id=key, split=split, rgb_sha256=assigned['source_hash'], pack_sha256=assigned['pack_sha256'],
               baseline=seams.surface_score(before))
    for mode,pairs in dict(color=color, **proposed).items():
        changed,_ = seams.apply_pairs(parts, [(0,i,j,False) for i,j in pairs], image.size, 'color')
        after = seams.metrics(changed, truth, image.size) if pairs else before
        types = Counter()
        records = []
        for i,j in pairs:
            a,b = (seams.dominance(before['parts'][k]) for k in (i,j))
            kind = ('same_surface' if a==b else 'across_surfaces') if a and b else 'mixed_or_background'
            types[kind] += 1
            records.append([parts[i]['id'],parts[j]['id'],kind])
        row[mode] = dict(seams.surface_score(after), joins=records, join_types=types)
        if mode=='plane_and_id' and key=='legacy-light_gray_concrete-holdout-courtyard-plaster-low-left':
            seams.OUT=output
            seams.picture(image,parts,changed,truth,'courtyard','Same material and projected plane')
    return row


def summarize(rows):
    out={}
    for mode in ('baseline','color',*MODES,*ORACLES):
        totals=Counter()
        kinds=Counter()
        for r in rows:
            totals.update({k:v for k,v in r[mode].items() if k in r['baseline']})
            kinds.update(r[mode].get('join_types',{}))
        totals['f1']=2*totals['matched']/(totals['predicted']+totals['target'])
        out[mode]=dict(totals, joins=dict(kinds), views=len(rows))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    jobs=list(seams.suite.jobs([*seams.suite.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(frame,[(j,args.out) for j in jobs]))
    save(args.out/'cases.json',rows)
    previous=read(ROOT/'benchmark/reviews/concrete-shadow-boundaries-01/cases.json')
    assert len(rows)==len(previous)==204
    prior={r['id']:r for r in previous}
    for r in rows:
        assert r['baseline']==prior[r['id']]['baseline'] and r['color']==prior[r['id']]['color'],r['id']
    groups={'development':[r for r in rows if r['split']=='train'],
            'fixed':[r for r in rows if r['split'] in seams.suite.framed.EVALUATION],
            'additional':[r for r in rows if r['split'] not in (*seams.suite.framed.EVALUATION,'train')]}
    save(args.out/'summary.json',{k:summarize(v) for k,v in groups.items()})


if __name__=='__main__':
    main()
