"""Audit the visible support for frozen placement hits and misses in the fixed structure suite."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font


AREA_THRESHOLDS = (4, 16, 32, 64, 128)
PATCH_SIDES = (2, 4, 8, 16)
EXAMPLES = ('shape-benchmark-open-26.3--train-canopy-00-right',
            'shape-benchmark-26.3--train-canopy-04-left', 'legacy-light_gray_concrete-dev-shopfront-plaster-low-left')


def square_support(ids, count):
    """Largest tested axis-aligned square of identical native first-hit IDs for each region."""
    current = ids
    result = np.ones(count + 1, dtype=np.uint8)
    for side in PATCH_SIDES:
        offset = side // 2
        a, b = current[:-offset, :-offset], current[offset:, :-offset]
        c, d = current[:-offset, offset:], current[offset:, offset:]
        current = np.where((a == b) & (a == c) & (a == d), a, 0)
        result[np.flatnonzero(np.bincount(current.ravel()))] = side
    return result


def masks(labels, ids):
    faces, blocks = labels['faces'], labels['blocks']
    n = len(blocks)
    face_pixels = np.bincount(ids.ravel(), minlength=len(faces) + 1)
    support = np.zeros((n + 1, 3), dtype=np.int64)  # top, underside and all vertical faces
    by_face = np.zeros(len(faces) + 1, dtype=np.int32)
    surface_lut = np.zeros_like(by_face)
    for face in faces:
        y = face['normal_world'][1]
        axis = 0 if y > 0 else 1 if y < 0 else 2
        support[face['block'], axis] += face_pixels[face['id']]
        by_face[face['id']] = face['block']
        surface_lut[face['id']] = face['surface_id']
        assert face['visible_pixels'] == face_pixels[face['id']]
    surface_size = square_support(surface_lut[ids], int(surface_lut.max()))
    on_surface = np.zeros(n + 1, dtype=np.uint8)
    for face in faces:
        if face['normal_world'][1] > 0:
            i = face['block']
            on_surface[i] = max(on_surface[i], surface_size[face['surface_id']])
    # ID zero is background. All evaluated blocks have at least one first-hit pixel.
    assert np.all(support[1:].sum(1) > 0)
    return support[1:], on_surface[1:], by_face


def counts(mask, hit):
    return dict(target=int(mask.sum()), correct=int(np.count_nonzero(mask & hit)))


def one_block_moves(extras, missing, axes):
    """Maximum distinct extras that could move one grid step into a distinct missed target."""
    offsets = [tuple(sign if i == a else 0 for i in range(3)) for a in axes for sign in (-1, 1)]
    candidates = {cell: [p for dx,dy,dz in offsets
                         if (p := (cell[0]+dx,cell[1]+dy,cell[2]+dz)) in missing] for cell in extras}
    owner = {}

    def place(cell, visited):
        for dest in candidates[cell]:
            if dest in visited:
                continue
            visited.add(dest)
            if dest not in owner or place(owner[dest],visited):
                owner[dest] = cell
                return True
        return False

    return sum(place(cell, set()) for cell in sorted(candidates))


def score(job):
    split, frame, prior, row, run, out = job
    image, _, _, _, _, labels, ids, truth, _, placed, _ = read_view(run, split, frame, prior, row)
    assert len(truth) == len(labels['blocks'])
    hit = np.asarray([tuple(b['xyz_world']) in placed for b in labels['blocks']])
    assert all(b['id'] == i for i, b in enumerate(labels['blocks'], 1))
    support, on_surface, owners = masks(labels, ids)
    top, bottom, walls = support.T
    area = support.sum(1)
    top_only = (top > 0) & (walls == 0) & (bottom == 0)
    horizontal_only = walls == 0
    groups = dict(any_vertical=walls > 0, top_only=top_only, underside_only=(bottom > 0) & horizontal_only,
                  vertical_only=(walls > 0) & (top == 0) & (bottom == 0),
                  mixed_vertical_horizontal=(walls > 0) & ((top + bottom) > 0))
    assert np.all((top == 0) | (bottom == 0))
    optional = dict(strict=np.zeros(len(hit), dtype=bool), any_top_only=top_only, facade_only=horizontal_only)
    for limit in AREA_THRESHOLDS:
        groups[f'all_faces_under_{limit}px'] = area < limit
        groups[f'vertical_evidence_at_least_{limit}px'] = walls >= limit
        optional[f'top_block_under_{limit}px'] = top_only & (area < limit)
    for side in PATCH_SIDES:
        optional[f'top_surface_lacks_{side}x{side}'] = top_only & (on_surface < side)
    metrics = row['relative_geometry']
    assert len(placed - truth) == metrics['extras'] and len(truth) == metrics['target']
    assert int(hit.sum()) == metrics['correct']
    extras = placed - truth
    missing = truth - placed
    moves = {name:one_block_moves(extras,missing,axes) for name,axes in
             [('any_axis',(0,1,2)),('horizontal_axes',(0,2)),('x_only',(0,)),('y_only',(1,)),('z_only',(2,))]}
    wall_missing = {tuple(b['xyz_world']) for i,b in enumerate(labels['blocks']) if not hit[i] and walls[i]}
    moves['any_axis_into_vertical_evidence'] = one_block_moves(extras, wall_missing, (0,1,2))
    if frame['id'] in EXAMPLES:
        mask = optional['top_surface_lacks_8x8']
        # Blue for other horizontal misses, orange for vertical misses, pink for optional slivers.
        palette = np.full((len(hit) + 1, 3), [62, 181, 128], dtype=np.uint8)
        palette[1:][~hit & horizontal_only] = [46, 146, 248]
        palette[1:][~hit & ~horizontal_only] = [251, 145, 48]
        palette[1:][~hit & mask] = [255, 20, 157]
        fg = ids != 0
        rgb = np.asarray(image).copy()
        rgb[fg] = .28 * rgb[fg] + .72 * palette[owners[ids[fg]]]
        box = Image.fromarray(fg).getbbox()
        l,t,r,b = box
        crop = (max(0,l-15), max(0,t-15), min(image.width,r+15), min(image.height,b+15))
        overlay = Image.fromarray(rgb)
        # Enlarge an actual missed reference pixel chosen near the center of the missed roof.
        candidates = mask & ~hit
        if not candidates.any():
            candidates = ~hit
        if not candidates.any():
            candidates = hit
        lut = np.r_[False, candidates]
        yy,xx = np.nonzero(lut[owners[ids]])
        center = np.argmin((xx-np.median(xx))**2+(yy-np.median(yy))**2)
        xc,yc = int(xx[center]),int(yy[center])
        close = overlay.crop((xc-50,yc-36,xc+51,yc+37)).resize((303,219),Image.Resampling.NEAREST)
        tiles = [image, image.crop(crop), overlay.crop(crop), close]
        page = Image.new('RGB',(1280,370),'#162934')
        for i,pic in enumerate(tiles):
            thumb=ImageOps.contain(pic,(309,243),Image.Resampling.LANCZOS) if i<3 else pic
            page.paste(thumb,(i*320+(320-thumb.width)//2,53+(243-thumb.height)//2))
        draw=ImageDraw.Draw(page)
        draw.text((13,8), frame['id'].split('--')[-1]+'     '+str(len(missing))+' missed visible blocks',font=font(25),fill='white')
        detail_label = ('Missed detail' if missing else 'Recovered detail')+' at 3× native pixels'
        for i,caption in enumerate(('Complete input frame','Building detail','Scored reference blocks',detail_label)):
            draw.text((i*320+10,300),caption,font=font(17),fill='white')
        draw.text((10,339),'Green  recovered     Orange  missed with a visible wall face     Blue  other horizontal-only miss     Pink  top-only miss on a surface lacking an 8 × 8 patch',font=font(16),fill='white')
        page.save(out/(frame['id']+'.jpg'),quality=93)
    return dict(split=split,id=frame['id'],size=image.size,translation=row['translation'],
                rgb_sha256=row['rgb_sha256'],pack_sha256=row['pack_sha256'],
                target=len(hit),correct=int(hit.sum()),extras=len(extras),one_block_moves=moves,
                optional={name:counts(mask,hit) for name,mask in optional.items()},
                support={name:counts(mask,hit) for name,mask in groups.items()})


def rates(correct, missed, extras):
    return dict(correct=correct, missed=missed, extras=extras, f1=2*correct/(2*correct+missed+extras)
                if correct or missed or extras else None)


def summarize(rows):
    correct = sum(r['correct'] for r in rows)
    extras = sum(r['extras'] for r in rows)
    target = sum(r['target'] for r in rows)
    result = dict(views=len(rows), baseline=rates(correct, target-correct, extras), rules={}, support={})
    for key in rows[0]['optional']:
        optional = sum(r['optional'][key]['target'] for r in rows)
        matched = sum(r['optional'][key]['correct'] for r in rows)
        missed = optional - matched
        result['rules'][key] = dict(optional_target=optional, optional_correct=matched, optional_missed=missed,
            scored=rates(correct-matched, target-correct-missed, extras),
            strict_if_all_optional_misses_recovered=rates(correct+missed, target-correct-missed, extras))
    for key in rows[0]['support']:
        total = sum(r['support'][key]['target'] for r in rows)
        got = sum(r['support'][key]['correct'] for r in rows)
        result['support'][key] = dict(target=total, correct=got, missed=total-got)
    result['one_block_moves'] = {}
    for key in rows[0]['one_block_moves']:
        moves = sum(r['one_block_moves'][key] for r in rows)
        result['one_block_moves'][key] = dict(moves=moves,
            strict_if_all_moves_correct=rates(correct+moves, target-correct-moves, extras-moves))
    return result


def main(args):
    proof = read(framed.REVIEW/'roof-height-01/verification.json')
    source = read(args.run/'summary.json')
    records = read(checked(args.run/'cases.json', source['cases_sha256']))
    is_baseline = source['cases_sha256'] == proof['cases_sha256']
    if not is_baseline and args.out.resolve() == (framed.REVIEW/'facade-visibility-01').resolve():
        raise ValueError('Choose a separate --out directory for the candidate audit.')
    index = {(r['split'],r['id']):r for r in records}
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s,f,p,index[s,f['id']],args.run,args.out) for s,f,p in framed.cases(framed.EVALUATION)]
    assert len(jobs) == 104
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(score, jobs))
    totals = summarize(rows)
    observed = tuple(totals['baseline'][k] for k in ('correct','missed','extras'))
    assert observed == tuple(source['groups']['fixed'][k] for k in ('correct','missed','extras'))
    if is_baseline:
        assert observed == (20462, 3240, 1157)
    save(args.out/'frames.json',rows)
    totals.update(reference_default_revision='1c2f77d', estimator_source_revision=source['revision'],
                  method=source.get('candidate'), scored_cases_sha256=source['cases_sha256'],
                  frame_evidence_sha256=digest(args.out/'frames.json'),
                  resolutions=dict(Counter('x'.join(map(str,r['size'])) for r in rows)))
    save(args.out/'summary.json',totals)
    sheet = Image.new('RGB',(1280,len(EXAMPLES)*370))
    for i,key in enumerate(EXAMPLES):
        with Image.open(args.out/(key+'.jpg')) as panel:
            sheet.paste(panel,(0,370*i))
    sheet.save(args.out/'facade-examples.jpg',quality=94)
    print(totals['baseline'], totals['rules']['top_surface_lacks_4x4'], totals['rules']['facade_only'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('../roof-height-round/runs/production'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/facade-visibility-01'))
    parser.add_argument('--workers',type=int,default=4)
    main(parser.parse_args())
