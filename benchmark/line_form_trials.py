"""Score full-frame outlines and independently score signed panel proposals."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from benchmark.entrances import inspect as inspect_openings
from benchmark.fixtures import Building, UP
from benchmark.geometry_metrics import truth_mask
from benchmark.textures import area, intersection
from calibration.projection import camera_project
from engine.geometry import project, transform
from engine.line_outline import contour, group_lines, silhouette
from engine.remap import structure_mask
from engine.structural_lines import detect


def outline_score(mask, actual):
    estimated = mask.resize((actual.shape[1], actual.shape[0]), Image.Resampling.NEAREST)
    pixels = np.asarray(estimated) > 0
    a = contour(estimated)
    b = contour(Image.fromarray(np.uint8(actual)*255))
    def nearby(boundary):
        return np.asarray(Image.fromarray(np.uint8(boundary)).filter(ImageFilter.MaxFilter(5))) > 0
    precision = (a & nearby(b)).sum()/max(1, a.sum())
    recall = (b & nearby(a)).sum()/max(1, b.sum())
    return dict(iou=round(float((pixels & actual).sum()/max(1, (pixels | actual).sum())), 4),
                boundary_f1=round(float(2*precision*recall/max(1e-8, precision+recall)), 4))


def grade_panels(predicted, regions, pose):
    possible = []
    for i, region in enumerate(regions):
        quad = region['cap'] if region['kind']=='protrusion' else region['footprint']
        inv = np.linalg.inv(transform(camera_project(quad, pose)))
        for j, panel in enumerate(predicted):
            rect = project(inv, panel['quad'])
            overlap = intersection(rect, (0, 0))
            iou = overlap / max(1e-8, 1 + area(rect) - overlap)
            if iou >= .5:
                possible.append((iou, i, j))
    paired, used = {}, set()
    for overlap, i, j in sorted(possible, reverse=True):
        if i not in paired and j not in used:
            paired[i] = (j, overlap)
            used.add(j)
    rows = []
    for i, region in enumerate(regions):
        j, overlap = paired.get(i, (None, 0))
        rows.append(dict(target=region['id'], actual=region['kind'],
                         decision=predicted[j]['label'] if j is not None else 'missed',
                         iou=round(float(overlap), 3)))
    extra = Counter(panel['label'] for i, panel in enumerate(predicted) if i not in used)
    return dict(rows=rows, extra=extra)


def score_flat_controls(root, *, cube_turns=False):
    """Count relief calls contained on verified flush wall, excluding the canopy."""
    from benchmark.line_form import inspect
    buildings = {b['id']:b for b in json.loads((root/'buildings.json').read_text())}
    cells = json.loads((root/'layout.json').read_text())['blocks']
    counts = {}
    for view in json.loads((root/'views.json').read_text()):
        if view['kind'] not in ('dark-inlay', 'physical-shade'):
            continue
        path = root/'captures'/(view['id']+'.png')
        with Image.open(path) as image:
            result, _ = inspect(image, cube_turns=cube_turns)
        b = buildings[view['building_id']]
        pose = json.loads(path.with_suffix('.json').read_text())
        (x,y,_),(u,v,z) = b['facade_bounds']
        wall = camera_project([[x,v,z],[u,v,z],[u,y,z],[x,y,z]],pose)
        inv = np.linalg.inv(transform(wall))
        roof = np.array([[c['x'],c['y'],c['z']] for c in cells
                         if c['page']==view['page'] and c['role']=='shade_roof'])
        roof_box = None
        if len(roof):
            corners = list(product(*zip(roof.min(0),roof.max(0)+1)))
            pixels = camera_project(corners,pose)
            roof_box = (pixels.min(0),pixels.max(0))
        inside = Counter()
        for panel in result['panels']:
            q = np.asarray(panel['quad'])
            uv = project(inv,q)
            if np.any(uv<0) or np.any(uv>1):
                continue
            if roof_box and np.all(q.max(0)>=roof_box[0]) and np.all(q.min(0)<=roof_box[1]):
                continue
            inside[panel['label']] += 1
        counts[view['id']]=inside
    return counts


def score_view(args):
    root, view, building, mode = args
    path = root/'captures'/(view['id']+'.png')
    with Image.open(path) as im:
        image = im.convert('RGB')
    lines = detect(image)
    mask = silhouette(image, lines)
    predictions = {}
    if view['kind'] == 'relief' and mode == 'all':
        from engine.line_relief import analyze
        predictions['line_junctions'] = analyze(image, group_lines(lines, mask), mask)
        old = inspect_openings(path)
        predictions['opening_inspector'] = [
            dict(quad=r['rim'], label='unknown' if r['depth'] is None else
                 ('recess' if r['depth'] else 'flat')) for r in old['candidates']]
    # Read camera and world geometry only after the image-only predictions.
    pose = json.loads(path.with_suffix('.json').read_text())
    if 'planes' not in building:
        verified = json.loads(path.with_name(path.stem+'-blocks.json').read_text())
        world = Building(building['page'], view['split'], view['kind'], {})
        world.cells = {(v['x']-world.origin,v['y']-UP,v['z']):(v['block_id'],'solid')
                       for v in verified if v['block_id']!='air'}
        world.finish()
        building = dict(building, planes=world.data['planes'])
    actual = truth_mask(building, pose, tuple(n//4 for n in image.size), 4)
    masks = dict(previous=Image.fromarray(structure_mask(image).astype('uint8')*255),
                 without_lines=silhouette(image, []), line_first=mask)
    return view['id'], dict(outlines={k:outline_score(m, actual) for k,m in masks.items()},
                            panels={k:grade_panels(v, building['regions'], pose) for k,v in predictions.items()})


def run(roots, split, mode, jobs):
    tasks = []
    for root in roots:
        views = json.loads((root/'views.json').read_text())
        buildings = {b['id']:b for b in json.loads((root/'buildings.json').read_text())}
        picked = set()
        for view in views:
            if view['split']!=split or 'left' not in view['angle'] and 'right' not in view['angle']:
                continue
            key = view['kind'], view['angle']
            if key in picked:
                continue
            picked.add(key)
            tasks.append((root, view, buildings[view['building_id']], mode))
    if not tasks:
        raise ValueError('No eligible views')
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        views = dict(pool.map(score_view, tasks))
    means = {}
    for method in next(iter(views.values()))['outlines']:
        means[method] = {metric:round(float(np.mean([r['outlines'][method][metric] for r in views.values()])),4)
                         for metric in ('iou','boundary_f1')}
    return dict(split=split, outline_means=means, views=views)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+', type=Path)
    p.add_argument('--split', required=True, choices=('dev','test'))
    p.add_argument('--mode', default='all', choices=('all','outline'))
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--flat-controls', type=Path, help='Additional flat-inlay and shaded wall capture directory')
    p.add_argument('--out', type=Path, required=True)
    args=p.parse_args()
    report=run(args.roots,args.split,args.mode,args.jobs)
    if args.flat_controls:
        report['flat_controls']=score_flat_controls(args.flat_controls)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['outline_means'])
