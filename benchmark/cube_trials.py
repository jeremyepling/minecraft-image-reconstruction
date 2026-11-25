"""Compare perspective cube constraints against the previous full-frame outline."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.fixtures import Building, UP
from benchmark.geometry_metrics import covers_point, enriched, facing, truth_mask
from benchmark.line_form import inspect
from benchmark.line_form_trials import grade_panels, outline_score, score_flat_controls
from calibration.projection import camera_project


MODES = {'line_first': False, 'cube_turns': True}


def predictor_digests():
    source_root = Path(__file__).resolve().parents[1]
    files = ['benchmark/line_form.py','engine/cube_axes.py','engine/cube_contour.py',
             'engine/cube_faces.py','engine/line_relief.py','engine/line_outline.py',
             'engine/structural_lines.py']
    return {p:hashlib.sha256((source_root/p).read_text().encode()).hexdigest() for p in files}


def axis_error(axes, building, pose):
    """Evaluate fit directions at 27 world-box probes in the recorded view."""
    low, high = np.asarray(building.get('bounds', building.get('facade_bounds')),float)
    probes = low + (high-low)*np.array(list(product((.1,.5,.9),repeat=3)))
    projected = camera_project(probes,pose)
    truth = np.array([camera_project(probes+d,pose)-projected for d in np.eye(3)])
    actual = np.asarray(axes)
    estimated = actual[:,None,:2]-projected[None,:,:]*actual[:,None,2:]
    lengths = np.linalg.norm(estimated,axis=2)[:,None,:]*np.linalg.norm(truth,axis=2)[None,:,:]
    cosines = abs(np.einsum('ipd,jpd->ijp',estimated,truth))/np.maximum(1.e-15,lengths)
    angles = np.degrees(np.arccos(np.clip(cosines,0.,1.)))
    means = np.array([[np.mean(angles[i,j][lengths[i,j]>1.e-10]) for j in range(3)] for i in range(3)])
    best = min(permutations(range(3)),key=lambda order: np.sum(means[order,range(3)]))
    return [round(float(angle),3) for angle in means[best,range(3)]]


def canopy_gaps(building, pose, mask):
    """Check clear voxel-center sightlines against the predicted silhouette."""
    projected = np.array([camera_project(p['vertices'], pose) for p in building['planes']
                          if facing(p, pose)])
    filled = eligible = 0
    for probe in enriched(building)['probes']:
        if probe['role'] != 'pillar_gap':
            continue
        point = camera_project(probe['point'], pose)
        if np.any(point < 0) or np.any(point >= mask.size) or covers_point(projected, point):
            continue
        eligible += 1
        filled += int(mask.getpixel(tuple(point.astype(int))) != 0)
    return dict(eligible=eligible, filled=filled)


def evaluate(args):
    root, view, building, group = args
    stem = root/'captures'/view['id']
    image_file = stem.with_suffix('.png')
    with Image.open(image_file) as frame:
        image = frame.convert('RGB')
    relief = view['kind'] == 'relief'
    predicted = {name:inspect(image, local=relief, cube_turns=enabled)
                 for name, enabled in MODES.items()}
    # Scoring first receives poses and world readbacks after both predictions.
    pose = json.loads(stem.with_suffix('.json').read_text())
    if 'planes' not in building:
        verified = json.loads(stem.with_name(stem.name+'-blocks.json').read_text())
        world = Building(building['page'], view['split'], view['kind'], {})
        world.cells = {(v['x']-world.origin,v['y']-UP,v['z']):(v['block_id'],'solid')
                       for v in verified if v['block_id']!='air'}
        world.finish()
        building = dict(building, planes=world.data['planes'])
    truth = truth_mask(building, pose, tuple(n//4 for n in image.size), 4)
    scores = {}
    for name, (result, mask) in predicted.items():
        score = dict(outline=outline_score(mask, truth))
        if name == 'cube_turns':
            axes = result['cube_turns']['axes']
            score['axes_fitted'] = len(axes) == 3
            contour = result['cube_turns']['contour']
            score['joined_turns'] = len(contour['turns'])
            score['changed_pixels'] = contour['changed_pixels']
            if score['axes_fitted']:
                score['mean_error_degrees_by_world_axis'] = axis_error(axes,building,pose)
        if relief:
            score['panels'] = grade_panels(result['panels'], building['regions'], pose)
        if view['kind'] == 'canopy':
            score['gap_probes'] = canopy_gaps(building, pose, mask)
        scores[name] = score
    return (view['id'],dict(group=group, sha256=hashlib.sha256(image_file.read_bytes()).hexdigest(),
                            angle=view['angle'], scores=scores))


def plan(roots, open_root, split):
    tasks = []
    for root, group in [(r,'oblique') for r in roots] + (
            [(open_root,'open_canopies')] if open_root else []):
        buildings = {b['id']:b for b in json.loads((root/'buildings.json').read_text())}
        picked = set()
        for v in json.loads((root/'views.json').read_text()):
            if v['split'] != split:
                continue
            if group == 'oblique':
                key = (v['kind'],v['angle'])
                if key in picked or not any(d in v['angle'] for d in ('left','right')):
                    continue
                picked.add(key)
            elif v['kind'] != 'canopy':
                continue
            tasks.append((root,v,buildings[v['building_id']],group))
    if not tasks or len({t[1]['id'] for t in tasks}) != len(tasks):
        raise ValueError('Expected unique nonempty view IDs across fixtures')
    return tasks


def summarize(views):
    groups = {}
    for group in sorted({v['group'] for v in views.values()}):
        cases = [v for v in views.values() if v['group']==group]
        methods = {}
        for method in MODES:
            items = [v['scores'][method] for v in cases]
            summary = {key:round(float(np.mean([r['outline'][key] for r in items])), 5)
                       for key in ('iou','boundary_f1')}
            if method == 'cube_turns':
                summary['axes_fitted'] = sum(r['axes_fitted'] for r in items)
                summary['outlines_changed'] = sum(r['changed_pixels']>0 for r in items)
                summary['joined_turns'] = sum(r['joined_turns'] for r in items)
                errors = [r['mean_error_degrees_by_world_axis'] for r in items if r['axes_fitted']]
                if errors:
                    summary['mean_axis_error_degrees'] = round(float(np.mean(errors)),3)
            gaps = [r['gap_probes'] for r in items if 'gap_probes' in r]
            if gaps:
                summary['clear_sightlines'] = sum(r['eligible'] for r in gaps)
                summary['incorrectly_filled'] = sum(r['filled'] for r in gaps)
            panels = [r['panels'] for r in items if 'panels' in r]
            if panels:
                rows = [row for p in panels for row in p['rows']]
                extras = sum((Counter(p['extra']) for p in panels), Counter())
                summary['panels'] = dict(total=len(rows),
                    localized=sum(r['decision']!='missed' for r in rows),
                    correct=sum(r['decision']==r['actual'] for r in rows),
                    wrong=sum(r['decision'] not in (r['actual'],'unknown','missed') for r in rows),
                    unknown=sum(r['decision']=='unknown' for r in rows),
                    missed=sum(r['decision']=='missed' for r in rows),
                    by_actual={c:dict(Counter(r['decision'] for r in rows if r['actual']==c))
                               for c in ('recess','flat','protrusion')},
                    extra_proposals=dict(extras))
            methods[method] = summary
        groups[group] = dict(views=len(cases), methods=methods)
    return groups


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+', type=Path)
    p.add_argument('--split', required=True, choices=('dev','test'))
    p.add_argument('--open-root', type=Path, help='Additional open-canopy worlds, all views')
    p.add_argument('--flat-controls', type=Path)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    tasks = plan(args.roots, args.open_root, args.split)
    digests = predictor_digests()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        views = dict(pool.map(evaluate, tasks))
    report = dict(minecraft='26.3', split=args.split, baseline='f0634fc line-first', sources=digests,
                  summary=summarize(views), views=views)
    if args.flat_controls:
        report['flat_controls'] = {k:score_flat_controls(args.flat_controls,cube_turns=enabled)
                                   for k,enabled in MODES.items()}
    if predictor_digests() != digests:
        report['sources_changed_during_run'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
