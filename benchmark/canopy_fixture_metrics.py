"""Evaluate registered decoded blocks on authored roof faces and clear gap rays."""
import argparse
from collections import Counter
from math import floor
from pathlib import Path
from statistics import median

import numpy as np

from benchmark.fixtures import clear_sightline
from benchmark.texture_camera_eval import ROOT, checked, digest, read, save
from benchmark.visible_depth_transfer import rgb_path
from calibration.projection import camera_project
from training.visible_labels import visible_labels

PROBES = ROOT/'benchmark/reviews/canopy-fixtures-01/canopies.json'


def gap_width(cells, z, bottom, top):
    """Widest internal span at each measured post course, between decoded blocks."""
    spans = []
    for y in range(bottom, top):
        xs = sorted(x for x, row, plane in cells if plane == z and row == y)
        gaps = [b-a-1 for a,b in zip(xs,xs[1:]) if b-a > 1]
        if gaps:
            spans.append(max(gaps))
    return median(spans) if len(spans) >= 2 else None


def raster(pose, cells):
    states = dict.fromkeys(cells, dict(block_id='smooth_stone', state='Block{minecraft:smooth_stone}'))
    if not cells:
        return [], np.zeros(pose['image_size'][::-1], bool)
    try:
        labels, faces, _ = visible_labels(pose, states, (0,0,0))
    except ValueError as error:
        if not str(error).startswith('No visible building faces'):
            raise
        return [], np.zeros(pose['image_size'][::-1], bool)
    return labels['faces'], faces > 0


def score(f, cells, spec):
    rgb = rgb_path(f)
    pose = read(checked(rgb.with_suffix('.json'), f['files']['camera.json']))
    raw = read(checked(rgb.with_name(f['source_id']+'-blocks.json'), f['files']['world-readback.json']))
    occupied = {(b['x'], b['y'], b['z']) for b in raw if b['y'] > 64 and b['block_id'].split(':')[-1] != 'air'}
    reference, truth_mask = raster(pose, occupied)
    placed, build_mask = raster(pose, cells)
    owned = {(tuple(b['xyz_world']), b['direction']) for b in placed}
    counts = Counter()
    left, right, z, bottom, top = (spec[k] for k in ('left','right','front_z','bottom','roof_y'))
    # The authored canopy consists of one upper course, front posts and optional rear wall.
    for b in reference:
        point = tuple(b['xyz_world'])
        if point[1] == top:
            category = {1:'roof_top', 0:'roof_rim', -1:'roof_underside'}[b['normal_world'][1]]
        elif point[2] == z and point[0] in (left-1,right):
            category = 'front_post'
        else:
            continue
        counts[category+'_target'] += 1
        counts[category+'_recovered'] += int((point,b['direction']) in owned)
    rays = []
    for x in range(left,right):
        for y in range(bottom,top):
            p = [x+.5,y+.5,z+.5]
            xy = camera_project(p,pose)
            if np.any(xy < 0) or np.any(xy >= pose['image_size']):
                continue
            if not clear_sightline(pose['eye_world'],p,occupied):
                continue
            ix,iy = map(floor,xy)
            rays.append(dict(point=p, foreground_filled=not clear_sightline(pose['eye_world'],p,cells),
                             background=not bool(truth_mask[iy,ix]), projected_filled=bool(build_mask[iy,ix])))
    measured = gap_width(cells, z, bottom, top)
    assert gap_width(occupied, z, bottom, top) == right-left
    return dict(id=f['id'], faces=dict(counts), rays=rays, gap_target=right-left, gap_estimate=measured)


def summarize(cases):
    counts = Counter()
    for row in cases:
        counts.update(row['faces'])
    rays = [ray for row in cases for ray in row['rays']]
    errors = [abs(r['gap_estimate']-r['gap_target']) for r in cases if r['gap_estimate'] is not None]
    return dict(frames=len(cases), face_counts=dict(counts), clear_rays=len(rays),
                foreground_filled=sum(ray['foreground_filled'] for ray in rays),
                background_rays=sum(r['background'] for r in rays),
                background_filled=sum(r['background'] and r['projected_filled'] for r in rays),
                gap_mae=sum(errors)/len(errors) if errors else None, gap_measured=len(errors),
                gap_missing=len(cases)-len(errors))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    frames = read(args.inputs/'screen.json')['selected']
    specs = read(PROBES)
    output = []
    for frame in frames:
        key = frame['id']
        if key in specs:
            path = args.predictions/key/'cells.json'
            row = score(frame,set(map(tuple,read(path)['world'])),specs[key])
            output.append(dict(row,cells_sha256=digest(path)))
    result = summarize(output)
    save(args.out,dict(summary=result, cases=output))
    print(result)
