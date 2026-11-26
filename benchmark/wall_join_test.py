"""Test pairwise joins on the current Surface Finder, preserving every original mask pixel."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageFilter

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.exterior_set import Capture
from benchmark.mask_metrics import footprint, measure
from benchmark.surface_reference import recorded
from engine.normal_regions import _patch
from engine.surface_model import canonical
from runtime.mode import image_only


BASE = 'single-wall-depth-01'
DEPTH = 'single-wall-v2-fields-01'
SAM = 'mobile-regions-01'
SOURCE_METHOD = 'single-v2'
SETTINGS = dict(min_samples=40, max_normal_y=.5, normal_angle_deg=12,
                min_model_quality=.70, min_model_coverage=.78, min_iou_gain=.08,
                min_distinct_prompts=2, max_depth_at_centroids=.060,
                max_rgb_quantile_distance=22, max_area_overlap=.02)
# Two operating points of the same rule. Selection is frozen on the two development layouts.
PRESETS = dict(conservative=dict(min_union_iou=.58, max_depth_at_midpoint=.010),
               moderate=dict(min_union_iou=.45, max_depth_at_midpoint=.022))


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Changed input {path}')
    return path


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def plane(x, y, z):
    """Estimate affine inverse depth on a perspective plane; reduce the effect of boundary noise."""
    step = max(1, len(x) // 700)
    a = np.column_stack((x[::step], y[::step], np.ones(len(x[::step]))))
    values = z[::step]
    if np.linalg.matrix_rank(a) < 3:
        return None
    coef = np.linalg.lstsq(a, values, rcond=None)[0]
    for _ in range(3):
        error = np.abs(a @ coef - values)
        keep = error <= np.percentile(error, 75)
        if np.linalg.matrix_rank(a[keep]) < 3:
            return None
        coef = np.linalg.lstsq(a[keep], values[keep], rcond=None)[0]
    return coef


def pair_evidence(image, parts, depth, proposals, records):
    """Use predictions and RGB only. A MobileSAM prompt corroborates a pair without adding pixels."""
    low, high = np.percentile(depth, [2, 98])
    if high <= low:
        return []
    z = (depth - low) / (high - low)
    rgb = np.asarray(image.convert('RGB').resize(z.shape[::-1], Image.Resampling.BOX))
    retain = [i for i, r in enumerate(records) if r['quality'] >= SETTINGS['min_model_quality']]
    if not retain:
        return []
    proposals = proposals[retain]
    areas = np.array([records[i]['area'] for i in retain])
    wall = []
    for i, part in enumerate(parts):
        normal = np.asarray(part['normal_camera'])
        if abs(normal[1]) > SETTINGS['max_normal_y']:
            continue
        mask = footprint(part, z.shape, image.size, True)
        y, x = np.nonzero(mask)
        if len(x) < SETTINGS['min_samples']:
            continue
        coef = plane(x, y, z[y, x])
        if coef is None:
            continue
        hits = proposals[:, y, x].sum(axis=1)
        colors = np.percentile(rgb[y, x], (25, 50, 75), axis=0)
        wall.append(dict(i=i, mask=mask, area=len(x), center=np.array([x.mean(), y.mean(), 1.]),
                         normal=normal, plane=coef, colors=colors, hits=hits))
    output = []
    for index, a in enumerate(wall):
        for b in wall[index + 1:]:
            p, q = parts[a['i']], parts[b['i']]
            if p['support']['size'] != q['support']['size']:
                continue
            if np.dot(a['normal'], b['normal']) < np.cos(np.deg2rad(SETTINGS['normal_angle_deg'])):
                continue
            color_error = np.abs(a['colors'] - b['colors']).mean()
            if color_error > SETTINGS['max_rgb_quantile_distance']:
                continue
            offsets = (a['plane'] - b['plane']) @ np.column_stack([a['center'], b['center']])
            midpoint = abs(float(offsets.mean()))
            if max(abs(offsets)) > SETTINGS['max_depth_at_centroids'] or midpoint > max(
                    s['max_depth_at_midpoint'] for s in PRESETS.values()):
                continue
            contact = a['mask'] & b['mask']
            shared = int(contact.sum())
            if shared > SETTINGS['max_area_overlap'] * min(a['area'], b['area']):
                continue
            hits = a['hits'] + b['hits']
            if shared:
                hits = hits.astype(int) - proposals[:, contact].sum(axis=1).astype(int)
            union_area = a['area'] + b['area'] - shared
            union_iou = hits / (areas + union_area - hits)
            best_single = np.maximum(a['hits'] / (areas + a['area'] - a['hits']),
                                     b['hits'] / (areas + b['area'] - b['hits']))
            keep = ((a['hits'] >= a['area'] * SETTINGS['min_model_coverage']) &
                    (b['hits'] >= b['area'] * SETTINGS['min_model_coverage']) &
                    (union_iou >= min(s['min_union_iou'] for s in PRESETS.values())) &
                    (union_iou - best_single >= SETTINGS['min_iou_gain']))
            # Multiple slots of the same prompt do not provide independent support.
            prompts = {}
            for j in np.flatnonzero(keep):
                k = retain[j]
                point = tuple(records[k]['point'])
                item = (float(union_iou[j]), k)
                if point not in prompts or item > prompts[point]:
                    prompts[point] = item
            votes = sorted(prompts.values(), reverse=True)
            if len(votes) < SETTINGS['min_distinct_prompts']:
                continue
            output.append(dict(i=a['i'], j=b['i'], first=p['id'], second=q['id'],
                               midpoint=midpoint, centroid_errors=offsets.tolist(), rgb_distance=float(color_error),
                               votes=votes, confidence=votes[SETTINGS['min_distinct_prompts']-1][0]))
    return output


def unite(image, first, second):
    """Coalesce overlapping or touching runs only. A doorway or an occluder keeps its exact shape."""
    size = first['support']['size']
    union = []
    for y, lo, hi in sorted(first['support']['runs'] + second['support']['runs']):
        if union and union[-1][0] == y and lo <= union[-1][2]:
            union[-1][2] = max(hi, union[-1][2])
        else:
            union.append([y, lo, hi])
    members = [first, second]
    weights = np.asarray([p['support_area'] for p in members], dtype=float)
    normal = np.average([p['normal_camera'] for p in members], weights=weights, axis=0)
    normal /= np.linalg.norm(normal)
    depths = [p.get('relative_inverse_depth') for p in members]
    result = dict(id=f"join-{first['id']}__{second['id']}", joined_from=[p['id'] for p in members],
                  normal_camera=normal.tolist(), support=dict(size=list(size), runs=union),
                  support_area=sum(b-a for _, a, b in union), geometry_status='mask only',
                  relative_inverse_depth=float(np.average(depths, weights=weights)) if None not in depths else None)
    gray = np.asarray(image.convert('L').resize(size, Image.Resampling.BOX)
                      .filter(ImageFilter.GaussianBlur(.9)), np.float32)[..., None]
    chart = _patch(union, gray, np.asarray(image.size)/size, size, normal)
    if chart:
        result.update(quad=canonical(chart['quad']), geometry_status='mask with sampling chart')
    return result


def join(image, parts, evidence, settings):
    """Pair the strongest corroborated pieces first, with at most one join per input."""
    used, merged, events = set(), {}, []
    for e in sorted(evidence, key=lambda v: (-v['confidence'], v['midpoint'], v['i'], v['j'])):
        i, j = e['i'], e['j']
        if (i in used or j in used or e['confidence'] < settings['min_union_iou']
                or e['midpoint'] > settings['max_depth_at_midpoint']):
            continue
        merged[i] = unite(image, parts[i], parts[j])
        used.update((i,j))
        events.append(dict(parts=[e['first'],e['second']], result=merged[i]['id'],
                           depth_difference=e['midpoint'], model_votes=e['votes']))
    return [merged[i] if i in merged else p for i, p in enumerate(parts) if i not in used or i in merged], events


def capture_index(data):
    captures = {}
    for folder in ('camera-round-01', 'camera-gray-concrete-v1', 'surface-confirm-v1', 'normal-confirm-v1'):
        root = data / folder
        index = read(root/'index.json')
        for frame in index['frames']:
            case = Capture(frame.get('floor', index.get('floor', 'smooth_stone')), root, frame)
            captures[f'{case.floor}/{case.id}'] = case
    return captures


def frame_inputs(reviews, manifests, row, case):
    sha = row['sha256']
    with Image.open(checked(case.path('image'), sha)) as photo:
        image = photo.convert('RGB')
    def path(source, name):
        return checked(reviews/source/name, manifests[source]['artifacts'][name])
    parts = read_gzip(path(BASE, row['predictions']))[SOURCE_METHOD]
    with np.load(path(DEPTH, f'fields/{sha}.npz'), allow_pickle=False) as data:
        depth = data['inverse_depth']
    with np.load(path(SAM, f'proposals/{sha}.npz'), allow_pickle=False) as data:
        proposals = np.unpackbits(data['packed'], axis=2, count=int(data['shape'][2])).astype(bool)
    records = read_gzip(path(SAM, f'proposals/{sha}.json.gz'))['candidates']
    if depth.shape != proposals.shape[1:] or len(records) != len(proposals) or not np.isfinite(depth).all():
        raise ValueError(f'Invalid numerical fields for {row["key"]}')
    return image, parts, depth, proposals, records


@image_only()
def run(source, out):
    reviews, data = source/'benchmark/reviews', source/'benchmark/data'
    manifests = {name:read(reviews/name/'manifest.json') for name in (BASE, DEPTH, SAM)}
    base = manifests[BASE]
    rows = read(checked(reviews/BASE/'cases.json', base['artifacts']['cases.json']))
    captures = capture_index(data)
    archive = checked(data/'visible-geometry-v1.zip', base['reference_sha256'])
    if (len(rows) != 56 or captures.keys() != {r['key'] for r in rows} or
            any(m['fixtures'] != base['fixtures'] for m in manifests.values())):
        raise ValueError('Wrong fixture set')
    out.mkdir(parents=True, exist_ok=False)
    dev = [r for r in rows if r['cohort'] == 'development']
    if len(dev) != 20 or {r['layout'] for r in dev} != {'shopfront','wing'}:
        raise ValueError('Wrong development split')
    grades = {setting:Counter() for setting in PRESETS}
    for row in dev:
        case = captures[row['key']]
        image, parts, *fields = frame_inputs(reviews, manifests, row, case)
        evidence = pair_evidence(image, parts, *fields)
        _, ref = recorded(case, archive)
        for name, preset in PRESETS.items():
            predicted, _ = join(image, parts, evidence, preset)
            score, _ = measure(predicted, ref, use_mask=True)
            grades[name].update({k:score[k] for k in ('matched','proposed','eligible','correct_pixels')})
    def priority(name):
        s = grades[name]
        return s['matched'], 2*s['matched']/(s['eligible']+s['proposed']), s['correct_pixels']
    selected = max(PRESETS, key=priority)
    save(out/'development-freeze.json', dict(selected=selected, settings=PRESETS, scores=grades,
         rule='Most development matches, then highest surface F1, then most correctly assigned pixels; conservative on ties.'))
    print(f'Frozen on development: {selected}, {grades}', flush=True)
    results, artifacts = [], {}
    for row in rows:
        case = captures[row['key']]
        image, parts, *fields = frame_inputs(reviews, manifests, row, case)
        start = perf_counter()
        evidence = pair_evidence(image, parts, *fields)
        predictions, events = join(image, parts, evidence, PRESETS[selected])
        seconds = perf_counter() - start
        name = 'predictions/' + row['key'] + '.json.gz'
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(predictions, separators=(',',':'), allow_nan=False).encode(), mtime=0))
        artifacts[name] = digest(path)
        results.append(dict(key=row['key'],sha256=row['sha256'],layout=row['layout'],cohort=row['cohort'],
                            predictions=name,prediction_sha256=artifacts[name],cached_processing_s=seconds,joins=events))
    save(out/'cases.json', results)
    for name in ('cases.json', 'development-freeze.json'):
        artifacts[name] = digest(out/name)
    save(out/'manifest.json', dict(run_id=out.name, method='coplanar-wall-join', baseline=SOURCE_METHOD,
        code=revision(), fixtures=base['fixtures'], version=base['version'], reference_sha256=base['reference_sha256'],
        frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in results],
        settings=dict(**SETTINGS, **PRESETS[selected]), frozen='development-freeze.json',
        sources={name:digest(reviews/name/'manifest.json') for name in manifests},
        inference='Frozen input inference, zero new model calls.',
        cached_processing_s=sum(r['cached_processing_s'] for r in results), artifacts=artifacts,
        reproduce='python -m benchmark.wall_join_test --source-root . --out runs/wall-join-replay'))
    print(f'{len(results)} views, {sum(len(r["joins"]) for r in results)} joins, {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    run(options.source_root.resolve(), options.out.resolve())
