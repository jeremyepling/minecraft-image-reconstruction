"""Split existing vertical walls with agreeing, disjoint MobileSAM proposals."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from PIL import Image, ImageFilter

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.mask_metrics import aggregate, footprint, measure
from benchmark.mobile_region_probe import METHOD, WALLS
from benchmark.normal_round import groups
from benchmark.surface_metrics import DOWNSAMPLE
from benchmark.surface_reference import ARCHIVE, recorded
from engine.camera_lines import _runs
from engine.normal_regions import _local_mask, _patch
from engine.surface_model import canonical
from runtime.mode import image_only


SETTINGS = dict(maximum_vertical_normal_y=.5, minimum_containment=.6,
                minimum_parent_share=.1, minimum_samples=32,
                maximum_overlap_with_smaller=.05, minimum_explained=.8)


def choose_parts(parent, masks, records, selected):
    """Find a set of substantial parts explaining a parent, in prediction space only."""
    area = parent.sum()
    candidates = []
    for i in selected:
        hit = masks[i] & parent
        n = int(hit.sum())
        if n < max(SETTINGS['minimum_samples'], area * SETTINGS['minimum_parent_share']):
            continue
        if n > area * (1 - SETTINGS['minimum_parent_share']):
            continue
        if n >= records[i]['area'] * SETTINGS['minimum_containment']:
            candidates.append((i, n, hit))
    candidates.sort(key=lambda entry: (-entry[1], -records[entry[0]]['quality'], entry[0]))
    best = None
    # A large object mask can include both wall parts. Try each seed so it cannot veto the parts.
    for seed in candidates:
        members = [seed]
        covered = seed[2].copy()
        for other in candidates:
            if other[0] == seed[0]:
                continue
            if any(np.count_nonzero(m[2] & other[2]) >
                   min(m[1], other[1]) * SETTINGS['maximum_overlap_with_smaller'] for m in members):
                continue
            members.append(other)
            covered |= other[2]
        count = int(covered.sum())
        if len(members) < 2 or count < area * SETTINGS['minimum_explained']:
            continue
        score = (count, -len(members), sum(records[m[0]]['quality'] for m in members))
        if best is None or score > best[0]:
            best = score, [m[0] for m in members]
    return best[1] if best else []


def children(image, part, members, masks, records, *, minimum_seed_area=0):
    """Grow seeds only inside the exact original mask on its native working lattice."""
    local, left, top = _local_mask(np.asarray(part['support']['runs']))
    size = tuple(part['support']['size'])
    h, w = masks.shape[1:]
    x = ((np.arange(local.shape[1]) + left + .5) * w / size[0]).astype(int).clip(0, w - 1)
    y = ((np.arange(local.shape[0]) + top + .5) * h / size[1]).astype(int).clip(0, h - 1)
    labels = np.zeros(local.shape, np.int32)
    # Set high quality last so it owns the small allowed seed overlaps.
    members = sorted(members, key=lambda i: (records[i]['quality'], i))
    for number, i in enumerate(members, 1):
        labels[local & masks[i][np.ix_(y, x)]] = number
    if minimum_seed_area:
        # Tiny slivers can appear where a coarse proposal touches a fine parent rim.
        # Relabel them from the nearest reachable substantial piece inside this parent.
        for number in range(1,len(members)+1):
            for runs in _runs(labels==number):
                if sum(b-a for _,a,b in runs)<minimum_seed_area:
                    for row,a,b in runs:
                        labels[row,a:b]=0
    while True:
        border = np.pad(labels, 1)
        neighbor = np.maximum.reduce([border[:-2, 1:-1], border[2:, 1:-1],
                                      border[1:-1, :-2], border[1:-1, 2:]])
        add = local & (labels == 0) & (neighbor > 0)
        if not add.any():
            break
        labels[add] = neighbor[add]
    gray = np.asarray(image.convert('L').resize(size, Image.Resampling.BOX)
                      .filter(ImageFilter.GaussianBlur(.9)), np.float32)[..., None]
    scale = np.asarray(image.size) / size
    output = []
    for i in np.unique(labels[local]):
        support = local & (labels == i)
        # Preserve disconnected portions and their holes. A component with no reachable seed
        # remains an unresolved wall portion, with no MobileSAM identity.
        for component in _runs(support):
            runs = [[gy + top, gx0 + left, gx1 + left] for gy, gx0, gx1 in component]
            child = dict(id=f"{part['id']}-mobile-{len(output)}", parent_id=part['id'],
                         normal_camera=part['normal_camera'], geometry_status='mask only',
                         relative_inverse_depth=part.get('relative_inverse_depth'),
                         mobile_sam_proposal=members[i-1] if i else None,
                         support=dict(size=list(size), runs=runs),
                         support_area=sum(b-a for _, a, b in runs))
            chart = _patch(runs, gray, scale, size, np.asarray(part['normal_camera']))
            if chart:
                child.update(quad=canonical(chart['quad']), geometry_status='mask with sampling chart')
            output.append(child)
    if sum(o['support_area'] for o in output) != int(local.sum()):
        raise ValueError('Possible loss of parent pixels')
    return output


def refine(image, regions, masks, records, selected):
    result = []
    events = []
    for p in regions:
        if abs(p['normal_camera'][1]) > SETTINGS['maximum_vertical_normal_y']:
            result.append(p)
            continue
        parent = footprint(p, masks.shape[1:], image.size, True)
        parts = choose_parts(parent, masks, records, selected)
        if not parts:
            result.append(p)
            continue
        split = children(image, p, parts, masks, records)
        result.extend(split)
        events.append(dict(parent=p['id'], mobile_masks=parts, children=[c['id'] for c in split]))
    return result, events


@image_only()
def run(probe, out, split, junction_stack=None):
    probe_manifest = read(probe / 'manifest.json')
    baseline_manifest = read(WALLS / 'manifest.json')
    cohort, _, fixtures = groups()
    if (digest(WALLS/'manifest.json') != probe_manifest['baseline_manifest']
            or fixtures != probe_manifest['fixtures'] or digest(ARCHIVE) != probe_manifest['reference_sha256']):
        raise ValueError('Frozen suite or baseline changed')
    cases = {f'{c.floor}/{c.id}': c for group in cohort.values() for c in group}
    dest = out / split
    dest.mkdir(parents=True, exist_ok=False)
    source_root = junction_stack or WALLS
    source_manifest = read(source_root / 'manifest.json')
    if digest(source_root/'cases.json') != source_manifest['artifacts']['cases.json']:
        raise ValueError('Parent predictions manifest does not match')
    if junction_stack and (source_manifest['fixture'] != fixtures or source_manifest['reference_sha256'] != digest(ARCHIVE)):
        raise ValueError('Mismatched stacked prediction fixture')
    method = 'junction-then-underside' if junction_stack else METHOD
    parent_rows = read(source_root / 'cases.json')
    if len(cases) != len(parent_rows) or set(cases) != {r['key'] for r in parent_rows}:
        raise ValueError('Mismatched frame IDs')
    rows = []
    for r in parent_rows:
        if (r['cohort'] == 'development') != (split == 'development'):
            continue
        key = r['key']
        capture = cases[key]
        sha = r['sha256']
        if digest(capture.path('image')) != sha:
            raise ValueError(f'Image changed: {key}')
        paths = [probe/'proposals'/f'{sha}{ext}' for ext in ('.npz', '.json.gz')]
        for p in paths:
            if digest(p) != probe_manifest['artifacts'][p.relative_to(probe).as_posix()]:
                raise ValueError(f'Proposal cache changed: {p}')
        with np.load(paths[0], allow_pickle=False) as archive:
            masks = np.unpackbits(archive['packed'], count=int(archive['shape'][2]), axis=2).astype(bool)
        with gzip.open(paths[1], 'rt', encoding='utf8') as f:
            meta = json.load(f)
        source = source_root / r['predictions']
        if digest(source) != source_manifest['artifacts'][r['predictions']]:
            raise ValueError(f'Baseline prediction changed: {key}')
        with gzip.open(source, 'rt', encoding='utf8') as f:
            baseline = json.load(f)[method]
        with Image.open(capture.path('image')) as f:
            image = f.convert('RGB')
        start = perf_counter()
        prediction, events = refine(image, baseline, masks, meta['candidates'], meta['selected'])
        elapsed = perf_counter()-start
        output = dest / 'predictions' / f'{key}.json.gz'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(gzip.compress(json.dumps(prediction, separators=(',', ':')).encode(), mtime=0))
        # Saved reference data first enters algorithm evaluation here.
        _, ref = recorded(capture)
        scores, matches = measure(prediction, ref, use_mask=True)
        if junction_stack:
            old, pairs = measure(baseline, ref, use_mask=True)
            before = {pid for _, pid in pairs}
            if before != set(r['methods'][method]['found']):
                raise ValueError(f'Parent assignments changed: {key}')
        else:
            old = r['scores'][METHOD]
            before = set(r['scopes'][METHOD]['matched'])
        after = {pid for _, pid in matches}
        for invariant in ('opening_filled', 'gaps_half_filled', 'unclaimed_pixels', 'background_claimed'):
            if scores[invariant] != old[invariant]:
                raise ValueError(f'Exterior changed: {key}, {invariant}')
        rows.append(dict(key=key, sha256=sha, layout=r['layout'], cohort=r['cohort'],
                         baseline=old, split=scores, gained=sorted(after-before), lost=sorted(before-after),
                         assignments={prediction[i]['id']: pid for i, pid in matches},
                         events=events, processing_s=elapsed, predictions=output.relative_to(out).as_posix(),
                         prediction_sha256=digest(output)))
    save(dest/'cases.json', rows)
    scores = {k:aggregate([r[k] for r in rows]) for k in ('baseline', 'split')}
    scores.update(frames=len(rows), gained=sum(len(r['gained']) for r in rows),
                  lost=sum(len(r['lost']) for r in rows), walls_split=sum(len(r['events']) for r in rows),
                  image_only_cached_processing_s=sum(r['processing_s'] for r in rows))
    save(dest/'scores.json', scores)
    save(dest/'manifest.json', dict(method=f'{method} + MobileSAM wall parts',
         run_id=out.name, split=split, settings=SETTINGS, fixtures=fixtures, baseline_manifest=digest(source_root/'manifest.json'),
         probe_manifest=digest(probe/'manifest.json'), code_revision=subprocess.check_output(
             ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(), runner_sha256=digest(Path(__file__)),
         artifacts={n:digest(dest/n) for n in ('cases.json', 'scores.json')}))
    print(split, scores['baseline']['matched'], scores['split']['matched'], 'gained', scores['gained'],
          'lost', scores['lost'], 'walls split', scores['walls_split'], 'guesses',
          scores['baseline']['proposed'], scores['split']['proposed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'regression'], required=True)
    parser.add_argument('--junction-stack', type=Path, help='Use saved junction-then-underside masks as the parent walls')
    args = parser.parse_args()
    run(args.probe, args.out, args.split, args.junction_stack)
