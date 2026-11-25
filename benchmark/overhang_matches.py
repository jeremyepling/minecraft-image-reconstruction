"""Audit complementarity of six archived surface methods against recorded plane identities."""
import argparse
import gzip
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from benchmark.camera_labels import rasterize
from benchmark.camera_round import ROOT, digest, read, save
from benchmark.mask_metrics import PROTOCOL, footprint
from benchmark.normal_round import groups
from benchmark.surface_detection_maps import _native_reference, _ratios
from benchmark.surface_metrics import _one_to_one
from benchmark.surface_reference import ARCHIVE, recorded
from engine.surface_perimeter import _native_alpha
from runtime.mode import image_only


METHODS = ('current-b', 'mask-first', 'color-edges', 'depth-split', 'scalelsd', 'deeplsd')
DEFAULT = 'color-edges'
PERIMETERS = ROOT / 'benchmark/reviews/perimeters-01'
ORIGINAL = ROOT / 'benchmark/reviews/mask-four-01'


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def native_hits(part, name, native, bins):
    """Use the original B quadrilateral or the numerical region's actual footprint."""
    if name == 'current-b':
        quad = np.column_stack((part['quad'], np.ones(4)))
        data = rasterize(quad, native.shape[1], native.shape[0])
        if data is None:
            return np.zeros(bins, dtype=int)
        box, mask = data[:2]
        pixels = native[box][mask]
    else:
        data = _native_alpha(part['support'], native.shape[::-1])
        if data is None:
            return np.zeros(bins, dtype=int)
        alpha, (x, y, xx, yy) = data
        pixels = native[y:yy, x:xx][np.asarray(alpha) > 0]
    return np.bincount(pixels, minlength=bins)


def pair(ious, targets, available):
    choices = [(float(ious[i, target]), i, target) for i in available for target in targets
               if ious[i, target] >= PROTOCOL['iou']]
    return dict(_one_to_one(choices))


def score_parts(parts, name, ref, native, rows):
    ids = ref['id_map']
    bins = len(rows) + 1
    sampled = np.zeros((len(parts), bins), dtype=int)
    full = np.zeros_like(sampled)
    for i, part in enumerate(parts):
        mask = footprint(part, ids.shape, ref['image_size'], use_mask=name != 'current-b')
        sampled[i] = np.bincount(ids[mask], minlength=bins)
        full[i] = native_hits(part, name, native, bins)
    stride_iou = _ratios(sampled, np.bincount(ids.ravel(), minlength=bins))
    native_iou = _ratios(full, np.bincount(native.ravel(), minlength=bins))
    eligible = [row['id'] for row in rows if row['eligible']]
    small = [row['id'] for row in rows if not row['eligible']]
    primary = pair(stride_iou, eligible, range(len(parts)))
    secondary = pair(native_iou, small, [i for i in range(len(parts)) if i not in primary])
    matched = {pid: i for i, pid in {**primary, **secondary}.items()}
    for row in rows:
        pid = row['id']
        ious, hits, size = ((stride_iou, sampled, row['samples']) if row['eligible']
                            else (native_iou, full, row['pixels']))
        i = matched.get(pid)
        best = cover = None
        if np.any(hits[:, pid]):
            best = int(ious[:, pid].argmax())
            cover = int(hits[:, pid].argmax())
        row['methods'][name] = dict(
            proposal=parts[i]['id'] if i is not None else None,
            iou=float(ious[i, pid]) if i is not None else None,
            best_iou=float(ious[best, pid]) if best is not None else 0,
            best_proposal=parts[best]['id'] if best is not None else None,
            coverage=float(hits[cover, pid] / size) if cover is not None else 0,
            covering_proposal=parts[cover]['id'] if cover is not None else None,
            covering_matched=primary.get(cover))
    return {pid: parts[i]['id'] for i, pid in primary.items()}


def report(rows):
    matched = {m: {i for i, row in enumerate(rows) if row['methods'][m]['proposal'] is not None}
               for m in METHODS}
    baseline = matched[DEFAULT]
    stats = {m: dict(found=len(ids), extra=len(ids - baseline), loss=len(baseline - ids))
             for m, ids in matched.items()}
    return dict(total=len(rows), methods=stats, any=len(set.union(*matched.values())),
                any_without_b=len(set.union(*(matched[m] for m in METHODS if m != 'current-b'))))


def summarize(views):
    summary = {}
    scopes = {
        'all': list(views.values()), 'sheet': [v for v in views.values() if v['sheet']],
        'development': [v for v in views.values() if v['cohort'] == 'development'],
        'regression': [v for v in views.values() if v['cohort'] != 'development']}
    for scope, subset in scopes.items():
        for eligible in (True, False):
            rows = [r for v in subset for r in v['references'] if r['eligible'] == eligible]
            by_orientation = dict(all=rows, down=[r for r in rows if r['normal'][1] == -1],
                                  vertical=[r for r in rows if r['normal'][1] == 0],
                                  top=[r for r in rows if r['normal'][1] == 1])
            for name, oriented in by_orientation.items():
                label = 'eligible' if eligible else 'small'
                summary[f'{scope}-{name}-{label}'] = report(oriented)
    for key, view in views.items():
        if view['sheet']:
            rows = [r for r in view['references'] if r['eligible']]
            summary[f'{key}-all'] = report(rows)
            summary[f'{key}-down'] = report([r for r in rows if r['normal'][1] == -1])
    return summary


def run(out):
    pm, bm = (read(d / 'manifest.json') for d in (PERIMETERS, ORIGINAL))
    sources = [(PERIMETERS, pm['artifacts']), (ORIGINAL, bm['artifact_sha256'])]
    cases = {}
    for directory, hashes in sources:
        verify(digest(directory / 'cases.json') == hashes['cases.json'], f'Cases changed in {directory}')
        cases[directory] = {r['key']: r for r in read(directory / 'cases.json')}
    verify(pm['reference_sha256'] == bm['reference_sha256'] == digest(ARCHIVE), 'Reference changed')
    cohorts, _, fixtures = groups()
    verify(pm['fixtures'] == bm['fixtures'] == fixtures, 'Fixtures changed')
    lookup = {f'{c.floor}/{c.id}': c for cohort in cohorts.values() for c in cohort}
    verify(cases[PERIMETERS].keys() == cases[ORIGINAL].keys() == lookup.keys(), 'Frame IDs changed')
    sheet_path = PERIMETERS / 'surface-detection.json'
    verify(digest(sheet_path) == pm['artifacts'][sheet_path.name], 'Review sample changed')
    sheet = {r['frame']: r for r in read(sheet_path)['views']}
    views = {}
    for key, current in cases[PERIMETERS].items():
        case = lookup[key]
        verify(cases[ORIGINAL][key]['sha256'] == current['sha256'] == case.frame['image_sha256'],
               f'Screenshot hashes differ for {key}')
        pose, ref = recorded(case)
        native = _native_reference(pose, ref, read(case.path('labels'))['faces'])
        sizes = np.bincount(native.ravel(), minlength=max(ref['patches']) + 1)
        rows = [dict(id=i, normal=p['normal'], eligible=p['eligible'], samples=p['support_px'],
                     pixels=int(sizes[i]), faces=p['faces'], methods={}) for i, p in ref['patches'].items()]
        for directory, hashes in sources:
            source = cases[directory][key]
            path = source['predictions']
            verify(digest(directory / path) == hashes[path], f'Predictions changed for {key}')
            with gzip.open(directory / path, 'rt', encoding='utf8') as file:
                predictions = json.load(file)
            for method, parts in predictions.items():
                matches = score_parts(parts, method, ref, native, rows)
                verify(len(matches) == source['scores'][method]['matched'], f'Score differs for {key}/{method}')
                if directory == ORIGINAL:
                    saved = {m['recorded_surface']: m['proposal'] for m in source['matched_regions'][method]}
                    verify(matches == saved, f'Matched identities differ for {key}/{method}')
                if method == DEFAULT and key in sheet:
                    saved = {r['id'] for r in sheet[key]['references'] if r['prediction'] is not None}
                    current_ids = {r['id'] for r in rows if r['methods'][method]['proposal'] is not None}
                    verify(current_ids == saved, f'Sheet matches differ for {key}')
        views[key] = dict(layout=current['layout'], sheet=key in sheet, sha256=current['sha256'],
                          cohort=current['cohort'], references=rows)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'results.json', dict(methods=METHODS, views=views,
         selection='Diagnostic one-to-one matches from saved predictions. Small surfaces are paired separately at native resolution after regular matches.',
         source_manifests={d.name: digest(d / 'manifest.json') for d, _ in sources}, reference_sha256=digest(ARCHIVE)))
    save(out / 'summary.json', summarize(views))
    print(out.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='New directory for per-reference records and totals')
    args = parser.parse_args()
    with image_only():
        run(args.out or ROOT / 'runs' / f'overhang-method-audit-{uuid4().hex[:8]}')
