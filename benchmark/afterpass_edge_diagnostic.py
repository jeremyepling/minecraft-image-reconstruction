"""Test ECT and RGB on the same proposed wall boundaries, independently of surface assembly."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.afterpass_headroom import contrast_at_boundary, contrast_score, load_table
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.single_wall_sheet import checked, load_gzip
from benchmark.wall_join_test import capture_index
from runtime.mode import image_only


METHODS = ('rgb', 'ect_geometry', 'ect_geometry_minus_appearance')
DEV = 'development'
REG = 'previously_reviewed_regression'


def sample_scores(candidate, geometry, margin):
    # Coordinates are centers on the same four-pixel lattice as the two inner sample cells.
    pairs = np.asarray(candidate['samples']).reshape(-1, 4, 2)
    midpoint = pairs[:, 1:3].mean(axis=1)
    # Search a 3x3 neighborhood around the midpoint with bilinear sampling.
    offsets = np.asarray([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)])
    y, x = np.clip(midpoint[:, None, :] + offsets, [0, 0], np.asarray(geometry.shape) - 1).transpose(2, 0, 1)
    y0, x0 = y.astype(int), x.astype(int)
    y1, x1 = np.minimum(y0 + 1, geometry.shape[0]-1), np.minimum(x0 + 1, geometry.shape[1]-1)
    dy, dx = y-y0, x-x0
    features = []
    for field in (geometry, margin):
        nearby = ((1-dy)*((1-dx)*field[y0,x0] + dx*field[y0,x1]) +
                  dy*((1-dx)*field[y1,x0] + dx*field[y1,x1]))
        features.append(float(np.median(nearby.max(axis=1))))
    return dict(zip(METHODS[1:], features))


def features(row, capture, archive, case_hashes, fields, field_hashes, rgb_archive):
    sha = row['sha256']
    checked(capture.path('image'), sha)
    boundary = load_gzip(archive, row['boundary_evidence'], case_hashes)
    rel = 'fields/'+sha+'.npz'
    checked(fields/rel, field_hashes[rel])
    with np.load(fields/rel, allow_pickle=False) as data:
        channels = [data[c] for c in ('depth','normal','reflectance','illumination')]
    with Image.open(capture.path('image')) as photo:
        grid = (photo.height//4, photo.width//4)
        if any(c.shape != grid or c.dtype != np.uint8 for c in channels):
            raise ValueError(f'Misaligned ECT arrays on {row["key"]}')
        rgb = np.asarray(photo.convert('RGB').resize(grid[::-1], Image.Resampling.BOX), float)
    d, n, r, i = channels
    geometry = np.maximum(d, n).astype(float)
    margin = geometry - np.maximum(r, i)
    records = []
    for cut, ref in zip(boundary['inventory'], boundary['reference_diagnostics'], strict=True):
        parent, proposal = cut['parent'], cut['proposal']
        old = rgb_archive[(row['key'], parent, proposal)]
        rgb_score = contrast_at_boundary(rgb, cut)
        if ((parent, proposal) != (ref['parent'], ref['proposal']) or rgb_score != old['contrast']
                or ref['kind'] != old['truth'] or ref['material_seam'] != old['material_seam']
                or row['cohort'] != old['split']):
            raise ValueError('RGB control or scored boundary has changed')
        records.append(dict(frame=row['key'], parent=parent, proposal=proposal, samples=len(cut['samples']),
                            truth=ref['kind'], material_seam=ref['material_seam'], rgb=rgb_score,
                            **sample_scores(cut, geometry, margin)))
    return records


def grade(records, thresholds=None):
    summary = {}
    for method in METHODS:
        values = [dict(contrast=r[method], truth=r['truth']) for r in records]
        if thresholds is None:
            def rank(threshold):
                s = contrast_score(values, threshold)
                return (s['real_passed']/s['real'] + 1 - s['flat_passed']/s['flat'])/2, threshold
            threshold = max({v['contrast'] for v in values} | {256.}, key=rank)
        else:
            threshold = thresholds[method]
        seams = [r for r in records if r['truth'] == 'flush wall' and r['material_seam']]
        summary[method] = dict(contrast_score(values, threshold), threshold=threshold, material=len(seams),
                               material_passed=sum(r[method]>=threshold for r in seams))
    return summary


@image_only()
def run(source, fields, out):
    start = perf_counter()
    if fields.name == 'fields' and not (fields/'manifest.json').is_file():
        fields = fields.parent
    reviews = source/'benchmark/reviews'
    archive = reviews/'single-wall-depth-01'
    prior = reviews/'afterpass-headroom-01'
    manifest, control, field_manifest = [read(p/'manifest.json') for p in (archive, prior, fields)]
    checked(archive/'manifest.json', control['source_manifests'][archive.name])
    checked(source/'benchmark/data/visible-geometry-v1.zip', manifest['reference_sha256'])
    for key in ('reference_sha256','fixtures'):
        if not control[key] == field_manifest[key] == manifest[key]:
            raise ValueError('ECT or RGB used a different reference or fixture')
    rows = load_table(archive, 'cases.json', manifest)
    old_records = load_table(prior, 'contrast.json', control)
    old_scores = load_table(prior, 'summary.json', control)['contrast']
    rgb_archive = {(r['frame'],r['parent'],r['proposal']):r for r in old_records}
    captures = capture_index(source/'benchmark/data')
    roster = {r['key']:r['sha256'] for r in rows}
    if (len(rows) != 56 or roster != control['per_frame_input_sha256'] or captures.keys() != roster.keys()
            or manifest['artifacts']['cases.json'] != field_manifest['roster_sha256']):
        raise ValueError('The scored screenshot roster changed')
    out.mkdir(parents=True, exist_ok=False)
    scores, all_records, thresholds = {}, [], None
    for split in (DEV, REG):
        selected = [row for row in rows if row['cohort'] == split]
        records = []
        for row in selected:
            records.extend(features(row, captures[row['key']], archive, manifest['artifacts'], fields,
                                    field_manifest['artifacts'], rgb_archive))
        scores[split] = grade(records, thresholds)
        if split == DEV:
            thresholds = {m:d['threshold'] for m,d in scores[DEV].items()}
            save(out/'development-freeze.json', thresholds)
        for rec in records:
            rec['split'] = split
            rec['flagged'] = {m:rec[m] >= thresholds[m] for m in METHODS}
        all_records.extend(records)
        previous = old_scores['development' if split==DEV else 'regression']
        if not all(scores[split]['rgb'][k] == value for k, value in previous.items()):
            raise ValueError('Published RGB control did not reproduce')
    scored_ids = {(r['frame'], r['parent'], r['proposal']) for r in all_records}
    if len(all_records) != len(scored_ids) or scored_ids != rgb_archive.keys():
        raise ValueError('Lost or duplicated an archived boundary')
    save(out/'boundaries.json', all_records)
    save(out/'scores.json', scores)
    save(out/'manifest.json', dict(run_id=out.name, code=revision(), method_ids=METHODS, inputs=roster,
         version=manifest['version'], fixtures=manifest['fixtures'], reference_sha256=manifest['reference_sha256'],
         source_manifests={p.name:digest(p/'manifest.json') for p in (archive,prior,fields)},
         checkpoint_sha256=field_manifest['checkpoint_sha256'], upstream_revision=field_manifest['code_revision'],
         source_scoring='Majority of paired samples on parallel planes at different offsets vs on one flat wall. '
                        'Mixed cases are reported individually and excluded from accuracy and AUC.',
         cases={split:{'views':sum(r['cohort'] == split for r in rows),
                       'proposals':sum(r['split'] == split for r in all_records),
                       'mixed':sum(r['split'] == split and r['truth'] == 'mixed' for r in all_records)}
                for split in (DEV, REG)},
         settings='At each sampled interface midpoint, bilinearly sample max(depth, normal) and '
                  'max(depth,normal)-max(reflectance,illumination) on 3x3 locations (one 4px grid cell tolerance per axis). '
                  'Take the largest local response then the median along the proposed cut. Use normalized uint8 units. '
                  'Freeze each threshold on development by maximum balanced accuracy, highest threshold on a tie.',
         scope='Boundary classification diagnostics. No surface partitions, accept/reject changes or block placement.',
         runtime_seconds={'new_inference':0,'cached_diagnostic':perf_counter()-start},
         artifacts={f:digest(out/f) for f in ('scores.json','boundaries.json','development-freeze.json')},
         reproduce='python -m benchmark.afterpass_edge_diagnostic '
                   f'--source-root "{source.resolve()}" --fields "{fields.resolve()}" --out runs/edge-diagnostic-replay'))
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--fields', type=Path, required=True, help='ECT numerical field run with manifest.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.source_root, args.fields, args.out))
