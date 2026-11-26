"""Test frozen RGB and ECT edge gates on intact walls in the finished Surface Finder."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.afterpass_edge_diagnostic import sample_scores
from benchmark.afterpass_headroom import contrast_at_boundary, load_table
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.single_wall_sheet import load_gzip
from benchmark.split_count_check import count_frame
from benchmark.split_round import pooled, score_regions
from benchmark.surface_reference import ARCHIVE
from benchmark.wall_join_test import capture_index, checked
from engine.single_wall_gate import partition
from runtime.mode import image_only, mark_image_only


METHODS = ('current', 'rgb', 'ect_geometry')
SOURCE_METHOD = 'single-v2'


def mask_on_native_grid(part):
    width, height = part['support']['size']
    mask = np.zeros((height, width), bool)
    for y, left, right in part['support']['runs']:
        mask[y, left:right] = True
    return mask


def split(image, parent, index, mask):
    pieces = partition(image, parent, index, mask)
    if not {'inside', 'remainder'} <= {p['single_wall_side'] for p in pieces}:
        return [parent]
    counts = np.sum([mask_on_native_grid(p) for p in pieces], axis=0)
    if not np.array_equal(counts, mask_on_native_grid(parent)):
        raise ValueError('A cut altered the native parent footprint or overlapped children')
    return pieces


def proposals(image, baseline, inventory, masks, geometry, thresholds):
    parents = {p['id']: p for p in baseline}
    rgb = np.asarray(image.resize(geometry.shape[::-1], Image.Resampling.BOX), float)
    predictions = {'current':baseline}
    evidence = []
    for candidate in inventory:
        parent, index = candidate['parent'], candidate['proposal']
        if parent not in parents:
            continue
        mask = masks[index]
        pieces = split(image, parents[parent], index, mask)
        if len(pieces) == 1:
            continue
        event = dict(parent=parent, proposal=index, rgb=contrast_at_boundary(rgb, candidate),
                     ect_geometry=sample_scores(candidate, geometry)['ect_geometry'], applied={})
        evidence.append((event, mask, pieces))
    for method in METHODS[1:]:
        parts = {key:[part] for key, part in parents.items()}
        for event, mask, pieces in evidence:
            parent, index = event['parent'], event['proposal']
            event['applied'][method] = False
            if event[method] < thresholds[method]:
                continue
            previous = parts[parent]
            # Multiple accepted outlines may cross one wall. Keep their fixed input order and
            # apply each to every surviving child it still splits. Never cut outside the parent.
            updated = (pieces if previous[0]['id'] == parent else
                       [p for child in previous for p in split(image, child, index, mask)])
            event['applied'][method] = len(updated) > len(previous)
            parts[parent] = updated
        predictions[method] = [p for children in parts.values() for p in children]
    return predictions, [event for event, _, _ in evidence]


def evaluate(job):
    source, sam, fields, row, capture, hashes, thresholds, out = job
    key, sha = row['key'], row['sha256']
    image_path = capture.path('image')
    checked(image_path, sha)
    with Image.open(image_path) as opened:
        image = opened.convert('RGB')
    baseline = load_gzip(source, row['predictions'], hashes)[SOURCE_METHOD]
    saved = load_gzip(source, row['boundary_evidence'], hashes)
    start = perf_counter()
    with np.load(checked(fields/'fields'/(sha+'.npz'), hashes['ect']), allow_pickle=False) as field:
        geometry = np.maximum(field['depth'], field['normal']).astype(float)
    if geometry.shape != (image.height//4, image.width//4):
        raise ValueError(f'Misaligned ECT map on {key}')
    parents = {p['id'] for p in baseline}
    inventory = [c for c in saved['inventory'] if c['parent'] in parents]
    with np.load(checked(sam/'proposals'/(sha+'.npz'), hashes['sam']), allow_pickle=False) as arrays:
        packed, width = arrays['packed'], int(arrays['shape'][2])
        masks = {i:np.unpackbits(packed[i], axis=1, count=width).astype(bool)
                 for i in {c['proposal'] for c in inventory}}
    predictions, events = proposals(image, baseline, inventory, masks, geometry, thresholds)
    elapsed = perf_counter()-start
    for name, parts in predictions.items():
        if len({p['id'] for p in parts}) != len(parts):
            raise ValueError(f'Duplicate ID for {key}/{name}')
    rel = 'predictions/'+key+'.json.gz'
    path = out/rel
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(gzip.compress(json.dumps(predictions, separators=(',',':'), allow_nan=False).encode(), mtime=0))
    # Predictions are complete. Recorded labels enter scoring here.
    refdiag = {(r['parent'],r['proposal']):r for r in saved['reference_diagnostics']}
    for event in events:
        diag = refdiag[event['parent'],event['proposal']]
        event.update(truth=diag['kind'], material_seam=diag['material_seam'])
    start = perf_counter()
    # Grade every frame. Identical full predictions only need one mask and texture count replay.
    distinct = {m:parts for m,parts in predictions.items() if m == 'current' or parts != baseline}
    scores, scopes = score_regions(capture, distinct, key)
    counts, _ = count_frame(capture, image, dict(scopes=scopes,key=key), distinct, tuple(distinct))
    sizes, size_changes = {}, {}
    for method in METHODS:
        ref = method if method in distinct else 'current'
        scores[method], scopes[method] = scores[ref], scopes[ref]
        sizes[method] = {k:counts[ref]['totals'][k] for k in row['sizes'][SOURCE_METHOD]}
        size_changes[method] = counts[ref]['comparisons']
    if scores['current'] != row['scores'][SOURCE_METHOD] or sizes['current'] != row['sizes'][SOURCE_METHOD]:
        raise ValueError(f'Baseline count or score changed in {key}')
    return dict(key=key, sha256=sha, layout=row['layout'], cohort=row['cohort'], predictions=rel, events=events,
                cached_processing_s=elapsed, cached_scoring_s=perf_counter()-start, scores=scores, scopes=scopes,
                sizes=sizes, size_changes=size_changes)


def summarize(rows, method):
    result = pooled(rows, method)
    sizes, cuts = Counter(), Counter()
    for row in rows:
        sizes.update(row['sizes'][method])
        for event in row['events']:
            if event['applied'].get(method):
                cuts[event['truth']] += 1
                if event['truth'] == 'flush wall' and event['material_seam']:
                    cuts['material seam'] += 1
    return dict(result, sizes=dict(sizes), applied_cuts=dict(cuts),
                cuttable=sum(len(r['events']) for r in rows), views=len(rows))


@image_only()
def run(source_root, fields, diagnostic, out, workers=3):
    start = perf_counter()
    reviews = source_root/'benchmark/reviews'
    source, sam = reviews/'single-wall-depth-01', reviews/'mobile-regions-01'
    base, mobile, ect, diag = [read(p/'manifest.json') for p in (source,sam,fields,diagnostic)]
    checked(source/'manifest.json', diag['source_manifests'][source.name])
    checked(sam/'manifest.json', base['sources'][sam.name])
    if digest(fields/'manifest.json') not in diag['source_manifests'].values():
        raise ValueError('ECT outputs differ from the original threshold diagnostic')
    checked(ARCHIVE, base['reference_sha256'])
    rows = load_table(source, 'cases.json', base)
    captures = capture_index(source_root/'benchmark/data')
    roster = {r['key']:r['sha256'] for r in rows}
    if (len(roster) != len(rows) or len(rows) != 56 or captures.keys() != roster.keys() or roster != diag['inputs']):
        raise ValueError('Source images changed')
    freeze = load_table(diagnostic, 'development-freeze.json', diag)
    thresholds = {m:freeze[m] for m in METHODS[1:]}
    out.mkdir(exist_ok=False, parents=True)
    jobs = []
    for row in rows:
        sha = row['sha256']
        hashes = {k:base['artifacts'][k] for k in (row['predictions'], row['boundary_evidence'])}
        hashes.update(ect=ect['artifacts']['fields/'+sha+'.npz'], sam=mobile['artifacts']['proposals/'+sha+'.npz'])
        jobs.append((source,sam,fields,row,captures[row['key']],hashes,thresholds,out))
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        results = list(pool.map(evaluate, jobs))
    headroom = reviews/'afterpass-headroom-01'
    checked(headroom/'manifest.json', diag['source_manifests'][headroom.name])
    prior = load_table(headroom, 'individual-cuts.json', read(headroom/'manifest.json'))
    eligible = {(r['frame'],r['parent'],r['proposal']) for r in prior}
    measured = {(r['key'],e['parent'],e['proposal']) for r in results for e in r['events']}
    if measured != eligible:
        raise ValueError('Cut opportunities changed from the earlier independent inventory')
    boundaries = load_table(diagnostic, 'boundaries.json', diag)
    signals = {(r['frame'],r['parent'],r['proposal']):r for r in boundaries}
    for row in results:
        for event in row['events']:
            original = signals[row['key'],event['parent'],event['proposal']]
            if any(event[k] != original[k] for k in ('rgb','ect_geometry','truth','material_seam')):
                raise ValueError('Gate responses or reference checks changed from the frozen diagnostic')
    scopes = ('all','development','previously_reviewed_regression',*sorted({r['layout'] for r in rows}))
    table = {scope:{m:summarize([r for r in results if scope in ('all',r['layout'],r['cohort'])],m) for m in METHODS}
             for scope in scopes}
    save(out/'cases.json',results)
    save(out/'scores.json',table)
    save(out/'manifest.json',dict(run_id=out.name,code=revision(),version=base['version'],fixtures=base['fixtures'],
         reference_sha256=base['reference_sha256'], source_sha256=roster, thresholds=thresholds,
         checkpoint_sha256=ect['checkpoint_sha256'], upstream_revision=ect['code_revision'],
         inputs={p.name:digest(p/'manifest.json') for p in (source,sam,fields,diagnostic,reviews/'afterpass-headroom-01')},
         design='Supplemental follow-up after inspecting the complete boundary diagnostic. Thresholds remain frozen '
                'on 20 development views. All six other buildings are previously reviewed regression data.',
         method='Keep the finished default. On its intact parent masks, take every pre-existing usable MobileSAM '
                'interface that passes the method threshold. Partition recursively in the archived proposal order. '
                'Verify each cut keeps all native parent pixels, all holes, and has disjoint child masks.',
         timings=dict(wall_s=perf_counter()-start,cached_processing_s=sum(r['cached_processing_s'] for r in results),
                      scoring_s=sum(r['cached_scoring_s'] for r in results),num_workers=workers,new_inference_s=0),
         artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()},
         reproduce=f'python -m benchmark.afterpass_wall_cuts --source-root "{source_root.resolve()}" '
                   f'--fields "{fields.resolve()}" --diagnostic "{diagnostic.resolve()}" --out runs/edge-cuts-replay'))
    for m,r in table['all'].items():
        print(m, r['matched'],r['unpaired'],r['sizes'],r['applied_cuts'],flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--fields',type=Path,required=True)
    parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    run(args.source_root,args.fields,args.diagnostic,args.out,args.workers)
