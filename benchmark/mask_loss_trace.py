"""Localize information loss in the published B predictor using saved geometry for scoring."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.camera_labels import rasterize
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.depth_probe_score import occlusion_pairs
from benchmark.normal_fields import cached_fields
from benchmark.score_saved_planes import ROUND
from benchmark.surface_metrics import DOWNSAMPLE, IOU_THRESHOLD, _one_to_one, observed_support
from benchmark.surface_reference import ARCHIVE, recorded, review_case
from engine import normal_regions as b
from engine.camera_lines import _runs
from engine.geometry import bilinear
from engine.surface_model import canonical
from runtime.mode import image_only


METHOD = 'normal-regions'
ANGLE = 30.  # Same signed angular tolerance as B's grouping step.
STAGES = ('orientation_oracle', 'initial_groups', 'fit_inputs', 'fittable', 'emitted_support', 'quad')


def observe_b(image, normals):
    """Passively record the inputs/outputs of B's original helpers in this single process."""
    stages = defaultdict(list)
    original_runs, original_fit = b._runs, b._patch

    def runs(mask):
        pieces = original_runs(mask)
        # Only the top-level call sees global components. Subdivision calls use local coordinates.
        if sys._getframe(1).f_code is b.detect.__code__:
            stages['initial_groups'].extend(pieces)
            stages['analysis_size'] = list(mask.shape[::-1])
        return pieces

    def fit(rows, *args):
        stages['fit_inputs'].append(rows)
        proposal = original_fit(rows, *args)
        stages['fit_outputs'].append(proposal)
        if proposal is not None:
            stages['fittable'].append(rows)
        return proposal

    with ExitStack() as hooks:
        hooks.enter_context(patch.object(b, '_runs', runs))
        hooks.enter_context(patch.object(b, '_patch', fit))
        proposals = b.detect(image, normals)
    return stages, proposals


def overlap_table(masks, ids, eligible):
    """One record per truth surface, with both its best candidate and a one-to-one assignment."""
    sizes = np.bincount(ids.ravel())
    rows = {pid:dict(iou=0., candidate=None, recall=0., precision=0.) for pid in eligible}
    pairs = []
    for i, mask in enumerate(masks):
        hits = np.bincount(ids[mask], minlength=len(sizes))
        area = int(hits.sum())
        for pid in np.flatnonzero(hits):
            if pid not in rows:
                continue
            intersection = int(hits[pid])
            iou = intersection / (area + int(sizes[pid]) - intersection)
            if iou > rows[pid]['iou']:
                rows[pid].update(iou=iou, candidate=i, recall=intersection/int(sizes[pid]),
                                 precision=intersection/area)
            if iou >= IOU_THRESHOLD:
                pairs.append((iou, i, int(pid)))
    assigned = {pid:i for i,pid in _one_to_one(pairs)}
    for pid, row in rows.items():
        row['assigned'] = assigned.get(pid)
        for k in ('iou', 'recall', 'precision'):
            row[k] = round(row[k], 5)
    return rows


def raster_quad(quad, shape):
    homogeneous = np.column_stack((np.asarray(quad)/DOWNSAMPLE, np.ones(4)))
    hit = rasterize(homogeneous, shape[1], shape[0])
    mask = np.zeros(shape, bool)
    if hit:
        mask[hit[0]] = hit[1]
    return mask


def mask_iou(mask, target):
    return round(float(np.count_nonzero(mask & target) / max(1, np.count_nonzero(mask | target))), 5)


def orientation_oracle(reference, normals, eligible):
    # Deliberately uses true direction ONLY for a diagnostic. No true silhouette or spatial crop.
    ids = reference['id_map']
    y,x = np.indices(ids.shape, dtype=float)
    xy = (np.stack((x,y),axis=-1)+.5)*DOWNSAMPLE
    xy *= np.array(normals.shape[1::-1]) / reference['image_size']
    v = bilinear(normals, xy.reshape(-1,2)).reshape(*ids.shape,3)
    v = v / np.maximum(np.linalg.norm(v,axis=2,keepdims=True),1e-9)
    by_direction = defaultdict(set)
    for pid in eligible:
        by_direction[tuple(reference['patches'][pid]['normal_camera'])].add(pid)
    tables, fractions = {}, {}
    for normal, targets in by_direction.items():
        consistent = v @ np.array(normal) >= np.cos(np.deg2rad(ANGLE))
        pieces = _runs(consistent)
        def masks():
            for rows in pieces:
                mask = np.zeros(ids.shape,bool)
                for yy, start, end in rows:
                    mask[yy,start:end] = True
                yield mask
        tables.update(overlap_table(masks(), ids, targets))
        fractions.update({pid:round(float(consistent[ids==pid].mean()),5) for pid in targets})
    return tables, fractions


def loss_stage(stages, normal_fraction):
    """Use the furthest observed evidence for attribution. Threshold recoveries can be non-monotonic."""
    for stage, label in (
        ('quad','outline_matched'), ('emitted_support','distorted_by_quad'),
        ('fittable','filtered_after_fit'), ('fit_inputs','rejected_by_fit'),
        ('initial_groups','group_prefilter'), ('orientation_oracle','lost_in_grouping')):
        if stages[stage]['assigned'] is not None:
            return label
    return 'orientation_present_identity_ambiguous' if normal_fraction >= .5 else 'weak_raw_orientation'


def parities(proposals, planes):
    if len(proposals) != len(planes):
        raise ValueError('B emitted a different number of rectangles')
    for current, saved in zip(proposals, planes):
        keys = ('id','source','support','confidence','support_area')
        if canonical(current['quad']) != saved['quad'] or any(current[k] != saved[k] for k in keys):
            raise ValueError(f'B prediction differs from the published result for {saved["id"]}')


def trace(row, cache):
    fields = cached_fields(row['sha256'], cache)
    if fields is None:
        raise ValueError(f'Cached numerical model fields unavailable or incompatible for {row["key"]}')
    case = review_case(ROUND, row)
    source = case.path('image')
    if digest(source) != row['sha256']:
        raise ValueError(f'Screenshot mismatch on {row["key"]}')
    with Image.open(source) as image:
        intermediate, proposals = observe_b(image.convert('RGB'), fields['normal'])
    with gzip.open(ROUND/row['prediction'], 'rt', encoding='utf8') as stream:
        planes = json.load(stream)[METHOD]['model']['planes']
    parities(proposals, planes)
    _, reference = recorded(case)
    ids = reference['id_map']
    eligible = {k for k,v in reference['patches'].items() if v['eligible']}
    size = reference['image_size']
    tables = {}
    tables['orientation_oracle'], fractions = orientation_oracle(reference, fields['normal'], eligible)
    for stage in ('initial_groups', 'fit_inputs', 'fittable'):
        masks = (observed_support(dict(size=intermediate['analysis_size'], runs=runs), ids.shape, size)
                 for runs in intermediate[stage])
        tables[stage] = overlap_table(masks, ids, eligible)
    tables['emitted_support'] = overlap_table(
        (observed_support(p['support'], ids.shape, size) for p in planes), ids, eligible)
    tables['quad'] = overlap_table((raster_quad(p['quad'], ids.shape) for p in planes), ids, eligible)
    original = row['methods'][METHOD]
    qmatches = original['outlines']['matches']
    downstream = {}
    for m, pitch, lift in zip(qmatches, original['pitch'], original['lift'], strict=True):
        pid = m['patch_id']
        if tables['quad'][pid]['assigned'] != m['proposal_index'] or lift['patch_id'] != pid:
            raise ValueError('Mismatch against original B one-to-one scoring')
        downstream[pid] = dict(counts=pitch['counts'], expected_counts=pitch['expected_counts'],
                               spans_within_half=pitch['half_block_both'], units=lift['units'],
                               depth_error_blocks=lift.get('depth_error_blocks'),
                               grid_registered=bool(pitch['half_block_both'] and (pitch['grid_f1'] or 0.)>=.5))
    if sum(v['assigned'] is not None for v in tables['quad'].values()) != len(qmatches):
        raise ValueError('Unexpected extra matches')
    pairs = occlusion_pairs(reference,fields['depth'])['pairs']
    raw_depth = []
    for pair in pairs:
        a,bid = pair['near_patch'], pair['far_patch']
        parallel = reference['patches'][a]['axis'] == reference['patches'][bid]['axis']
        raw_depth.append(dict(**pair,parallel=parallel))
    surfaces = []
    emitted_indices = {id(proposal):i for i,proposal in enumerate(proposals)}
    for pid in sorted(eligible):
        stages = {s:t[pid] for s,t in tables.items()}
        best_input = stages['fit_inputs']['candidate']
        fit = intermediate['fit_outputs'][best_input] if best_input is not None else None
        support_index = stages['emitted_support']['assigned']
        if support_index is None:
            support_index = stages['emitted_support']['candidate']
        source_to_quad = None if support_index is None else mask_iou(
            raster_quad(planes[support_index]['quad'], ids.shape), ids==pid)
        y,x = np.where(ids==pid)
        photo_box = [int(x.min()*DOWNSAMPLE), int(y.min()*DOWNSAMPLE),
                     int((x.max()+1)*DOWNSAMPLE), int((y.max()+1)*DOWNSAMPLE)]
        surfaces.append(dict(patch=pid, bbox_xyxy=photo_box, reference_pixels=len(y),
                             orientation_fraction=fractions[pid], stages=stages,
                             best_fit_input_returned_quad=fit is not None,
                             best_fit_input_emitted_index=emitted_indices.get(id(fit)),
                             emitted_support_index_for_same_quad=support_index, same_quad_iou=source_to_quad,
                             loss=loss_stage(stages,fractions[pid]), downstream=downstream.get(pid)))
    return dict(frame=row['key'], sha256=row['sha256'], layout=row['layout'],
                split='development' if row['cohort']=='development' else 'reviewed_regression',
                original_cohort=row['cohort'], prediction_sha256=digest(ROUND/row['prediction']),
                proposals=len(planes), surfaces=surfaces, raw_depth_pairs=raw_depth,
                candidate_counts={s:len(intermediate[s]) for s in ('initial_groups','fit_inputs','fittable')})


def summarize(rows):
    surfaces = [s for row in rows for s in row['surfaces']]
    downstream = [s['downstream'] for s in surfaces if s['downstream'] is not None]
    pairs = [pair for row in rows for pair in row['raw_depth_pairs']]
    return dict(frames=len(rows), surfaces=len(surfaces), estimated_patches=sum(r['proposals'] for r in rows),
                mostly_correct_orientation=sum(s['orientation_fraction']>=.5 for s in surfaces),
                matched={stage:sum(s['stages'][stage]['assigned'] is not None for s in surfaces) for stage in STAGES},
                localization=dict(sorted(Counter(s['loss'] for s in surfaces).items())),
                both_support_and_quad=sum(s['stages']['emitted_support']['assigned'] is not None and
                                          s['stages']['quad']['assigned'] is not None for s in surfaces),
                count_returned=sum(s['counts'] is not None for s in downstream),
                spans_within_half_block=sum(s['spans_within_half'] for s in downstream),
                grid_registered=sum(s['grid_registered'] for s in downstream),
                metric_depth_returned=sum(s['depth_error_blocks'] is not None for s in downstream),
                depth_within_one_block=sum(s['depth_error_blocks'] is not None and s['depth_error_blocks']<=1
                                           for s in downstream),
                known_boundary_depth=dict(pairs=len(pairs), correct=sum(p['correct'] for p in pairs),
                    parallel_pairs=sum(p['parallel'] for p in pairs),
                    parallel_correct=sum(p['parallel'] and p['correct'] for p in pairs)))


@image_only()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, required=True, help='Read-only numerical normal/depth cache')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--development-only',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    baseline=read(ROUND/'cases.json')
    rows=[]
    started=perf_counter()
    for split in ('development','regression','blind'):
        if args.development_only and split!='development':
            continue
        for row in baseline:
            if row['cohort']==split:
                rows.append(trace(row,args.fields))
        print(f'{split}: {len(rows)} frames checked',flush=True)
    summary=dict(all=summarize(rows))
    for key in ('split','layout'):
        summary[key]={k:summarize([r for r in rows if r[key]==k]) for k in sorted({r[key] for r in rows})}
    per_picture={r['frame']:summarize([r]) for r in rows}
    payload=json.dumps(rows,separators=(',',':'),allow_nan=False).encode()
    (args.out/'surfaces.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    save(args.out/'summary.json', summary)
    save(args.out/'pictures.json', per_picture)
    baseline_manifest=read(ROUND/'manifest.json')
    save(args.out/'manifest.json',dict(run_id=args.out.name, diagnostic=True, method=METHOD, version='26.3',
         baseline_run='normals-01', code=revision(), baseline_source=baseline_manifest['code'],
         model_sha256=baseline_manifest['model_sha256'], numerical_inputs='verified cached normal and relative-depth tensors',
         raw_orientation_degrees=ANGLE, matching_iou=IOU_THRESHOLD, native_pixel_stride=DOWNSAMPLE,
         reference_sha256=digest(ARCHIVE), fixture_sha256=baseline_manifest['fixtures'],
         elapsed_cached_seconds=round(perf_counter()-started,3),
         baseline_prediction_parity='IDs, RLE support, quads, confidence, source and support area replayed exactly',
         downstream='Published B stops at surfaces.json. Ordinary assembly consumes separate facade geometry.',
         artifacts={p.name:digest(p) for p in args.out.iterdir() if p.is_file()}))
    print(summary['all'])


if __name__=='__main__':
    main()
