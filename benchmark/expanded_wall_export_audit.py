"""Reopen and verify frozen and experimental chart-to-export receipts."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from json import dumps, loads
from pathlib import Path
import sys
from unittest.mock import patch

CHECKOUT = next(folder for folder in Path(__file__).resolve().parents
                if (folder/'benchmark/expanded_baseline.py').is_file())
sys.path.insert(0, str(CHECKOUT))

import cv2
import numpy as np

from benchmark import expanded_scores, floating_cloud_replay
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import assemble, axis_vector
from benchmark.relative_face_scoring import evaluate_relative_voxels, yaw_from_anchor_normal
from benchmark.sky_crop_guard import pooled
from engine.mask_first import _image_depth
from engine.normal_regions import _field
from engine.surface_fields import MODELS
from engine.surface_perimeter import _native_alpha


DATASETS = {
    'visible-builds-v2': (12, 144, 'cd8bfb2953dd3418d0c3666a76cd12f2e0c2193f31e7545a93c1dca13d01e34f'),
    'visible-builds-v3': (18, 144, '6e7b25fa56aa39d04a34bb3984551cf11441c11acb1a021b045d4f86e824aa28'),
    'decorative-builds-v1': (8, 40, '09d5612ee175718b979c73363eb40fe05959d90f96aab894e82efa9548eaf064'),
}
ATLAS_SHA = '49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950'
BASELINE_CHECKOUT = WORKTREES / 'crop-ceiling-expanded'
BASE = BASELINE_CHECKOUT / 'runs/native-crop-ceiling'
OUT = ROOT / 'runs/wall-grid-audit'
REVISION = 'd4f2e0959b6d7abe84000f05d139a9627b382b5f'
SETTINGS = dict(min_edges=1, edge_pixels=2.5, coplanar_pixels=.5)
SETTINGS_BY_ARM = {
    'grids/signed-two-edges': SETTINGS | {'min_edges': 2},
    'grids/signed-tight-edges': SETTINGS | {'edge_pixels': .0001, 'coplanar_pixels': .0001},
    'grids/source-normal-two-edges': SETTINGS | {'min_edges': 2},
}
CONTROLS = {
    'rulers/spacing': 'rulers/noop',
    'rulers/origin': 'rulers/spacing',
    'rulers/source_resample': 'source',
    'rulers/source_material': 'rulers/source_resample',
    'grids/signed': 'grids/source-normal',
    'grids/signed-two-edges': 'grids/signed',
    'grids/signed-tight-edges': 'grids/signed',
    'grids/source-normal-two-edges': 'grids/source-normal',
}


def cases():
    framing_audit = read(BASELINE_CHECKOUT/'training/framing-audit-v2.json')
    assert framing_audit['min_margin_fraction'] == .02
    assert framing_audit['margin_order'] == ['left', 'top', 'right', 'bottom']
    framing = {row['id']: row for row in framing_audit['frames']}
    for dataset, (n_test, n_all, manifest_sha) in DATASETS.items():
        dataset_folder = SOURCE / dataset
        manifest = dataset_folder / 'manifest.json'
        assert digest(manifest) == manifest_sha
        records = read(manifest)['frames']
        assert len(records) == n_all == len({r['id'] for r in records})
        split_by_layout = {}
        for record in records:
            layout = record.get('layout') or record.get('structure')
            if layout:
                assert split_by_layout.setdefault(layout, record['split']) == record['split']
        test = [r for r in records if r['split'] == 'test']
        assert len(test) == n_test
        existing = {p.stem for p in (BASE/'pairs'/dataset).glob('*.json')}
        assert {r['id'] for r in test} == existing
        for record in test:
            name = record['id']
            frame = dataset_folder/'data/test/frames'/name
            source = frame/'rgb.png'
            proof = framing[name] if dataset == 'visible-builds-v2' else None
            if proof:
                assert proof['accepted'] and proof['rgb_sha256'] == record['rgb_sha256']
                assert proof['split'] == 'test' and proof['reason'] == 'whole_structure'
                assert all(margin >= size*.02 for margin, size in zip(
                    proof['projected_margins_px'], proof['image_size']*2, strict=True))
            variant = dict(image=str(source), source_sha256=record['rgb_sha256'])
            yield dict(record, dataset=dataset, frame=str(frame), variants={'sky': variant}), proof


def with_v2_framing(case, proof):
    frame_json = Path(case['frame'])/'frame.json'

    def adapter(path):
        data = read(path)
        return data | {'framing': proof} if proof and path == frame_json else data

    return patch.object(expanded_scores, 'read', adapter)


def native_masks(case, handoff):
    frame = Path(case['frame'])
    labels = read(frame/'labels.json')
    with np.load(frame/'labels.npz', allow_pickle=False) as arrays:
        owners = arrays['face_id']
    lut = np.zeros(int(owners.max()) + 1, dtype=np.int32)
    for face in labels['faces']:
        lut[face['id']] = face['surface_id']
    surfaces = lut[owners]
    assert np.count_nonzero(surfaces) == np.count_nonzero(owners)
    size = owners.shape[::-1]
    surface_file = read(handoff/'surfaces.json')
    planes = surface_file['planes']
    assert list(size) == surface_file['image_size']
    assert {int(p['id'].removeprefix('plane-')) for p in planes} == set(lut) - {0}
    assert len(planes) == len(set(lut)) - 1
    pixels = np.bincount(surfaces.ravel())
    cache = (BASE/'fields'/case['dataset']/case['id'] if case['dataset'] == 'visible-builds-v2'
             else WORKTREES/'full-expanded-controls/runs/expanded-baseline-01'/case['dataset']/case['id']/'handoffs')
    model_sha = {kind: spec[2] for kind, spec in MODELS.items()}
    assert read(handoff/'build-manifest.json')['model_sha256'] == model_sha
    if case['dataset'] != 'visible-builds-v2':
        receipt = read(cache.parent/'receipt.json')
        assert receipt['source_sha256'] == case['rgb_sha256']
        assert {kind: receipt['model_sha256'][kind] for kind in model_sha} == model_sha
    with np.load(cache/'normal.npz', allow_pickle=False) as array:
        normal = _field(array['value'], size)
    with np.load(cache/'depth.npz', allow_pickle=False) as array:
        depth = _image_depth(array['value'], size)
    for plane in planes:
        surface_id = int(plane['id'].removeprefix('plane-'))
        alpha, (x0, y0, x1, y1) = _native_alpha(plane['support'], size)
        actual = np.asarray(alpha) > 0
        expected = surfaces[y0:y1, x0:x1] == surface_id
        assert int(actual.sum()) == pixels[surface_id] == plane['support_area']
        assert np.array_equal(actual, expected)
        direction = np.median(normal[y0:y1, x0:x1][actual], axis=0)
        direction /= max(np.linalg.norm(direction), 1e-9)
        assert np.allclose(direction, plane['normal_camera'], rtol=0, atol=1e-7)
        assert float(np.median(depth[y0:y1, x0:x1][actual])) == plane['relative_inverse_depth']
    counts = [row.get('counts') or [None, None] for row in read(handoff/'counts.json')['surfaces']]
    return dict(planes=len(planes), pixels=int(pixels[1:].sum()),
                counts_both_axes=sum(all(x is not None for x in d) for d in counts),
                counts_any_axis=sum(any(x is not None for x in d) for d in counts),
                field_sha256={kind: digest(cache/f'{kind}.npz') for kind in model_sha})


def arm_paths(case, key):
    name = case['id']
    if key in ('native', 'source'):
        arm = BASE / key
        charts = arm/'atlas/connected/sky'/name/'charts.json'
        handoff = arm/'sky'/name/'source-handoff'
    else:
        experiment, method = key.split('/', maxsplit=1)
        run = WORKTREES/f'expanded-wall-{experiment}'/'runs'/f'wall-{experiment}'
        arm = run/method
        source_method = method.removesuffix('-two-edges').removesuffix('-tight-edges')
        chart_dir = run/source_method/'charts'
        charts = chart_dir/'sky'/name/'charts.json'
        crop_arm = 'source' if method in ('source_resample', 'source_material') else 'native'
        handoff = BASE/crop_arm/'sky'/name/'source-handoff'
        if method == 'counts':
            charts = arm/'atlas/connected/sky'/name/'charts.json'
            handoff = arm/'sky'/name/'source-handoff'
    return charts, handoff, arm/'assembly/connected-atlas/sky'/name


def verify_ruler_handoff(case, key, packet, source_provenance, native_provenance):
    if key in ('rulers/source_resample', 'rulers/source_material'):
        assert packet['source_packet_sha256'] == digest(arm_paths(case, 'source')[0])
        assert source_provenance['atlas_sha256'] == packet['atlas_sha256']
    if key in ('rulers/noop', 'rulers/spacing', 'rulers/origin'):
        assert packet['native_packet_sha256'] == digest(arm_paths(case, 'native')[0])
        assert packet['ruler_arm'] == key.split('/')[1]
    if key == 'rulers/counts':
        _, original, _ = arm_paths(case, 'native')
        _, replacement, _ = arm_paths(case, key)
        assert digest(original/'surfaces.json') == digest(replacement/'surfaces.json') == native_provenance['surfaces_sha256']
        assert digest(original/'build-manifest.json') == digest(replacement/'build-manifest.json')
        old = read(original/'counts.json')
        new = read(replacement/'counts.json')
        old_rows = old.pop('surfaces')
        new_rows = new.pop('surfaces')
        assert old == new
        for prior, after in zip(old_rows, new_rows, strict=True):
            updated = after.pop('counts', None)
            previous = prior.pop('counts', None)
            assert prior == after
            if updated != previous:
                assert len(updated) == 2 and all(np.isfinite(x) and x > 0 for x in updated)


def verify_export(case, reference, key, allow_missing):
    charts, handoff, folder = arm_paths(case, key)
    receipt_path = folder/'score.json'
    if allow_missing and not receipt_path.is_file():
        return None
    receipt = read(receipt_path)
    packet = read(charts)
    original_sha = case['rgb_sha256']
    assert original_sha == receipt['source_sha256'] == packet['source_sha256']
    assert receipt['split'] == 'test' and receipt['dataset'] == case['dataset']
    assert receipt['mode'] == 'connected-atlas' and receipt['background'] == 'sky'
    assert receipt['reference_sha256'] == reference.hashes
    assert packet['atlas_sha256'] == ATLAS_SHA
    assert receipt['chart_sha256'] == digest(charts)
    provenance = floating_cloud_replay.check_handoff(packet, handoff, original_sha)
    assert all(receipt[k] == v for k, v in provenance.items())

    chart_rows = packet['charts']
    assembly = assemble(chart_rows, **SETTINGS_BY_ARM.get(key, SETTINGS))
    groups = assembly['groups']
    selected = groups[0] if groups else dict(blocks={}, origins={}, assigned=set())
    local = {p: 'minecraft:'+m.removeprefix('minecraft:') for p, m in selected['blocks'].items()}
    pack = folder/'reconstruction.zip'
    actual = decoded_pack(pack)
    assert receipt['pack_sha256'] == digest(pack)
    assert actual == local
    saved_placement = read(folder/'placements.json')
    def position_map(rows):
        return {tuple(xyz): material for xyz, material in rows}
    assert actual == position_map(saved_placement['local'])
    assert receipt['charts_seen'] == len(chart_rows)
    assert receipt['oriented_charts'] == assembly['oriented_charts']
    assert receipt['kept_charts'] == len(selected['origins'])
    assert receipt['components'] == len(groups)
    assert receipt['component_sizes'] == [len(group['blocks']) for group in groups]
    assert receipt['selected_origins'] == {k: list(v) for k, v in selected['origins'].items()}
    assert all(receipt[k] == loads(dumps(assembly[k])) for k in ('joins', 'conflicts', 'depth_vetoes'))
    assert receipt['relative_depth_field_sha256'] is None

    signed = key.startswith('grids/signed')
    assert receipt['reference_only'] == signed
    anchor = assembly['orientation_hint']
    yaw = None if not anchor else yaw_from_anchor_normal(
        anchor['normal_camera'], axis_vector(anchor['local_normal_axis']), reference.pose)
    assert receipt['yaw'] == yaw
    quarter = 0 if signed or not actual else (None if not yaw else yaw['yaw_quarters'])
    score, aligned = evaluate_relative_voxels(actual, reference, yaw_quarters=quarter, relaxed_yaw=not signed,
                                              assigned=selected['assigned'])
    assert receipt['score'] == score
    assert aligned == position_map(saved_placement['aligned'])
    display = score['fixed'] or score['relaxed_yaw']
    pack_metrics = {key: receipt[key] for key in ('charts_seen', 'oriented_charts', 'kept_charts', 'components', 'pack_sha256')}
    return dict(score=display, fixed=score['fixed'] is not None, provenance=provenance,
                chart_sha256=receipt['chart_sha256'], reference_sha256=reference.hashes,
                atlas_sha256=packet['atlas_sha256'],
                positions=aligned, **pack_metrics)


def registration_delta(baseline, candidate, truth):
    before = baseline['positions']
    current = candidate['positions']
    gained = current.keys()-before.keys()
    lost = before.keys()-current.keys()
    changed_material = {p for p in (before.keys() & current.keys()) if before[p] != current[p]}
    return dict(recovered=len(gained & truth.keys()), lost=len(lost & truth.keys()),
                extras_added=len(gained-truth.keys()), extras_removed=len(lost-truth.keys()),
                exact_ids_gained=sum(current.get(p) == truth[p] and before.get(p) != truth[p]
                                     for p in (gained | changed_material) & truth.keys()),
                exact_ids_lost=sum(before.get(p) == truth[p] and current.get(p) != truth[p]
                                   for p in (lost | changed_material) & truth.keys()),
                rotation_changed=candidate['score']['yaw_quarters'] != baseline['score']['yaw_quarters'],
                translation_changed=candidate['score']['translation'] != baseline['score']['translation'])


def score_case(job):
    case, proof, keys, validate_masks, allow_missing = job
    cv2.setNumThreads(1)
    case_id = case['id']
    sha = case['rgb_sha256']
    assert sha == digest(Path(case['variants']['sky']['image']))
    with with_v2_framing(case, proof):
        reference = expanded_scores.load_reference(case)
    scores = {key: verify_export(case, reference, key, allow_missing) for key in keys}
    for key, export in scores.items():
        if export and key.startswith('rulers/'):
            packet = read(arm_paths(case, key)[0])
            native_charts = arm_paths(case, 'native')[0]
            source_charts = arm_paths(case, 'source')[0]
            verify_ruler_handoff(case, key, packet, read(source_charts), read(native_charts))
    material, resampled = (scores.get(k) for k in ('rulers/source_material', 'rulers/source_resample'))
    if material and resampled:
        assert material['positions'].keys() == resampled['positions'].keys()
        assert material['score']['geometry'] == resampled['score']['geometry']
    pair = read(BASE/'pairs'/case['dataset']/(case_id+'.json'))
    assert pair['source_sha256'] == sha
    for key in {'source', 'native'} & scores.keys():
        assert scores[key]['score'] == pair[key]['score']
        assert scores[key]['pack_sha256'] == pair[key]['pack_sha256']
        assert scores[key]['charts_seen'] == pair[key]['charts']
    masks = native_masks(case, arm_paths(case, 'native')[1]) if validate_masks else None
    if masks:
        assert masks['planes'] == pair['native']['planes']
    changes = {}
    for key, result in scores.items():
        if not result:
            continue
        controls = {'native', CONTROLS.get(key)} - {None, key}
        changes[key] = {control: registration_delta(scores[control], result, reference.truth)
                        for control in sorted(controls) if scores.get(control)}
    for result in scores.values():
        if result:
            del result['positions']
    return dict(id=case_id, dataset=case['dataset'], source_sha256=sha, scores=scores, changes=changes, masks=masks)


def summarize(records, keys):
    cohorts = {}
    for dataset in (*DATASETS, 'all'):
        batch = [r for r in records if dataset in ('all', r['dataset'])]
        summaries = {}
        for key in keys:
            available = [r for r in batch if r['scores'][key]]
            if not available:
                continue
            arm = [r['scores'][key] for r in available]
            scores = [r['score'] for r in arm]
            others, changes = Counter(), {}
            for row, entry in zip(available, arm, strict=True):
                score = entry['score']
                counts = score['scores']
                others.update(assigned=counts['assigned'], fallback=counts['fallback'], fixed=entry['fixed'],
                              assigned_correct=counts['assigned_correct'], assigned_positions=counts['assigned_positions'],
                              exact_ids=counts['material']['correct'], correct_positions=counts['material']['positions'],
                              charts_seen=entry['charts_seen'], kept_charts=entry['kept_charts'],
                              oriented_charts=entry['oriented_charts'], components=entry['components'])
                for control, delta in row['changes'].get(key, {}).items():
                    changes.setdefault(control, Counter()).update(delta)
            summaries[key] = dict(views=len(arm), metrics=pooled(scores), totals=dict(others), changes=changes)
        cohorts[dataset] = summaries
    return cohorts


def extend(report, records, keys):
    assert report['revision'] == REVISION and report['atlas_sha256'] == ATLAS_SHA
    assert report['settings'] == SETTINGS and report['settings_by_arm'] == SETTINGS_BY_ARM
    previous = {row['id']: row for row in report['cases']}
    assert {row['id'] for row in records} == previous.keys()
    for row in records:
        prior = previous[row['id']]
        assert prior['dataset'] == row['dataset'] and prior['source_sha256'] == row['source_sha256']
        for key in prior['scores'].keys() & row['scores'].keys():
            assert prior['scores'][key] == row['scores'][key]
        row['scores'] = prior['scores'] | row['scores']
        for key, comparisons in prior['changes'].items():
            row['changes'][key] = comparisons | row['changes'].get(key, {})
        row['masks'] = row['masks'] or prior['masks']
    return list(dict.fromkeys([*report['cohorts']['all'], *keys]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', default=['source', 'native'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--out', default='baseline-audit.json.gz')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--allow-missing', action='store_true')
    parser.add_argument('--extend', action='store_true', help='Append rechecked arms to an existing full audit')
    args = parser.parse_args()
    jobs = [(case, proof, args.arms, args.masks, args.allow_missing) for case, proof in cases()]
    with ProcessPoolExecutor(args.workers) as executor:
        records = list(executor.map(score_case, jobs))
    target = OUT/args.out
    keys = extend(read(target), records, args.arms) if args.extend else args.arms
    summary = summarize(records, keys)
    save(target, dict(revision=REVISION, cohorts=summary, settings=SETTINGS, settings_by_arm=SETTINGS_BY_ARM,
                      atlas_sha256=ATLAS_SHA, manifests=DATASETS, cases=records))
    for dataset, arms in summary.items():
        for key, metrics in arms.items():
            rates = ' '.join(f'{name}={v["f1"]*100:.3f}%' for name, v in metrics['metrics'].items())
            print(dataset, key, metrics['views'], rates, metrics['totals'])
    if args.masks:
        totals = Counter()
        for record in records:
            totals.update({k:v for k,v in record['masks'].items() if k != 'field_sha256'})
        print('verified native masks', dict(totals))


if __name__ == '__main__':
    main()
