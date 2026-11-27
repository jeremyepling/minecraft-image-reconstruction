"""Independently decode and score the saved atlas and lattice replacements."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from zipfile import ZipFile

os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

import numpy as np

from benchmark import expanded_fixed_handoffs as fixed, expanded_scores as grader, typed_structures as typed
from benchmark.camera_metrics import registration
from benchmark.expanded_baseline import COHORTS, MEASURED_REVISION, ROOT, SOURCE, digest, frames, read, save
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers


ORDINARY = ROOT / 'runs/expanded-baseline-01'
RELEASED = ROOT / 'benchmark/reviews/expanded-baseline-01'
ATLAS = ROOT.parent / 'view-conditioned-atlas/runs/view-conditioned-atlas'
GRID = ROOT.parent / 'full-block-lattice/runs/full-block-lattice-01'
JOINT = ROOT.parent / 'view-atlas-full-block-grid/runs/view-atlas-lattice-01'
FIXED_ATLAS_INPUTS = ROOT.parent / 'typed-structure-scores/runs/typed-structures/inputs'
REVIEW = ROOT / 'benchmark/reviews/view-atlas-factorial-01'
DATASETS = {grader.FIXED_DATASET: 104, **COHORTS}
ARMS = ('C', 'C_point', 'A', 'G', 'G_point', 'G+A', 'I', 'I_point', 'I+A')
PAIRINGS = ('A/C', 'A/C_point', 'G/C', 'G+A/G', 'G+A/G_point', 'G+A/A', 'I/G',
            'I+A/I', 'I+A/I_point', 'I+A/G', 'I+A/A', 'I+A/G+A')
RESCUE = {'G_rescue': 'G', 'G+A_rescue': 'G+A', 'I_rescue': 'I', 'I+A_rescue': 'I+A'}


def frame_key(row):
    return row['dataset'], row['id']


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


@lru_cache(maxsize=1)
def fixed_sites():
    return read(typed.REVIEW / 'sites.json.gz')


def baseline(entry, control, ordinary):
    dataset, key = frame_key(entry)
    if dataset == grader.FIXED_DATASET:
        _, _, archive_path = fixed._release()
        with ZipFile(archive_path) as archive:
            pack = archive.read(f'{key}/reconstruction.zip')
        sites = fixed_sites()[key]
        methods = {tuple(r[:3]): r[4] for r in sites['rows']}
        require(sites['pack_sha256'] == control['pack_sha256'], 'Fixed site pack differs from released control')
        status = fixed._jobs()[key][3]['status']
        image = entry['rgb']
    else:
        folder = ordinary / dataset / key
        build, receipt = read(folder / 'build-manifest.json'), read(folder / 'receipt.json')
        require(receipt['revision'] == MEASURED_REVISION and receipt['source_sha256'] == control['source_sha256'],
                'Ordinary receipt does not match frozen source and revision')
        pack = (folder / 'reconstruction.zip').read_bytes()
        require(receipt['pack_sha256'] == control['pack_sha256'], 'Ordinary receipt pack differs from control')
        methods = {(row['x'], row['y'], row['z']): row['material_source']
                   for row in read(folder / 'placements.json')['blocks']}
        status = build['placement']
        image = Path(entry['frame']) / 'rgb.png'
    require(digest(image) == control['source_sha256'] == entry['rgb_sha256'], 'Source checksum differs')
    require(sha256(pack).hexdigest() == control['pack_sha256'], 'Baseline pack checksum differs')
    blocks = decoded_pack(BytesIO(pack))
    require(blocks.keys() == methods.keys(), 'Baseline assignment differs from its export')
    return pack, blocks, methods, status


def reference(entry):
    if entry['dataset'] != grader.FIXED_DATASET:
        return grader.load_reference(entry), None
    labels, raster, transform = fixed.fixed_reference(entry)
    truth = {tuple(b['xyz_world']): b['block_id'] for b in labels['blocks']}
    fields = fixed._jobs()[entry['id']][1]['files']
    hashes = {name: fields[name] for name in ('camera.json', 'labels.json', 'labels.npz', 'world-readback.json') if name in fields}
    ref = grader.FrameReference(truth, horizontal_slivers(labels, raster), labels['camera'], {'full_cube': set(truth)}, hashes)
    return ref, transform


def package(path, expected):
    payload = path.read_bytes()
    require(sha256(payload).hexdigest() == expected, f'Pack checksum differs for {path}')
    return payload, decoded_pack(BytesIO(payload))


def assignment(path):
    rows = read(path)
    methods = {tuple(row[:3]): row[3] for row in rows}
    require(len(methods) == len(rows), f'Repeated material assignment in {path}')
    return methods


def same_projection(left, right):
    a, b = left.get('P'), right.get('P')
    if a is None or b is None:
        return a is None and b is None
    a, b = (np.asarray(p, dtype=float) for p in (a, b))
    return np.allclose(a / np.linalg.norm(a[2, :3]), b / np.linalg.norm(b[2, :3]), rtol=1e-10, atol=1e-8)


def grade_pack(payload, blocks, status, methods, ref, transform, alignments):
    ready = grader.camera_fields(status)['camera_usable']
    require(ready or not blocks, 'Populated export without a source camera')
    require(blocks.keys() == methods.keys(), 'Assignment locations differ from decoded inventory')
    if transform is None and ready:
        transform = registration(np.asarray(status['P'], float), ref.pose)
    camera_cells = typed.aligned_sites(list(blocks), transform, (0, 0, 0)) if blocks else []
    alignment_key = frozenset(camera_cells)
    if alignment_key not in alignments:
        alignments[alignment_key] = best_translation_fast(alignment_key, ref.truth)
    translation, overlap = alignments[alignment_key]
    world_cells = [tuple(x + d for x, d in zip(position, translation, strict=True)) for position in camera_cells]

    def score_at(sites):
        mapping = dict(zip(blocks, sites, strict=True))
        named = {mapping[p]: name for p, name in blocks.items()}
        assigned = {mapping[p] for p, source in methods.items() if source != 'fallback'}
        return typed.geometry_score(named, ref.truth, ref.optional), typed.score(named, ref.truth, ref.optional, assigned), named

    geometry, scores, names = score_at(world_cells)
    registered, registered_scores, _ = score_at(camera_cells)
    require(geometry['strict']['correct'] == overlap, 'Selected integer translation has inconsistent overlap')
    scored = dict(geometry=geometry, scores=scores, registered=registered, registered_scores=registered_scores,
                  targets=grader.target_breakdown(names, ref), translation=list(translation), exported=len(blocks),
                  pack_sha256=sha256(payload).hexdigest(), camera_usable=ready, camera_source=status.get('camera_source'),
                  camera_reason=status.get('atlas_camera_reason') or status.get('reason'))
    return scored, names


def _same_cells(parent, child, candidate):
    require(parent.keys() == child.keys(), f'{candidate} changed occupied local coordinates')


def _id_changes(before, after):
    _same_cells(before, after, 'Material-only policy')
    return sum(before[p] != after[p] for p in before)


def placement_changes(before, after, ref):
    added, removed = after.keys() - before.keys(), before.keys() - after.keys()
    truth = ref.truth
    hit_add, hit_remove = added & truth.keys(), removed & truth.keys()
    new_exact = {p for p in after.keys() & truth.keys() if after[p] == truth[p] and before.get(p) != truth[p]}
    old_exact = {p for p in before.keys() & truth.keys() if before[p] == truth[p] and after.get(p) != truth[p]}
    partial = truth.keys() - ref.by_kind.get('full_cube', set())
    return dict(recovered=len(hit_add), lost=len(hit_remove), attached_required=len(hit_add - ref.optional),
                lost_required=len(hit_remove - ref.optional), extras_added=len(added - truth.keys()),
                extras_removed=len(removed - truth.keys()), partial_recovered=len(hit_add & partial),
                partial_lost=len(hit_remove & partial), exact_gained=len(new_exact), exact_lost=len(old_exact),
                shared_position_exact_gained=len(new_exact - added), shared_position_exact_lost=len(old_exact - removed),
                required_exact_gained=len(new_exact - ref.optional), required_exact_lost=len(old_exact - ref.optional))


def atlas_only(entry, root, ordinary, c, c_methods, c_status, control):
    dataset, key = frame_key(entry)
    info = read(root / 'cases' / dataset / f'{key}.json')
    expected = dict(dataset=dataset, id=key, split=entry['split'], source_sha256=entry['rgb_sha256'],
                    baseline_pack_sha256=control['pack_sha256'], exported=len(c), geometry=control['geometry'],
                    baseline=control['scores'], cells_sha256=sha256(str(sorted(c)).encode()).hexdigest())
    require(all(info[k] == v for k, v in expected.items()), 'Standalone atlas receipt does not match its control')
    if dataset == grader.FIXED_DATASET:
        p_input = read(FIXED_ATLAS_INPUTS / f'{key}.json')['camera']
    else:
        p_input = read(ordinary / dataset / key / 'handoffs/material-input.json.gz')['camera']
    require(same_projection(p_input or {}, c_status), 'Standalone atlas used a different ordinary camera')
    a_pack, a = package(root / 'packs' / dataset / f'{key}.zip', info['pack_sha256'])
    _same_cells(c, a, 'A')
    details = read(root / 'details' / dataset / f'{key}.json.gz')
    require(len(details) == info['evaluated_cells'], 'Standalone eligible count differs')
    detail_cells = {tuple(r['xyz']) for r in details}
    require(len(detail_cells) == len(details) and detail_cells <= c.keys(), 'Standalone eligibility coordinates differ')
    point_ids = dict(c)
    for rec in details:
        p = tuple(rec['xyz'])
        require(c[p] == 'minecraft:' + rec['old'] and a[p] == 'minecraft:' + rec['view'], 'Standalone ID evidence differs')
        point_ids[p] = 'minecraft:' + rec['static']
    require(_id_changes(c, a) == info['changed'] and _id_changes(point_ids, a) == info['changed_from_point'],
            'Standalone changed ID counts differ')
    require(all(c[p] == a[p] for p in c.keys() - detail_cells), 'Standalone changed ID without eligible evidence')
    point_pack, point = typed.typed_pack({p: name.removeprefix('minecraft:') for p, name in point_ids.items()})
    base = (c_status, c_methods)
    policy = dict(eligible=len(details), occupied=len(c), point_edits=_id_changes(c, point),
                  view_edits=_id_changes(c, a), differs_from_point=_id_changes(point, a),
                  view_edited_inherited_fallback=sum(c[p] != a[p] and c_methods[p] == 'fallback' for p in c))
    return {'C_point': (point_pack, point, *base), 'A': (a_pack, a, *base)}, info, policy


def grid_arms(entry, control, ref, roots, ordinary_status, c):
    original, combined = (root.joinpath(*frame_key(entry)) for root in roots)
    final = combined / 'final'
    g_info, i_info, receipt = map(read, (original / 'result.json', combined / 'result.json', final / 'receipt.json'))
    for record in (g_info, i_info):
        require(all(record[k] == entry[k] for k in ('dataset', 'id', 'split', 'dataset_manifest_sha256', 'rgb_sha256')),
                'Lattice case does not match frozen capture metadata')
        require(record['grader_sha256'] == ref.hashes, 'Lattice grader fingerprints differ')
        require(all(record['baseline'][k] == control[k] for k in ('pack_sha256', 'geometry', 'scores', 'translation')),
                'Lattice arm carries a different ordinary control')
    check = (*frame_key(entry), entry['rgb_sha256'], control['pack_sha256'])
    require(tuple(receipt[k] for k in ('dataset', 'id', 'source_sha256', 'baseline_pack_sha256')) == check,
            'Final matcher receipt failed source and baseline verification')
    require(receipt['policies']['baseline']['sha256'] == control['pack_sha256'], 'Final matcher baseline pack differs')
    rows, policies = {}, {}
    persisted = read(final / 'metrics.json')['metrics']
    for name, prefix, selected, labels in (
        ('grid', 'G', g_info, ('grid', 'grid_point', 'grid_view')),
        ('joint', 'I', i_info, ('joint_anchor', 'joint_point', 'joint_view')),
    ):
        status = read(final / f'{name}-camera.json')
        if not selected['rescued']:
            require(same_projection(status, ordinary_status), f'{prefix} changed a supported camera')
        base, point, view = labels
        parts = {}
        for report_key, artifact in zip((prefix, f'{prefix}_point', f'{prefix}+A'), labels, strict=True):
            policy = receipt['policies'][artifact]
            payload, blocks = package(final / f'{artifact}.zip', policy['sha256'])
            methods = assignment(final / f'{artifact}-assignments.json.gz')
            require(policy['participating'] == len(blocks), f'{artifact} occupied count differs')
            require(policy['fallback'] == sum(v == 'fallback' for v in methods.values()), f'{artifact} fallback count differs')
            rows[report_key] = (payload, blocks, status, methods)
            parts[artifact] = blocks
        require(receipt['policies'][base]['sha256'] == selected['source']['pack_sha256'], f'{prefix} selected geometry pack differs')
        require(all(persisted[base][k] == selected['source'][k] for k in ('scores', 'geometry', 'translation')),
                f'{prefix} final replay differs from its selected lattice score')
        require(all(parts[base].get(p) == v for p, v in c.items()), f'{prefix} changed a retained ordinary cell')
        eligible = read(final / f'{name}-eligibility.json.gz')
        expected_cells = {tuple(r['xyz']) for r in eligible}
        require(len(expected_cells) == len(eligible) == receipt['policies'][base]['eligible'], f'{prefix} eligibility count differs')
        require(expected_cells <= parts[base].keys(), f'{prefix} eligibility uses unsupported cells')
        for rec in eligible:
            p = tuple(rec['xyz'])
            require(tuple(parts[k][p] for k in labels) == tuple('minecraft:' + rec[k] for k in ('old', 'static', 'view')),
                    f'{prefix} point and view evidence differs')
        for variant in (point, view):
            require(_id_changes(parts[base], parts[variant]) == receipt['policies'][variant]['changed'],
                    f'{variant} changed ID count differs')
            require(all(parts[base][p] == parts[variant][p] for p in parts[base].keys() - expected_cells),
                    f'{variant} changed ID without eligible evidence')
        policies[prefix] = dict(eligible=len(eligible), occupied=len(parts[base]),
                                point_edits=_id_changes(parts[base], parts[point]),
                                view_edits=_id_changes(parts[base], parts[view]),
                                differs_from_point=_id_changes(parts[point], parts[view]),
                                view_edited_inherited_fallback=receipt['policies'][base]['fallback'] - receipt['policies'][view]['fallback'],
                                rescues=int(selected['rescued']), accepted_lattice=int(selected['changed']),
                                anchors=selected['evidence'].get('confirmed', 0),
                                **{f'depth_{k}': n for k, n in selected['evidence'].get('cuts', {}).items()})
    comparable = dict(zip(('G', 'G_point', 'G+A', 'I', 'I_point', 'I+A'),
                          ('grid', 'grid_point', 'grid_view', 'joint_anchor', 'joint_point', 'joint_view'), strict=True))
    expected = {arm: persisted[artifact] for arm, artifact in comparable.items()}
    return rows, expected, policies


def score_frame(task):
    entry, control, ordinary, atlas, grid, joint = task
    try:
        return score_frame_verified(entry, control, ordinary, atlas, grid, joint)
    except Exception as error:
        raise ValueError(f'{entry["dataset"]}/{entry["id"]}: {error}') from error


def score_frame_verified(entry, control, ordinary, atlas, grid, joint):
    c_pack, c, methods, status = baseline(entry, control, ordinary)
    ref, transform = reference(entry)
    require(control.get('reference_sha256', ref.hashes) == ref.hashes, 'Ordinary control uses different native targets')
    require(len(ref.truth) == control['visible_truth'] and len(ref.truth) - len(ref.optional) == control['required_truth'],
            'Ordinary target counts differ')
    data = {'C': (c_pack, c, status, methods)}
    material, a_info, a_policy = atlas_only(entry, atlas, ordinary, c, methods, status, control)
    lattice, expected_lattice, lattice_policy = grid_arms(entry, control, ref, (grid, joint), status, c)
    data.update(material | lattice)
    alignments, scored, aligned = {}, {}, {}
    for arm in ARMS:
        scored[arm], aligned[arm] = grade_pack(*data[arm], ref, transform, alignments)
    for k in ('pack_sha256', 'geometry', 'scores', 'translation', 'camera_usable'):
        require(scored['C'][k] == control[k], f'Independent ordinary score differs at {k}')
    for arm, receipt_key in (('C_point', 'point'), ('A', 'source')):
        require(scored[arm]['scores'] == a_info[receipt_key], f'Independent {arm} ID score differs')
    for arm, expected in expected_lattice.items():
        for k in ('pack_sha256', 'geometry', 'scores', 'targets', 'translation', 'camera_usable', 'registered', 'registered_scores'):
            require(scored[arm][k] == expected[k], f'Independent {arm} score differs at {k}')
    for group in (('C', 'C_point', 'A'), ('G', 'G_point', 'G+A'), ('I', 'I_point', 'I+A')):
        for candidate in group[1:]:
            base = group[0]
            for k in ('geometry', 'registered', 'translation'):
                require(scored[candidate][k] == scored[base][k], f'{candidate} changes {k} in a fixed-cell pair')
            require(aligned[candidate].keys() == aligned[base].keys(), f'{candidate} changes aligned occupied cells')
    for arm, source in RESCUE.items():
        scored[arm] = scored['C' if scored['C']['camera_usable'] else source]
        aligned[arm] = aligned['C' if scored['C']['camera_usable'] else source]
    pairs = PAIRINGS + tuple(f'{arm}/C' for arm in RESCUE) + ('G+A_rescue/G_rescue', 'I+A_rescue/I_rescue', 'I+A_rescue/G+A_rescue')
    deltas = {pair: placement_changes(aligned[before], aligned[after], ref)
              for pair in pairs for after, before in [pair.split('/')]}
    header = {k: entry[k] for k in ('dataset', 'id', 'split', 'layout', 'rgb_sha256', 'dataset_manifest_sha256')}
    return dict(header, arms=scored, deltas=deltas, matcher=dict(C=a_policy, **lattice_policy), reference_sha256=ref.hashes,
                ordinary_target=len(ref.truth), required_target=len(ref.truth) - len(ref.optional))


def pool_arm(arms):
    data = {name: typed.pooled_scores([a[name] for a in arms])
            for name in ('geometry', 'scores', 'registered', 'registered_scores')}
    targets = defaultdict(Counter)
    for arm in arms:
        for name, values in arm['targets'].items():
            targets[name].update(values)
    data['targets'] = dict(targets)
    data['camera_usable'] = sum(a['camera_usable'] for a in arms)
    data['camera_usable_nonempty'] = sum(a['camera_usable'] and a['exported'] > 0 for a in arms)
    data['camera_rejected_strict_misses'] = sum(a['geometry']['strict']['missed'] for a in arms if not a['camera_usable'])
    data['camera_rejected_facade_misses'] = sum(a['geometry']['facade']['missed'] for a in arms if not a['camera_usable'])
    data['exported'] = sum(a['exported'] for a in arms)
    data['empty'] = sum(a['exported'] == 0 for a in arms)
    return data


def metric_change(previous, current):
    geometry = {metric: 100 * (current['geometry'][metric]['f1'] - previous['geometry'][metric]['f1'])
                for metric in ('strict', 'facade')}
    typed_f1 = {metric: 100 * (current['scores'][metric]['f1'] - previous['scores'][metric]['f1'])
                for metric in ('strict', 'facade')}
    return dict(geometry_f1_pp=geometry, position_and_id_f1_pp=typed_f1,
                exact_id=current['scores']['strict']['correct'] - previous['scores']['strict']['correct'],
                required_exact_id=current['scores']['facade']['correct'] - previous['scores']['facade']['correct'])


def pooled(rows):
    arms = {arm: pool_arm([row['arms'][arm] for row in rows]) for arm in (*ARMS, *RESCUE)}
    pairs = {}
    for pair in rows[0]['deltas']:
        after, before = pair.split('/')
        deltas = Counter()
        for row in rows:
            deltas.update(row['deltas'][pair])
        pairs[pair] = dict(deltas, **metric_change(arms[before], arms[after]))
    for pair in ('A/C_point', 'G+A/G_point', 'I+A/I_point'):
        after, before = pair.split('/')
        require(arms[after]['geometry'] == arms[before]['geometry'], f'Pooling changed cells in {pair}')
    final, standalone = pairs['G+A/G'], pairs['A/C']
    footprint_final, footprint_standalone = pairs['G+A/G_point'], pairs['A/C_point']
    interaction = dict(
        matched_final_policy={metric: final['position_and_id_f1_pp'][metric] - standalone['position_and_id_f1_pp'][metric]
                              for metric in ('strict', 'facade')},
        matched_footprint_vs_point={metric: footprint_final['position_and_id_f1_pp'][metric] - footprint_standalone['position_and_id_f1_pp'][metric]
                                    for metric in ('strict', 'facade')},
        matched_final_correct_ids=final['exact_id'] - standalone['exact_id'],
        matched_footprint_vs_point_correct_ids=footprint_final['exact_id'] - footprint_standalone['exact_id'],
    )
    policies = {}
    for name in ('C', 'G', 'I'):
        counts = Counter()
        for row in rows:
            counts.update(row['matcher'][name])
        policies[name] = dict(counts, views_with_eligible=sum(row['matcher'][name]['eligible'] > 0 for row in rows),
                              views_with_view_edits=sum(row['matcher'][name]['view_edits'] > 0 for row in rows),
                              views_with_different_point_result=sum(row['matcher'][name]['differs_from_point'] > 0 for row in rows))
    return dict(tested=len(rows), visible_targets=sum(r['ordinary_target'] for r in rows),
                required_targets=sum(r['required_target'] for r in rows), arms=arms, paired=pairs,
                matcher=policies, descriptive_interaction_pp=interaction)


def summarize(rows):
    cohorts = {}
    for dataset, available in DATASETS.items():
        group = [row for row in rows if row['dataset'] == dataset]
        if not group:
            continue
        cohorts[dataset] = dict(pooled(group), available=available, splits={}, camera_groups={})
        for split in sorted({row['split'] for row in group}):
            selected = [row for row in group if row['split'] == split]
            cohorts[dataset]['splits'][split] = pooled(selected)
        groups = {
            'ordinary_camera_supported': [row for row in group if row['arms']['C']['camera_usable']],
            'ordinary_camera_rejected': [row for row in group if not row['arms']['C']['camera_usable']],
            'newly_supported_by_either_lattice': [row for row in group if not row['arms']['C']['camera_usable']
                                                 and any(row['arms'][arm]['camera_usable'] for arm in ('G', 'I'))],
        }
        for name, selected in groups.items():
            if selected:
                cohorts[dataset]['camera_groups'][name] = pooled(selected)
    return cohorts


def verify_policy_source(atlas, joint):
    names = ('view_conditioned_atlas.py', 'view_conditioned_atlas_match.py')
    standalone, integrated = (root.parents[1] / 'benchmark' for root in (atlas, joint))
    def digest_source(path):
        return sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
    hashes = {name: digest_source(standalone / name) for name in names}
    require(all(digest_source(integrated / name) == digest_value for name, digest_value in hashes.items()),
            'Standalone and lattice final matchers differ in code')
    return hashes


def prepare(args):
    control_path = RELEASED / 'cases.json'
    control_bytes = control_path.read_bytes()
    require(control_bytes == (args.ordinary / 'scores/cases.json').read_bytes(),
            'Ordinary control cases differ from the published release')
    controls = {frame_key(row): row for row in read(control_path)}
    roster = list(fixed.fixed_frames()) + list(frames(args.source, COHORTS))
    entries = {frame_key(row): row for row in roster}
    require(len(entries) == len(roster) and entries.keys() == controls.keys(), 'Ordinary and dataset frame lists differ')
    require(Counter(row['dataset'] for row in roster) == DATASETS, 'Unexpected cohort counts')
    require(Counter((row['dataset'], row['split']) for row in roster) ==
            Counter((row['dataset'], row['split']) for row in controls.values()), 'Dataset split counts differ')
    for key, row in entries.items():
        ordinary = controls[key]
        require(row['split'] == ordinary['split'] and row['rgb_sha256'] == ordinary['source_sha256'],
                f'Capture roster differs for {key}')
        known_manifest = ordinary.get('dataset_manifest_sha256', row['dataset_manifest_sha256'])
        require(row['dataset_manifest_sha256'] == known_manifest, f'Capture manifest differs for {key}')
    implementation = verify_policy_source(args.atlas, args.joint)
    atlas = fixed._export_atlas().provenance
    original = read(args.grid.parents[1] / 'benchmark/reviews/full-block-lattice-01/manifest.json')
    combined = read(args.joint.parents[1] / 'benchmark/reviews/view-atlas-lattice-01/manifest.json')
    require(atlas == original['atlas'] == combined['atlas'], 'Material catalog fingerprints differ')
    return entries, controls, dict(control_cases_sha256=sha256(control_bytes).hexdigest(), matcher_sha256=implementation,
                                   atlas=atlas, original_lattice_release_sha256=original['cases_sha256'],
                                   revised_lattice_release_sha256=combined['cases_sha256'])


def verify_upstream_point(rows, joint):
    review = joint.parents[1] / 'benchmark/reviews/view-atlas-lattice-01'
    point_runs = joint.parent / 'view-atlas-lattice-point-01'
    proof = review / 'upstream-point-cases.json.gz'
    by_key = {frame_key(r): r for r in rows}
    evidence = read(proof)
    require(len(evidence) == len(by_key) and {frame_key(r) for r in evidence} == by_key.keys(),
            'Upstream point replay used a different frame roster')
    for record in evidence:
        key = frame_key(record)
        row = by_key[key]
        require(row['rgb_sha256'] == record['source_sha256'] and row['split'] == record['split'],
                f'Upstream replay used a different source view for {key}')
        for field in ('geometry', 'scores', 'pack_sha256', 'translation', 'registered', 'registered_scores'):
            require(record['finite'][field] == row['arms']['I'][field], f'Upstream finite score differs for {key}/{field}')
        folder = point_runs.joinpath(*key)
        upstream = read(folder / 'result.json')
        require(upstream['rgb_sha256'] == row['rgb_sha256'] and upstream['grader_sha256'] == row['reference_sha256'],
                f'Upstream point artifact used different visible targets for {key}')
        require(upstream['source'] == record['point'], f'Upstream point score manifest differs for {key}')
        if upstream['changed']:
            _, blocks = package(folder / 'proposal.zip', record['point']['pack_sha256'])
            require(len(blocks) == record['point']['exported'], f'Upstream point pack count differs for {key}')
        else:
            require(record['point']['pack_sha256'] == row['arms']['C']['pack_sha256'], f'Upstream point keep arm differs for {key}')
    released = read(review / 'upstream-point-control.json')
    cohorts = {}
    for name in DATASETS:
        group = [r for r in evidence if r['dataset'] == name]
        arms = {arm: {metric: typed.pooled_scores([r[arm][metric] for r in group])
                      for metric in ('geometry', 'scores', 'registered', 'registered_scores')}
                for arm in ('point', 'finite')}
        for arm, stats in arms.items():
            require(all(value == released[name][arm][key] for key, value in stats.items()),
                    f'Upstream point comparison pooling differs for {name}/{arm}')
        cohorts[name] = dict(tested=len(group), **arms,
                             change=metric_change(arms['point'], arms['finite']))
    return dict(cases_sha256=digest(proof), cohorts=cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', type=Path, default=ATLAS)
    parser.add_argument('--grid', type=Path, default=GRID)
    parser.add_argument('--joint', type=Path, default=JOINT)
    parser.add_argument('--ordinary', type=Path, default=ORDINARY)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=REVIEW)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--ids', nargs='+', help='Subset for a scoring smoke test')
    args = parser.parse_args()
    entries, controls, provenance = prepare(args)
    chosen = [(key, row) for key, row in entries.items() if not args.ids or row['id'] in args.ids]
    require(not args.ids or {row['id'] for _, row in chosen} == set(args.ids), 'Unrecognized frame in --ids')
    tasks = [(row, controls[key], args.ordinary, args.atlas, args.grid, args.joint) for key, row in chosen]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(score_frame, tasks, chunksize=1):
            rows.append(result)
            if len(rows) % 24 == 0 or len(rows) == len(tasks):
                print(f'Independently decoded and scored {len(rows)}/{len(tasks)} paired frames', flush=True)
    cohorts = summarize(rows)
    if not args.ids:
        saved = read(RELEASED / 'summary.json')['cohorts']
        for dataset, group in cohorts.items():
            for field in ('geometry', 'scores'):
                require(group['arms']['C'][field] == saved[dataset][field], f'Released pooled {field} differs for {dataset}')
        provenance['upstream_point'] = verify_upstream_point(rows, args.joint)
    save(args.out / 'cases.json.gz', rows)
    save(args.out / 'summary.json', dict(version='26.3', control_revision=MEASURED_REVISION, **provenance,
                                         cases_sha256=digest(args.out / 'cases.json.gz'), cohorts=cohorts))


if __name__ == '__main__':
    main()
