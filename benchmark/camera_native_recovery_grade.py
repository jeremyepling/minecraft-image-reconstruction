"""Independently audit source camera receipts and score decoded reconstruction packs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from benchmark import expanded_fixed_handoffs as fixed
from benchmark import expanded_scores as scores
from benchmark.camera_horizon_replay import camera_validation
from benchmark.camera_horizon_report import ground_validation
from benchmark.camera_model_controls import (
    DEFAULT_BUILDS, REPEAT_BUILDS, check, data_hash, fixed_reference, frozen_frames,
    indexes, inventory_hash, summary,
)
from benchmark.camera_model_entry_check import MATCHED
from benchmark.camera_model_grade import deltas, verify_grade
from benchmark.expanded_baseline import ROOT, STABLE, WORKTREES, digest, read, save
from benchmark.mask_blocks import decoded_pack


CONTROLS = STABLE / 'benchmark/reviews/camera-model-controls-01'
CANDIDATE = WORKTREES / 'camera-native-floor-inverse'
NATIVE_RUN = CANDIDATE / 'runs/camera-native-floor-inverse-01'
NATIVE_REVIEW = CANDIDATE / 'benchmark/reviews/camera-native-floor-inverse-01'
CORNER_REVIEW = WORKTREES / 'camera-recovery-next'
CORNER_AUDIT = CORNER_REVIEW / 'benchmark/reviews/camera-recovery-next-01/native-calibration.json.gz'
CORNER_AUDIT_SHA256 = '00cc794da64c36cd4c1106aab111e4a62b17ed10a525d483c462b5c1b96bbae1'
SILHOUETTE = WORKTREES / 'camera-silhouette-ruler/benchmark/reviews/camera-silhouette-ruler-01'
OUTPUT = ROOT / 'benchmark/reviews/camera-native-recovery-controls-01'
CACHE = ROOT / 'runs/camera-native-recovery-controls-01'
WEIGHTS_SHA256 = 'weights_sha256'
GEOCALIB_WEIGHTS = '86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac'
FLOOR_ATLAS = '49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950'


def policy_pack(entry, incumbent):
    key = entry['id']
    policy = incumbent['source_policy']
    if entry['dataset'] == scores.FIXED_DATASET:
        check(policy, 'ordinary', f'{key} fixed incumbent policy')
        return None
    if policy == 'horizon-repeat-floor':
        return REPEAT_BUILDS / key / 'repeat'
    if policy == 'boundary-budget-repeat':
        return MATCHED / entry['dataset'] / key
    if policy not in ('ordinary', 'abstained'):
        raise ValueError(f'{key} unrecognized incumbent policy {policy}')
    return DEFAULT_BUILDS / entry['dataset'] / key


def frozen_case(entry, inventory, incumbent, archived):
    key, dataset = entry['id'], entry['dataset']
    check(digest(entry['rgb']), inventory['rgb_sha256'], f'{key} actual RGB')
    check(entry['rgb_sha256'], inventory['rgb_sha256'], f'{key} frozen RGB')
    if dataset == scores.FIXED_DATASET:
        # The archived fixed reference loader verifies the frozen labels and frame.
        reference, _ = fixed_reference(entry)
        path = entry['rgb']
        camera = (path.with_name('camera.json') if entry['split'] in ('train', 'test', 'validation', 'regression')
                  else path.with_suffix('.json'))
        check(digest(camera), reference.hashes['camera.json'], f'{key} fixed pose')
        payload = archived.read(f'{key}/reconstruction.zip')
    else:
        reference = scores.load_reference(entry)
        payload = (policy_pack(entry, incumbent) / 'reconstruction.zip').read_bytes()
        check(reference.hashes, incumbent['reference_sha256'], f'{key} incumbent reference')
    check(reference.hashes, inventory['source_reference_sha256'], f'{key} reference fingerprints')
    check(inventory_hash(reference.truth), inventory['truth_sha256'], f'{key} visible ID truth')
    check(data_hash(sorted(reference.optional)), inventory['facade_exempt_sha256'], f'{key} facade exemptions')
    pack_hash = sha256(payload).hexdigest()
    decoded = decoded_pack(BytesIO(payload))
    decoded_hash = inventory_hash(decoded)
    check(pack_hash, incumbent['pack_sha256'], f'{key} retained pack bytes')
    check(len(decoded), incumbent['exported'], f'{key} retained export count')
    if incumbent['source_policy'] != 'boundary-budget-repeat':
        mode = 'repeat' if incumbent['source_policy'] == 'horizon-repeat-floor' else 'ordinary'
        check(inventory[mode]['zip_sha256'], pack_hash, f'{key} frozen pack receipt')
        check(inventory[mode]['decoded_sha256'], decoded_hash, f'{key} frozen decoded receipt')
    else:
        check(incumbent['decoded_sha256'], decoded_hash, f'{key} matched decoded receipt')
    return decoded_hash


def verify_controls(incumbent, entries, inventory, workers):
    # ZipFile serializes seeks internally. Hashing expanded references can proceed in parallel.
    with ZipFile(fixed._release()[2]) as archive, ThreadPoolExecutor(workers) as pool:
        def verify(entry):
            key = entry['id']
            return key, frozen_case(entry, inventory[key], incumbent[key], archive)

        decoded = dict(pool.map(verify, entries.values()))
    check(Counter(r['source_policy'] for r in incumbent.values()),
          Counter(ordinary=170, **{'horizon-repeat-floor': 40, 'boundary-budget-repeat': 8, 'abstained': 70}),
          'incumbent camera policy counts')
    return decoded


def proposal_gate(proposal):
    shifted = proposal.get('wrong_scale', ())
    return (len(proposal['native_correlation']) == 4 and len(shifted) == 4
            and min(proposal['native_correlation']) >= .60
            and min(proposal['native_correlation']) - max(shifted) >= .26)


def evidence(camera):
    result = {k: v for k, v in camera['evidence'].items() if k != 'refined_proposals'}
    result['refined_proposals'] = len(camera['evidence'].get('refined_proposals', ()))
    return result


def source_receipts(entry, run, native, audit, production):
    key = entry['id']
    filename = key + '.json'
    receipts = {arm: run / arm / 'metric' / filename for arm in ('boundary', 'geocalib')}
    proposals = {arm: read(path) for arm, path in receipts.items()}
    for arm, proposal in proposals.items():
        camera = proposal['camera']
        check(proposal['rgb_sha256'], entry['rgb_sha256'], f'{key} {arm} prediction RGB')
        check(proposal['arm'], arm, f'{key} {arm} receipt identity')
        check(native['arms'][arm]['file_sha256'], digest(receipts[arm]), f'{key} {arm} receipt checksum')
        if arm == 'boundary':
            check(proposal[WEIGHTS_SHA256], None, f'{key} model-free seed')
        else:
            path = WORKTREES / 'camera-geocalib/runs/camera-geocalib-01/proposals' / filename
            seed = read(path)
            check(seed['rgb_sha256'], entry['rgb_sha256'], f'{key} learned seed RGB')
            check(seed[WEIGHTS_SHA256], GEOCALIB_WEIGHTS, f'{key} learned seed checkpoint')
            check(proposal[WEIGHTS_SHA256], GEOCALIB_WEIGHTS, f'{key} learned weights checksum')
        refined = camera['evidence'].get('refined_proposals', ())
        selected = [p for p in refined if proposal_gate(p)]
        for p in refined:
            check(p['accepted'], proposal_gate(p), f'{key} {arm} separated-region and scale gate')
            check(p['atlas_sha256'], FLOOR_ATLAS, f'{key} {arm} native floor atlas')
        check(len(selected), int(camera['P'] is not None), f'{key} {arm} acceptance')
        if selected:
            check(selected[0]['P'], camera['P'], f'{key} {arm} selected source projection')
        check(native['arms'][arm]['accepted'], bool(selected), f'{key} {arm} released acceptance')
    selected_path = run / 'metric' / filename
    model_free = proposals['boundary']['camera']['P'] is not None
    arm = 'geocalib' if not model_free and proposals['geocalib']['camera']['P'] is not None else 'boundary'
    selected = proposals[arm]
    check(read(selected_path), selected, f'{key} boundary-first composition')
    camera = selected['camera']
    for field, expectation in (
            ('dataset', entry['dataset']), ('split', entry['split']), ('rgb_sha256', entry['rgb_sha256']),
            ('camera', camera), ('choice', arm)):
        check(native[field], expectation, f'{key} native source {field}')
    for field, expectation in (
            ('source_sha256', entry['rgb_sha256']), ('split', entry['split']), ('seed', arm),
            ('receipt_sha256', digest(selected_path)), ('accepted', camera['P'] is not None),
            ('reason', camera['reason'])):
        check(audit[field], expectation, f'{key} independent corner gauge {field}')
    for field, expectation in (
            ('rgb_sha256', entry['rgb_sha256']), ('split', entry['split']), ('accepted', model_free),
            ('camera_source', proposals['boundary']['camera'].get('camera_source')),
            ('exact_matrix_or_abstention', True), ('research_receipt_sha256', digest(receipts['boundary']))):
        check(production[field], expectation, f'{key} production camera {field}')
    if camera['P'] is not None:
        for field in ('registered', 'relative'):
            check(native[field], audit[field], f'{key} {field} gauge')
    return dict(boundary=proposals['boundary']['camera'], selected=selected, paths=receipts,
                receipt_sha256=digest(selected_path), corner_audit=audit)


def load_corner_scorer():
    path = CORNER_REVIEW / 'benchmark/camera_relative_calibration.py'
    spec = spec_from_file_location('camera_relative_calibration', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.native_corner_score


def verify_export(entry, inventory, source, run, cache, native_corner_score):
    key = entry['id']
    receipt = run / 'scores' / (key + '.json')
    if not receipt.is_file():
        return None
    build = run / 'builds' / entry['dataset'] / key
    manifest_path = build / 'build-manifest.json'
    placement = read(manifest_path)['placement']
    camera = source['selected']['camera']
    check(placement['P'], camera['P'], f'{key} source projection through placement')
    check(placement['camera_source'], camera['camera_source'], f'{key} source policy through placement')
    stamp = dict(score=digest(receipt), manifest=digest(manifest_path),
                 placements=digest(build / 'placements.json'), pack=digest(build / 'reconstruction.zip'),
                 source=source['receipt_sha256'], grader=digest(Path(__file__)),
                 truth=inventory['truth_sha256'], corners=CORNER_AUDIT_SHA256)
    cached = cache / (key + '.json')
    if cached.is_file():
        result = read(cached)
        if result['hashes'] == stamp:
            return result['case']
    grade = verify_grade((entry, build, inventory))
    upstream = read(receipt)
    for field in ('geometry', 'scores', 'targets', 'registration', 'translation', 'reference_sha256',
                  'camera_usable', 'camera_source', 'source_sha256', 'exported', 'pack_sha256'):
        check(grade[field], upstream[field], f'{key} independently graded {field}')
    reference = Path(entry['frame'])
    registered = camera_validation(entry, camera)
    ground = ground_validation(entry, camera)
    relative = native_corner_score(camera['P'], read(reference / 'camera.json'), read(reference / 'labels.json')['faces'])
    check(registered, upstream['camera_validation'], f'{key} export registered camera')
    for field, observed in (('registered', registered), ('ground', ground), ('relative', relative)):
        check(observed, source['corner_audit'][field], f'{key} independently recalculated {field}')
    grade.update(native_floor_camera_reason=camera['reason'], native_floor_evidence=evidence(camera),
                 native_floor_seed=source['selected']['arm'], native_floor_projection_sha256=data_hash(camera['P']),
                 native_floor_source_receipt_sha256=source['receipt_sha256'],
                 native_camera=dict(registered=registered, relative=relative, ground=ground))
    save(cached, dict(hashes=stamp, case=grade))
    return grade


def policy_outcome(entry, incumbent, inventory, decoded, source, measured, name):
    retained = dict(incumbent, decoded_sha256=decoded, reference_sha256=inventory['source_reference_sha256'],
                    native_floor_coverage='incumbent_supported')
    if incumbent['camera_usable']:
        return retained
    camera = source['boundary'] if name == 'default' else source['selected']['camera']
    arm = 'boundary' if name == 'default' else source['selected']['arm']
    if camera['P'] is not None:
        check(measured is not None, True, f'{entry["id"]} complete export')
        check(measured['native_floor_seed'], arm, f'{entry["id"]} graded seed')
        return dict(measured, native_floor_coverage='candidate_export', previous_pack_sha256=incumbent['pack_sha256'])
    return dict(retained, native_floor_coverage='confirmed_abstention', native_floor_seed=arm,
                native_floor_camera_reason=camera['reason'], native_floor_evidence=evidence(camera),
                native_floor_source_receipt_sha256=digest(source['paths'][arm]))


def gauge(rows):
    diagnostics = [row['native_camera'] for row in rows if 'native_camera' in row]
    if not diagnostics:
        return dict(frames=0)
    return dict(frames=len(diagnostics), vertices=sum(r['relative']['check_vertices'] for r in diagnostics),
                within_2px=sum(r['relative']['check_within_2px'] for r in diagnostics),
                relative_worst_p90_px=max(r['relative']['p90_check_px'] for r in diagnostics),
                registered_worst_p90_px=max(r['registered']['p90_reprojection_px'] for r in diagnostics),
                ground_max_median_metric_span_fraction_error=max(
                    r['ground']['ground_metric_span_relative_error'] for r in diagnostics))


def write_policy(name, cases, entries, before, output, inputs):
    datasets = summary(cases, list(entries.values()))
    for dataset, group in datasets.items():
        for split, node in ((None, group), *group['splits'].items()):
            subset = [row for row in cases if row['dataset'] == dataset and (split is None or row['split'] == split)]
            previous = before[dataset] if split is None else before[dataset]['splits'][split]
            node.update(coverage=dict(Counter(row['native_floor_coverage'] for row in subset)),
                        delta={'incumbent': deltas(node, previous)}, native_camera=gauge(subset))
    decisions = Counter(row['native_floor_coverage'] for row in cases)
    casefile = output / f'independent-{name}-cases.json.gz'
    save(casefile, cases)
    save(output / f'independent-{name}-summary.json',
         dict(policy=name, datasets=datasets, expected_attempts=70, candidate_exports=decisions['candidate_export'],
              attempted_no_export=decisions['confirmed_abstention'], pending_attempts=0,
              confirmed_abstention_ids=[r['id'] for r in cases if r['native_floor_coverage'] == 'confirmed_abstention'],
              retained_supported=decisions['incumbent_supported'], cases_sha256=digest(casefile),
              native_camera=gauge(cases), inputs=inputs))


def verify_production(args, controls, entries, incumbent, decoded, sources, scored):
    candidate = args.native_review.parents[2]
    production = read(args.native_review / 'production-camera.json')
    cli_path = args.native_review / 'production-cli.json'
    live_path = args.native_review / 'live-primary-cli.json'
    negatives_path = args.native_review / 'production-controls/negative-controls.json'
    cli = read(cli_path)
    check(production['files'], cli['files'], 'production camera and CLI code fingerprints')
    for name, expected in production['files'].items():
        check(digest(candidate / 'engine' / name), expected, f'production engine {name}')
    reviewed_cases = []
    for case in cli['cases']:
        key = case['id']
        old = incumbent[key]
        entry = entries[key]
        latest = scored.get(key)
        rebuilt = latest is not None and latest['native_floor_seed'] == 'boundary'
        expected = args.run / 'builds' / old['dataset'] / key if rebuilt else policy_pack(entry, old)
        actual = args.run / 'normal-cli' / key
        manifest = read(actual / 'build-manifest.json')
        placement = manifest['placement']
        check(placement['P'], read(expected / 'build-manifest.json')['placement']['P'], f'{key} CLI projection')
        check(placement['camera_source'], case['camera'], f'{key} CLI camera source')
        check(old['source_policy'], case['source_policy'], f'{key} CLI incumbent precedence')
        check(manifest['image_sha256'], entry['rgb_sha256'], f'{key} CLI RGB')
        check(case['rgb_sha256'], entry['rgb_sha256'], f'{key} CLI RGB receipt')
        expected_hash = latest['decoded_sha256'] if rebuilt else decoded[key]
        packed = decoded_pack(actual / 'reconstruction.zip')
        check(inventory_hash(packed), expected_hash, f'{key} CLI decoded cells')
        check(case['decoded_sha256'], expected_hash, f'{key} CLI decoded receipt')
        check(len(packed), case['placed'], f'{key} CLI placement count')
        check(digest(expected / 'reconstruction.zip'), case['expected_pack_sha256'], f'{key} CLI expected pack')
        check(digest(actual / 'reconstruction.zip'), case['cli_pack_sha256'], f'{key} CLI pack')
        if key in sources:
            camera = sources[key]['boundary']
            check(placement['P'], camera['P'], f'{key} CLI boundary-only decision')
        reviewed_cases.append(dict(id=key, policy=case['source_policy'], camera=case['camera'],
                                   decoded_sha256=expected_hash))
    live = read(live_path)
    for case in live['cases']:
        key = case['id']
        use_old = incumbent[key]['camera_usable']
        actual = args.run / ('normal-cli' if use_old else 'live-primary') / key
        packed = decoded_pack(actual / 'reconstruction.zip')
        expected_hash = decoded[key] if use_old else scored[key]['decoded_sha256']
        check(case['verified_decoded_sha256'], expected_hash, f'{key} live CLI decoded receipt')
        check(inventory_hash(packed), expected_hash, f'{key} live CLI decoded cells')
        check(digest(actual / 'reconstruction.zip'), case['cli_pack_sha256'], f'{key} live CLI pack')
        check(case['source_sha256'], entries[key]['rgb_sha256'], f'{key} live CLI source')
        check(case['exact_matrix'], True, f'{key} live CLI projection receipt')
        manifest = read(actual / 'build-manifest.json')
        check(manifest['image_sha256'], entries[key]['rgb_sha256'], f'{key} live CLI manifest source')
        if use_old:
            camera = fixed.source_inputs(entries[key])[3]['P']
        else:
            camera = sources[key]['boundary']['P']
        check(manifest['placement']['P'], camera, f'{key} live CLI projection')
    negative_names = {'constant', 'noise', 'one_axis_stripes', 'vertical_wall', 'split_phase',
                      'near_far_phase', 'wrong_material'}
    negatives = read(negatives_path)
    check({r['fixture'] for r in negatives}, negative_names | {'positive'}, 'source control fixture IDs')
    for case in negatives:
        expected = case['fixture'] == 'positive'
        check(case['accepted'], expected, f'{case["fixture"]} production source control')
        check(case['expected_acceptance'], expected, f'{case["fixture"]} frozen source control')
    report = dict(production_legacy_and_native_cli=reviewed_cases, live_primary=live['cases'],
                  negative_controls=[{k: case[k] for k in ('fixture', 'accepted', 'reason')} for case in negatives],
                  engine_hashes=production['files'], input_hashes={p.name: digest(p) for p in
                                                                  (cli_path, live_path, negatives_path)})
    save(args.output / 'independent-production-check.json', report)


def verify_silhouette(entries, incumbent, inventory, controls, report, output):
    manifest_path = report / 'manifest.json'
    manifest = read(manifest_path)
    check(manifest['freeze'], '493350c', 'silhouette research freeze')
    check(manifest['frames'], digest(controls / 'frames.json'), 'silhouette paired frames')
    check(manifest['controls'], digest(controls / 'independent-matched-cases.json.gz'), 'silhouette incumbent cohort')
    check(len(manifest['incumbent_pack_hashes']), len(entries), 'silhouette carry-forward pack count')
    check(dict(manifest['incumbent_pack_hashes']), {key: row['pack_sha256'] for key, row in incumbent.items()},
          'silhouette carried pack hashes')
    records = indexes(report / 'cases.json.gz', manifest['records'])
    check(records.keys(), {key for key, row in incumbent.items() if not row['camera_usable']},
          'silhouette residual frame IDs')
    forced = Counter()
    oracle = dict(incumbent)
    for key, record in records.items():
        check(record['source']['candidate'], None, f'{key} silhouette source abstention')
        check(record['split'], entries[key]['split'], f'{key} silhouette source split')
        hashes = record['source']['hashes']
        check(hashes[str(Path(key) / 'rgb.png')], inventory[key]['rgb_sha256'], f'{key} silhouette source RGB')
        for attempt in record['forced']:
            forced[attempt['method']] += 1
            check(attempt['case']['source_sha256'], inventory[key]['rgb_sha256'], f'{key} forced trial RGB')
            check(attempt['case']['reference_sha256'], inventory[key]['source_reference_sha256'],
                  f'{key} forced trial reference')
        options = [incumbent[key], *(attempt['case'] for attempt in record['forced']
                                      if attempt['method'] == 'source-ground')]
        oracle[key] = max(options, key=lambda r: r['geometry']['strict']['f1'])
        check(oracle[key]['pack_sha256'], record['best_ground_pack'], f'{key} forced ground chooser pack')
    published = read(report / 'summary.json')
    check(dict(forced), published['export_count'], 'silhouette reference-only diagnostic export totals')
    composed = summary(list(oracle.values()), list(entries.values()))['visible-builds-v3']
    forced_chooser = {f'{metric}_{scope}': composed[metric][scope] for metric in ('geometry', 'scores')
                      for scope in ('strict', 'facade')}
    check(forced_chooser, published['forced_source_ground_chooser_v3'], 'silhouette forced ground chooser grade')
    save(output / 'independent-silhouette-check.json',
         dict(source_freeze=manifest['freeze'], checked_frames=len(entries), source_attempts=len(records),
              source_recoveries=0, retained_pack_matches=len(entries), diagnostic_forced_exports=dict(forced),
              reference_only_ground_chooser=forced_chooser,
              manifest_sha256=digest(manifest_path), cases_sha256=manifest['records']))


def grade(args):
    controls = args.controls
    frozen = read(controls / 'manifest.json')
    for name, expected in frozen['hashes'].items():
        check(digest(controls / name), expected, f'frozen {name}')
    previous = read(controls / 'independent-matched-summary.json')
    check(digest(controls / 'manifest.json'), previous['control_manifest_sha256'], 'incumbent frozen controls')
    incumbent = indexes(controls / 'independent-matched-cases.json.gz', previous['cases_sha256'])
    inventory = indexes(controls / 'inventory.json.gz', frozen['hashes']['inventory.json.gz'])
    entries = {entry['id']: entry for entry in frozen_frames()}
    frame_rows = indexes(controls / 'frames.json', frozen['hashes']['frames.json'])
    check(entries.keys(), incumbent.keys(), 'incumbent frame IDs')
    check(entries.keys(), inventory.keys(), 'reference inventory frame IDs')
    check(entries.keys(), frame_rows.keys(), 'frozen frame IDs')
    for key, entry in entries.items():
        for field, value in frame_rows[key].items():
            check(entry.get(field), value, f'{key} frozen {field}')
        for field in ('dataset', 'split'):
            check(incumbent[key][field], entry[field], f'{key} incumbent {field}')
        check(incumbent[key]['source_sha256'], entry['rgb_sha256'], f'{key} incumbent RGB')
    residual = {key for key, row in incumbent.items() if not row['camera_usable']}
    check(len(residual), 70, 'incumbent residual count')
    check(sum(incumbent[key]['exported'] for key in residual), 0, 'incumbent residual exports')
    native_path = args.native_review / 'native-cameras.json.gz'
    production_path = args.native_review / 'production-camera.json'
    native = {row['id']: row for row in read(native_path)}
    audit = read(CORNER_AUDIT)
    check(digest(CORNER_AUDIT), CORNER_AUDIT_SHA256, 'independent native corner report')
    diagnostics = {row['id']: row for row in audit['cases']}
    production = {row['id']: row for row in read(production_path)['cases']}
    for name, rows in (('research', native), ('corner', diagnostics), ('production', production)):
        check(rows.keys(), residual, f'{name} attempts')
    sources = {key: source_receipts(entries[key], args.run, native[key], diagnostics[key], production[key])
               for key in sorted(residual)}
    accepted = {key for key, source in sources.items() if source['selected']['camera']['P'] is not None}
    model_free = {key for key, source in sources.items() if source['boundary']['P'] is not None}
    check(len(accepted), 65, 'boundary-first composed acceptances')
    check(len(model_free), 63, 'model-free acceptances')
    check(Counter(entries[key]['split'] for key in accepted), Counter(train=48, validation=5, test=12),
          'composed accepted split counts')
    check(Counter(entries[key]['split'] for key in model_free), Counter(train=47, validation=5, test=11),
          'model-free accepted split counts')
    complete = {path.stem for path in (args.run / 'scores').glob('*.json')}
    check(complete - accepted, set(), 'export receipts for rejected or incumbent cases')
    native_corner_score = load_corner_scorer()
    with ThreadPoolExecutor(args.workers) as pool:
        def verify(key):
            return key, verify_export(entries[key], inventory[key], sources[key], args.run, args.cache, native_corner_score)

        measured = dict(pool.map(verify, sorted(complete)))
    print(f'{len(measured)}/{len(accepted)} new packs independently decoded and graded. '
          f'{len(accepted - complete)} pending terminal export receipts.', flush=True)
    decoded = verify_controls(incumbent, entries, inventory, args.workers)
    verify_silhouette(entries, incumbent, inventory, controls, SILHOUETTE, args.output)
    verify_production(args, controls, entries, incumbent, decoded, sources, measured)
    print(f'{len(decoded)} frozen sources, visible ID references, and incumbent exports verified.', flush=True)
    inputs = dict(frozen_frames=frozen['hashes']['frames.json'], incumbent_cases=previous['cases_sha256'],
                  incumbent_summary=digest(controls / 'independent-matched-summary.json'),
                  inventory=frozen['hashes']['inventory.json.gz'], native_sources=digest(native_path),
                  production_cameras=digest(production_path), corner_audit=CORNER_AUDIT_SHA256)
    for name, expected in (('default', model_free), ('composed', accepted)):
        if expected - complete:
            continue
        rows = [policy_outcome(entries[key], row, inventory[key], decoded[key], sources.get(key), measured.get(key), name)
                for key, row in incumbent.items()]
        write_policy(name, rows, entries, previous['datasets'], args.output, inputs)
    if accepted - complete and not args.allow_pending:
        raise RuntimeError(f'Missing {len(accepted - complete)} completed build receipts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, default=CONTROLS)
    parser.add_argument('--run', type=Path, default=NATIVE_RUN)
    parser.add_argument('--native-review', type=Path, default=NATIVE_REVIEW)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--allow-pending', action='store_true')
    grade(parser.parse_args())


if __name__ == '__main__':
    main()
