"""Verify the whole-mask ruler comparison and freeze a later transfer extension."""

import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark import expanded_fixed_handoffs as fixed
from benchmark import expanded_scores as scores
from benchmark import typed_structures as typed
from benchmark.camera_metrics import registration
from benchmark.camera_model_controls import check, data_hash, fixed_reference, frozen_frames, indexes, inventory_hash
from benchmark.camera_native_recovery_grade import CONTROLS, OUTPUT
from benchmark.expanded_baseline import ROOT, STABLE, WORKTREES, digest, read, save
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.full_block_lattice import atlas
from benchmark.mask_blocks import decoded_pack


WORK = STABLE.parent
CONTROL_ROOT = WORKTREES / 'surface-depth-round-controls'
CONTROL_FILE = CONTROL_ROOT / 'benchmark/reviews/surface-depth-round-controls-01/controls.json'
RULER_ROOT = WORKTREES / 'local-atlas-wall-rulers'
RULER_REVIEW = RULER_ROOT / 'benchmark/reviews/local-wall-rulers-01'
RULER_RUNS = [RULER_ROOT / 'runs' / name for name in ('local-wall-rulers-native-support-01', 'local-wall-rulers-01')]
REVIEW = ROOT / 'benchmark/reviews/wall-ruler-transfer-controls-01'
ARM = 'native_single'
SALT = 'wall-ruler-transfer-extension-v1'
PUBLISHED = 'e3bd220'


@lru_cache(maxsize=None)
def file_digest(path):
    return digest(path)


def verified_payload(record):
    path = WORK / record['path']
    check(file_digest(path), record['sha256'], str(path))
    if 'member' in record:
        with ZipFile(path) as archive:
            payload = archive.read(record['member'])
        check(sha256(payload).hexdigest(), record['member_sha256'], record['member'])
        return payload
    return path.read_bytes()


def verify_file_records(value):
    if not isinstance(value, dict):
        return 0
    if 'path' in value and 'sha256' in value:
        if 'member' in value:
            verified_payload(value)
        else:
            check(file_digest(WORK / value['path']), value['sha256'], value['path'])
        return 1
    return sum(verify_file_records(child) for child in value.values())


def rescore(raw, claim, reference, transform, assignment=None):
    local = decoded_pack(BytesIO(raw))
    registered = typed.aligned_sites(list(local), transform, (0, 0, 0))
    _, maximum = best_translation_fast(registered, reference.truth)
    placed = typed.aligned_sites(list(local), transform, claim['translation'])
    predicted = dict(zip(placed, local.values(), strict=True))
    geometry = typed.geometry_score(predicted, reference.truth, reference.optional)
    check(geometry['strict']['correct'], maximum, 'geometry-only alignment optimum')
    names = ('strict', 'facade', 'material')
    assigned = {p for p, k in zip(placed, local, strict=True) if assignment and assignment[k] != 'fallback'}
    if assignment is not None:
        names += scores.ASSIGNMENT_FIELDS
    detailed = typed.score(predicted, reference.truth, reference.optional, assigned)
    check({k: detailed[k] for k in names}, {k: claim['scores'][k] for k in names}, 'decoded typed scores')
    check(geometry, claim['geometry'], 'decoded geometry')
    check(len(local), claim['exported'], 'decoded export count')
    return predicted, local


def candidate_report(key, signature, pack_sha256=None):
    for run in RULER_RUNS:
        for path in sorted((run / 'exports' / key).glob('*/case.json')):
            report = read(path)
            if report['signature'] == signature and (pack_sha256 is None or report['case']['pack_sha256'] == pack_sha256):
                return path, report
    raise ValueError(f'No saved export for {key} and signature {signature}')


def candidate_inventory(path, report, frame, entry, reference, transform):
    case = report['case']
    key = frame['id']
    check(case['source_sha256'], frame['rgb']['sha256'], f'{key} export source')
    check(case['registration']['transform'], transform.tolist(), f'{key} fixed registration')
    pack = path.with_name('reconstruction.zip')
    check(file_digest(pack), case['pack_sha256'], f'{key} saved export')
    assignment = None
    if entry['dataset'] != scores.FIXED_DATASET:
        manifest = read(path.with_name('build-manifest.json'))
        check(manifest['image_sha256'], frame['rgb']['sha256'], f'{key} build source')
        check(manifest['placement']['P'], frame['destination']['P'], f'{key} source projection')
        build_blocks = read(path.with_name('placements.json'))['blocks']
        assignment = {(b['x'], b['y'], b['z']): b['material_source'] for b in build_blocks}
        build_ids = {(b['x'], b['y'], b['z']): 'minecraft:' + b['block_id'].removeprefix('minecraft:')
                     for b in build_blocks}
        check(len(assignment), len(build_blocks), f'{key} unique placement positions')
    placed, local = rescore(pack.read_bytes(), case, reference, transform, assignment)
    if assignment is not None:
        check(local, build_ids, f'{key} placements versus decoded pack')
    return placed, local


def paired(before, after, truth):
    gained = after.keys() - before.keys()
    lost = before.keys() - after.keys()
    return dict(recovered=len(gained & truth.keys()), lost=len(lost & truth.keys()),
                extra_added=len(gained - truth.keys()), extra_removed=len(lost - truth.keys()),
                changed_positions=before.keys() != after.keys(), changed_ids_or_positions=before != after)


def audit_pilot(controls, entries):
    manifest = read(RULER_REVIEW / 'manifest.json')
    check(manifest['base_revision'], PUBLISHED, 'ruler production base')
    check(controls['base_revision'], PUBLISHED, 'controls production base')
    check(file_digest(CONTROL_FILE), manifest['controls_sha256'], 'ruler controls')
    for name, expected in manifest['artifacts'].items():
        check(file_digest(RULER_REVIEW / name), expected, name)
    for name, expected in manifest['implementation'].items():
        check(file_digest(RULER_ROOT / 'benchmark' / name), expected, name)
    check(atlas().provenance, manifest['atlas'], 'prepared Java atlas')
    exports = read(RULER_REVIEW / 'exports.json')
    check([r['id'] for r in exports['cases']], controls['pilot']['frame_ids'], 'pilot IDs and order')
    exports_by_id = {r['id']: r for r in exports['cases']}
    results = []
    for frame in controls['pilot']['frames']:
        key = frame['id']
        verified = verify_file_records(frame)
        entry = entries[key]
        check(entry['rgb_sha256'], frame['rgb']['sha256'], f'{key} released source')
        pack_hash = frame['pack'].get('member_sha256', frame['pack']['sha256'])
        check(manifest['provenance'][key], dict(model_sha256=frame['replay'].get('model_sha256'),
              source_sha256=frame['rgb']['sha256'], pack_sha256=pack_hash,
              visible_sha256=frame['visible_ids_sha256']), f'{key} ruler provenance')
        if entry['dataset'] == scores.FIXED_DATASET:
            reference, transform = fixed_reference(entry)
        else:
            reference = scores.load_reference(entry)
            transform = registration(np.asarray(frame['destination']['P']), reference.pose)
        check(inventory_hash(reference.truth), frame['visible_ids_sha256'], f'{key} visible IDs')
        check(data_hash(sorted(reference.optional)), frame['facade_exempt_sha256'], f'{key} facade mask')
        released_transform = frame['baseline']['registration'].get('transform')
        if released_transform is not None:
            check(transform.tolist(), released_transform, f'{key} registered camera')
        before, local = rescore(verified_payload(frame['pack']), frame['baseline'], reference, transform)
        check(inventory_hash(local), frame['decoded_sha256'], f'{key} ordinary decoded IDs')
        corrected_path = RULER_RUNS[0] / f'{key}.json'
        corrected = read(corrected_path)
        check(corrected['source_sha256'], frame['rgb']['sha256'], f'{key} ruler source')
        check(corrected['pack_sha256'], frame['pack']['sha256'], f'{key} ruler parent receipt')
        signature = [[r['surface'], r['axis'], r['arms'][ARM]['level']] for r in corrected['records']
                     if r['arms'][ARM]['accepted'] and r['arms'][ARM]['legal_plane']]
        published = exports_by_id[key]['arms'][ARM]
        check(signature, published['signature'], f'{key} corrected ordered carrier signature')
        export_receipt = None
        after = before
        claim = frame['baseline']
        if signature:
            path, report = candidate_report(key, signature, published['pack_sha256'])
            after, _ = candidate_inventory(path, report, frame, entry, reference, transform)
            export_receipt = dict(path=path.relative_to(WORK).as_posix(), sha256=file_digest(path))
            claim = report['case']
        else:
            check(published['pack_sha256'], pack_hash, f'{key} inherited pack')
        changes = paired(before, after, reference.truth)
        check(changes, {k: published[k] for k in changes}, f'{key} paired decoded changes')
        for metric in ('geometry', 'scores', 'exported'):
            check(claim[metric], published[metric], f'{key} published {metric}')
        if exports_by_id[key].get('noop_decoded_equal'):
            path, report = candidate_report(key, [])
            noop, noop_local = candidate_inventory(path, report, frame, entry, reference, transform)
            check(noop_local, local, f'{key} no-op decoded IDs')
            check(noop, before, f'{key} no-op aligned IDs')
        results.append(dict(id=key, cohort=frame['dataset'], split=frame['split'], source=frame['rgb']['sha256'],
                            visible=frame['visible_ids_sha256'], ordinary_pack=pack_hash,
                            experimental_pack=published['pack_sha256'], signature=signature,
                            corrected_receipt_sha256=file_digest(corrected_path), export_receipt=export_receipt,
                            noop=bool(exports_by_id[key].get('noop_decoded_equal')), verified_file_records=verified,
                            baseline={k: frame['baseline'][k] for k in ('geometry', 'scores', 'exported')},
                            candidate={k: claim[k] for k in ('geometry', 'scores', 'exported')}, paired=changes))
    pools = {}
    for group, published in exports['summary'].items():
        subset = [r for r in results if group == 'all' or group == r['cohort'] or
                  group == ('development' if r['split'] == 'train' else 'transfer')]
        pools[group] = dict(tested=len(subset))
        for name, key in (('baseline', 'baseline'), ('candidate', ARM)):
            pool = {metric: typed.pooled_scores([r[name][metric] for r in subset]) for metric in ('geometry', 'scores')}
            for metric, value in pool.items():
                check(value, published[key][metric], f'{group} pooled {key} {metric}')
            pools[group][name] = pool
    return dict(base_revision=PUBLISHED, control_sha256=file_digest(CONTROL_FILE), atlas=manifest['atlas'],
                ruler_export_manifest_sha256=file_digest(RULER_REVIEW / 'manifest.json'), frames=results, pools=pools)


def select_extension(entries, rows, pilot):
    def by_hash(items, key):
        return min(items, key=lambda item: sha256(f'{SALT}:{key(item)}'.encode()).digest())

    def choose(group):
        return by_hash(group, lambda e: e['rgb_sha256'])

    def layout(name):
        return [e for e in entries.values() if e['layout'] == name and e['split'] != 'train']

    used = {f[field] for f in pilot for field in ('family', 'layout')}
    selected = []
    for split in ('test', 'regression'):
        candidates = [e for e in entries.values() if e['dataset'] == scores.FIXED_DATASET and e['split'] == split
                      and e['layout'] not in used and e['family'] not in used]
        name = by_hash({e['layout'] for e in candidates}, str)
        item = choose([e for e in candidates if e['layout'] == name])
        selected.append((item, f'fixed historical {split} family outside pilot'))
        used.update((item['layout'], item['family']))
    harbor_angles = defaultdict(list)
    for item in layout('harbor-01'):
        angle = item['id'].split(f'-{item["palette"]}-', 1)[1]
        harbor_angles[angle].append(item)
    angle = by_hash([a for a, group in harbor_angles.items() if len({e['palette'] for e in group}) == 2], str)
    for item in sorted(harbor_angles[angle], key=lambda e: e['id']):
        selected.append((item, 'Harbor 01 both palettes at the same hash-selected angle'))
    for supported in (True, False):
        item = choose([e for e in layout('castle-03') if rows[e['id']]['camera_usable'] == supported])
        selected.append((item, 'Castle 03 camera-supported stratum' if supported else 'Castle 03 camera-abstained stratum'))
    chapel = by_hash(['hillside-chapel-1', 'hillside-chapel-2'], str)
    selected.append((choose(layout(chapel)), 'new decorative pilot family'))
    selected.append((choose(layout('walled-garden-2')), 'new Walled Garden layout'))
    return selected


def audit_extension(controls, entries):
    path = CONTROL_ROOT / 'benchmark/surface_depth_control_receipt.py'
    spec = spec_from_file_location('surface_depth_controls', path)
    helper = module_from_spec(spec)
    spec.loader.exec_module(helper)
    cases_path = OUTPUT / 'independent-default-cases.json.gz'
    rows = indexes(cases_path, controls['release']['cases']['sha256'])
    inventory = indexes(CONTROLS / 'inventory.json.gz', controls['release']['inventory']['sha256'])
    archive = fixed._release()[2]
    selection = select_extension(entries, rows, controls['pilot']['frames'])
    frames = []
    for entry, stratum in selection:
        key = entry['id']
        if entry['dataset'] == scores.FIXED_DATASET:
            frame = fixed_extension(entry, rows[key], inventory[key], archive)
        else:
            frame = helper.verify_case(entry, rows[key], inventory[key], archive)
        frames.append(dict(stratum=stratum, **frame))
    return dict(base_revision=PUBLISHED, selection_salt=SALT,
                selection_rule='building and frame SHA-256 ordering within declared strata; no score filtering',
                reviewed_family_note='Harbor and Castle families occurred in the pilot; fixed views are historical regression',
                measured='ordinary decoded controls only; every extension candidate remains unmeasured',
                fixture_counts={'fixed-104': 104, **scores.COHORTS}, frames=frames)


def fixed_extension(entry, row, inventory, archive):
    key = entry['id']
    split, _, _, control = fixed._jobs()[key]
    rgb = entry['rgb']
    reference, transform = fixed_reference(entry)
    paths = {name: rgb.with_name(name) for name in reference.hashes}
    path_receipt = lambda path: dict(path=path.relative_to(WORK).as_posix(), sha256=file_digest(path))
    references = {name: path_receipt(path) for name, path in paths.items()}
    check({name: r['sha256'] for name, r in references.items()}, reference.hashes, f'{key} fixed native files')
    check(inventory_hash(reference.truth), inventory['truth_sha256'], f'{key} fixed visible IDs')
    check(data_hash(sorted(reference.optional)), inventory['facade_exempt_sha256'], f'{key} fixed optional')
    member = f'{key}/reconstruction.zip'
    pack = dict(path_receipt(archive), member=member, member_sha256=row['pack_sha256'])
    _, local = rescore(verified_payload(pack), row, reference, transform)
    check(inventory_hash(local), row['decoded_sha256'], f'{key} fixed decoded IDs')
    _, parts, counts, camera = fixed.source_inputs(entry)
    framed = fixed.suite.framed
    if split in framed.NATIVE_INPUTS:
        upstream_root = framed.NATIVE_INPUTS[split] / 'upstream' / key
        upstream = {name: path_receipt(upstream_root / filename) for name, filename in
                    (('surface_sha256', 'surfaces.json'), ('counts_sha256', 'counts.json'))}
    else:
        upstream = {'input_sha256': path_receipt(framed.DEPTH / 'inputs' / f'{key}.json')}
    check({name: record['sha256'] for name, record in upstream.items()}, control['input_hashes'], f'{key} fixed handoffs')
    check(camera['P'], control['status']['P'], f'{key} fixed camera projection')
    rgb_receipt = path_receipt(rgb)
    check(rgb_receipt['sha256'], entry['rgb_sha256'], f'{key} fixed source')
    return dict(id=key, dataset=entry['dataset'], split=split, family=entry['family'], layout=entry['layout'],
                palette=entry['palette'], rgb=rgb_receipt, dataset_manifest_sha256=entry['dataset_manifest_sha256'],
                visible_ids_sha256=inventory['truth_sha256'], facade_exempt_sha256=inventory['facade_exempt_sha256'],
                decoded_sha256=row['decoded_sha256'], camera_source=row['camera_source'], camera_reason=row['camera_reason'],
                baseline={k: row[k] for k in ('geometry', 'scores', 'registration', 'translation', 'camera_usable',
                                             'exported', 'visible_truth', 'required_truth')},
                pack=pack, reference=references, destination=dict(historical_placement=control['status']),
                replay=dict(method='expanded_fixed_handoffs.source_inputs', artifacts=upstream,
                            parts_sha256=data_hash(parts), counts_sha256=data_hash(counts),
                            source_camera=camera, source_camera_sha256=data_hash(camera['P'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REVIEW)
    parser.add_argument('--only', choices=['pilot', 'extension'])
    args = parser.parse_args()
    controls = read(CONTROL_FILE)
    entries = {e['id']: e for e in frozen_frames()}
    if args.only in (None, 'pilot'):
        save(args.out / 'whole-mask.json', audit_pilot(controls, entries))
        print(f'Verified {len(controls["pilot"]["frames"])} paired native-single ruler controls')
    if args.only in (None, 'extension'):
        extension = audit_extension(controls, entries)
        save(args.out / 'extension.json', extension)
        print(f'Froze and verified {len(extension["frames"])} extension controls')


if __name__ == '__main__':
    main()
