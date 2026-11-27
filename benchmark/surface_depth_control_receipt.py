"""Recheck the saved ordinary datapacks used by the surface-depth pilot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from benchmark import expanded_fixed_handoffs as fixed
from benchmark import expanded_scores as scores
from benchmark import typed_structures as typed
from benchmark.camera_model_controls import (
    DEFAULT_BUILDS, REPEAT_BUILDS, check, data_hash, fixed_archive_score, frozen_frames,
    indexes, inventory_hash, summary,
)
from benchmark.camera_model_entry_check import MATCHED
from benchmark.camera_model_grade import verify_grade
from benchmark.camera_native_recovery_grade import CONTROLS, NATIVE_RUN, OUTPUT, frozen_case, policy_pack
from benchmark.expanded_baseline import MEASURED_REVISION, ROOT, STABLE, digest, read, save
from benchmark.inset_default_verification import LOCAL, local_release
from benchmark.mask_blocks import decoded_pack


BASE_REVISION = 'e3bd220'
REVIEW = ROOT / 'benchmark/reviews/surface-depth-round-controls-01'
PROMOTION = ROOT / 'benchmark/reviews/camera-native-round-01/validation.json'
PILOT = (
    'train-amphitheater-02-ashlar-high-right',
    'train-viaduct-02-riverstone-high-right',
    'train-shrine-01-ochre-low-left',
    'train-observatory-02-seafoam-low-left',
    'validation-harbor-02-kiln-high-right',
    'validation-harbor-02-seafoam-high-right',
    'validation-harbor-03-seafoam-high-right',
    'test-castle-01-ashlar-high-right',
    'test-castle-02-ashlar-rear-oblique',
    'heldout-facade-oblique-left',
    'inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right',
    'train-market-court-1-cool-high-right',
    'test-walled-garden-1-warm-high-right',
)
FIELDS = ('normal.npz', 'depth.npz', 'mlsd.json.gz', 'recovery.json.gz', 'recovery-masks.npz',
          'planarity.json.gz', 'planarity.npz')
SCORE_FIELDS = ('geometry', 'scores', 'targets', 'registration', 'translation', 'exported',
                'camera_usable', 'camera_source', 'camera_reason',
                'source_sha256', 'reference_sha256', 'pack_sha256', 'decoded_sha256')


def location(path):
    return path.relative_to(STABLE.parent).as_posix()


def file_receipt(path, expected=None):
    actual = digest(path)
    if expected is not None:
        check(actual, expected, str(path))
    return dict(path=location(path), sha256=actual)


def archive_receipt(archive, member, member_hash):
    return dict(file_receipt(archive), member=member, member_sha256=member_hash)


def metrics(rows, available):
    pooled = scores.pooled(rows, available)
    pooled['camera']['supported_empty'] = sum(row['camera_usable'] and not row['exported'] for row in rows)
    return pooled


def fixed_geometry_pack(key, split, job):
    rows, packs = typed.releases()
    row = rows[key]
    expected = row.get('pack_sha256', row['previous_pack_sha256'])
    if key in packs:
        member = f'{key}/reconstruction.zip'
        for directory in reversed(typed.GEOMETRY_RELEASES):
            release = read(directory / 'summary.json')
            if any(record['id'] == key for record in release['changed']):
                path = directory / 'default-exports.zip'
                check(sha256(packs[key]).hexdigest(), expected, f'{key} fixed geometry release')
                return archive_receipt(path, member, expected), set(decoded_pack(BytesIO(packs[key])))
        raise ValueError(f'No archive records the released geometry for {key}')
    rows, archive = local_release()
    if rows[key]['methods']:
        path = LOCAL / 'exports.zip'
        member = f'clear_front/{split}/{key}/reconstruction.zip'
        payload = archive.read(member)
        check(sha256(payload).hexdigest(), expected, f'{key} fixed geometry release')
        return archive_receipt(path, member, expected), set(decoded_pack(BytesIO(payload)))
    path = fixed.suite.CONTROL / split / key / 'reconstruction.zip'
    check(job[3]['pack_sha256'], expected, f'{key} inherited fixed geometry')
    return file_receipt(path, expected), set(decoded_pack(path))


def fixed_case(entry, row, inventory, archive):
    key, rgb = entry['id'], entry['rgb']
    member = f'{key}/reconstruction.zip'
    with ZipFile(archive) as zipped:
        check(frozen_case(entry, inventory, row, zipped), row['decoded_sha256'], f'{key} decoded receipt')
        payload = zipped.read(member)
    _, reference, transform, tied = fixed_archive_score(entry, payload, row)
    check(transform, inventory['ordinary']['registration'], f'{key} reference registration')
    job = fixed._jobs()[key]
    split, frame, prior, control = job
    _, parts, counts, source_camera = fixed.source_inputs(entry)
    framed = fixed.suite.framed
    if split in framed.NATIVE_INPUTS:
        folder = framed.NATIVE_INPUTS[split] / 'upstream' / key
        upstream = {name: file_receipt(folder / filename, prior[name])
                    for name, filename in (('surface_sha256', 'surfaces.json'), ('counts_sha256', 'counts.json'))}
    else:
        upstream = {'input_sha256': file_receipt(framed.DEPTH / 'inputs' / f'{key}.json', prior['input_sha256'])}
    check({name: item['sha256'] for name, item in upstream.items()}, control['input_hashes'],
          f'{key} fixed archived input lineage')
    refpaths = {'camera.json': rgb.with_suffix('.json'),
                'world-readback.json': rgb.with_name(f'{rgb.stem}-blocks.json')}
    check(refpaths.keys(), reference.hashes.keys(), f'{key} fixed reference file paths')
    geometry_pack, geometry_cells = fixed_geometry_pack(key, split, job)
    check(set(decoded_pack(BytesIO(payload))), geometry_cells, f'{key} typed control position lineage')
    placed = control['status']
    check(source_camera['P'], placed['P'], f'{key} fixed source camera through placement')
    placement_records = fixed.suite.CONTROL / 'cases.json'
    records_sha = read(fixed.suite.CONTROL / 'summary.json')['cases_sha256']
    return dict(
        pack=archive_receipt(archive, member, row['pack_sha256']),
        destination=dict(geometry_pack=geometry_pack, historical_placement=placed,
                         historical_placement_records=file_receipt(placement_records, records_sha),
                         placement_record_sha256=data_hash(control)),
        reference={name: file_receipt(path, reference.hashes[name]) for name, path in refpaths.items()},
        replay=dict(method='expanded_fixed_handoffs.source_inputs',
                    artifacts=upstream, parts_sha256=data_hash(parts), counts_sha256=data_hash(counts),
                    source_camera=source_camera, source_camera_sha256=data_hash(source_camera['P'])),
        geometry_optimum_has_another_tie_break=tied,
        layout=frame.get('layout', entry['layout']),
    )


def expanded_camera(key, row, camera):
    if row['native_floor_coverage'] == 'candidate_export':
        path = NATIVE_RUN / 'boundary' / 'metric' / f'{key}.json'
        digest_expected = row['native_floor_source_receipt_sha256']
        check(data_hash(camera['P']), row['native_floor_projection_sha256'], f'{key} released native P')
    elif row['source_policy'] == 'horizon-repeat-floor':
        path = REPEAT_BUILDS / key / 'repeat.json'
        digest_expected = None
    elif row['source_policy'] == 'boundary-budget-repeat':
        path = MATCHED.parent / 'metric' / f'{key}.json'
        digest_expected = None
    else:
        return None
    payload = read(path)
    check(payload['rgb_sha256'], row['source_sha256'], f'{key} camera receipt RGB')
    check(payload['camera']['P'], camera['P'], f'{key} source projection through placement')
    check(payload['camera']['camera_source'], camera['camera_source'], f'{key} source camera policy')
    if row.get('fallback_camera_sha256'):
        check(data_hash(payload['camera']), row['fallback_camera_sha256'], f'{key} released fallback camera')
    return file_receipt(path, digest_expected)


def expanded_case(entry, row, inventory):
    key, dataset = entry['id'], entry['dataset']
    ordinary = DEFAULT_BUILDS / dataset / key
    if row['native_floor_coverage'] == 'candidate_export':
        build = NATIVE_RUN / 'builds' / dataset / key
    else:
        build = policy_pack(entry, row)
    if row['native_floor_coverage'] == 'candidate_export':
        check(read(NATIVE_RUN / 'scores' / f'{key}.json')['pack_sha256'], row['pack_sha256'],
              f'{key} native export receipt')
    else:
        check(frozen_case(entry, inventory, row, None), row['decoded_sha256'], f'{key} decoded receipt')
    grade = verify_grade((entry, build, inventory))
    for name in SCORE_FIELDS:
        check(grade[name], row.get(name), f'{key} rescored {name}')
    reference = scores.load_reference(entry)
    check(inventory_hash(reference.truth), inventory['truth_sha256'], f'{key} visible IDs')
    check(data_hash(sorted(reference.optional)), inventory['facade_exempt_sha256'], f'{key} facade exemptions')
    manifest = read(build / 'build-manifest.json')
    placement = manifest['placement']
    check(manifest['version'], '26.3', f'{key} build version')
    check(manifest['through'], 'placement', f'{key} build stages')
    camera = expanded_camera(key, row, placement)
    parent = read(ordinary / 'receipt.json')
    check(parent['revision'], MEASURED_REVISION, f'{key} ordinary field revision')
    check(parent['source_sha256'], row['source_sha256'], f'{key} field source RGB')
    check(parent['model_sha256'], manifest['model_sha256'], f'{key} producer models')
    check(parent['atlas_sha256'], placement['material_assignment']['atlas_sha256'], f'{key} material atlas')
    atlas_reason = grade['atlas_camera_reason']
    if row['camera_source'] in ('horizon-repeat-floor', 'boundary-budget-repeat'):
        parent_manifest = read(ordinary / 'build-manifest.json')
        atlas_reason = scores.camera_fields(parent_manifest['placement'])['atlas_camera_reason']
    check(atlas_reason, row['atlas_camera_reason'], f'{key} ordinary atlas fallback reason')
    handoffs = ordinary / 'handoffs'
    replay = dict(method='expanded_baseline.replay_source_fields', root=location(ordinary),
                  ordinary_receipt=file_receipt(ordinary / 'receipt.json'),
                  ordinary_manifest=file_receipt(ordinary / 'build-manifest.json'),
                  ordinary_pack=file_receipt(ordinary / 'reconstruction.zip', parent['pack_sha256']),
                  ordinary_pack_is_current_control=ordinary == build,
                  artifacts={name: file_receipt(handoffs / name) for name in FIELDS},
                  model_sha256=parent['model_sha256'], atlas_sha256=parent['atlas_sha256'])
    dest = dict(root=location(build), placement_camera_receipt=camera,
                P=placement['P'], projection_sha256=data_hash(placement['P']),
                artifacts={name: file_receipt(build / name)
                           for name in ('build-manifest.json', 'surfaces.json', 'counts.json', 'placements.json')})
    return dict(pack=file_receipt(build / 'reconstruction.zip', row['pack_sha256']), destination=dest,
                reference={name: file_receipt(Path(entry['frame']) / name, value)
                           for name, value in reference.hashes.items()},
                replay=replay, layout=entry['layout'])


def verify_case(entry, row, inventory, archive):
    key = entry['id']
    for name in ('dataset', 'split'):
        check(row[name], entry[name], f'{key} release {name}')
        check(inventory[name], entry[name], f'{key} inventory {name}')
    check(entry['rgb_sha256'], inventory['rgb_sha256'], f'{key} frozen inventory RGB')
    check(row['source_sha256'], entry['rgb_sha256'], f'{key} released RGB')
    check(row['reference_sha256'], inventory['source_reference_sha256'], f'{key} released reference')
    more = (fixed_case(entry, row, inventory, archive) if entry['dataset'] == scores.FIXED_DATASET else
            expanded_case(entry, row, inventory))
    return dict(id=key, dataset=entry['dataset'], split=entry['split'], family=entry['family'],
                palette=entry['palette'], dataset_manifest_sha256=entry['dataset_manifest_sha256'],
                rgb=file_receipt(entry['rgb'], entry['rgb_sha256']),
                visible_ids_sha256=inventory['truth_sha256'],
                facade_exempt_sha256=inventory['facade_exempt_sha256'],
                decoded_sha256=row['decoded_sha256'], native_floor_coverage=row['native_floor_coverage'],
                source_policy=row.get('source_policy', 'native-floor-atlas'),
                camera_source=row['camera_source'], camera_reason=row['camera_reason'],
                atlas_camera_reason=row['atlas_camera_reason'],
                baseline={field: row[field] for field in ('geometry', 'scores', 'registration', 'translation',
                          'camera_usable', 'exported', 'visible_truth', 'required_truth')}, **more)


def run(output, workers):
    cases_path = OUTPUT / 'independent-default-cases.json.gz'
    summary_path = OUTPUT / 'independent-default-summary.json'
    promotion = read(PROMOTION)
    for path in (cases_path, summary_path):
        check(digest(path), promotion['sha256'][path.relative_to(ROOT).as_posix()], f'{path.name} main promotion')
    release = read(summary_path)
    cases = indexes(cases_path, release['cases_sha256'])
    entries = {row['id']: row for row in frozen_frames()}
    frozen = read(CONTROLS / 'manifest.json')
    frames_path = CONTROLS / 'frames.json'
    frozen_frames_by_id = indexes(frames_path, release['inputs']['frozen_frames'])
    check(frozen['hashes'][frames_path.name], release['inputs']['frozen_frames'], 'published frames lineage')
    inventory_path = CONTROLS / 'inventory.json.gz'
    inventory = indexes(inventory_path, release['inputs']['inventory'])
    check(frozen['hashes'][inventory_path.name], release['inputs']['inventory'], 'published inventory lineage')
    check(entries.keys(), cases.keys(), 'full 288 frame IDs')
    check(entries.keys(), inventory.keys(), 'full 288 reference IDs')
    check(entries.keys(), frozen_frames_by_id.keys(), 'full 288 frozen frame IDs')
    for key, row in cases.items():
        for field, value in frozen_frames_by_id[key].items():
            check(value, entries[key].get(field), f'{key} full frozen {field}')
        for field in ('dataset', 'split'):
            check(row[field], entries[key][field], f'{key} full release {field}')
        check(row['source_sha256'], entries[key]['rgb_sha256'], f'{key} full release RGB')
    measured = summary(list(cases.values()), list(entries.values()))
    for dataset, published in release['datasets'].items():
        check(measured[dataset].keys(), published.keys() - {'coverage', 'delta', 'native_camera'},
              f'{dataset} summary fields')
        for field in measured[dataset].keys() - {'splits'}:
            check(measured[dataset][field], published[field], f'{dataset} {field}')
        for split, split_row in measured[dataset]['splits'].items():
            for field, value in split_row.items():
                check(value, published['splits'][split][field], f'{dataset} {split} {field}')
    archive = fixed._release()[2]

    def verify(key):
        return verify_case(entries[key], cases[key], inventory[key], archive)

    with ThreadPoolExecutor(workers) as pool:
        frames = list(pool.map(verify, PILOT))
    pilot = [cases[key] for key in PILOT]
    by_dataset = {}
    for dataset in measured:
        dataset_pilot = [row for row in pilot if row['dataset'] == dataset]
        groups = sorted({row['split'] for row in dataset_pilot})
        by_dataset[dataset] = dict(metrics(dataset_pilot, measured[dataset]['available']),
                                   splits={split: metrics([r for r in dataset_pilot if r['split'] == split],
                                                           measured[dataset]['splits'][split]['available'])
                                           for split in groups})
    receipt = dict(
        schema=1, minecraft_version='26.3', base_revision=BASE_REVISION,
        paths_relative_to=STABLE.parent.as_posix(),
        scoring=dict(visibility='single-image native visible targets; all extra cells charged',
                     placement='one whole-build integer XYZ shift selected by strict geometry',
                     facade='native 4x4 exemptions for thin horizontal-only cells',
                     typed='exact block ID at same geometry-aligned location'),
        release=dict(cases=file_receipt(cases_path), summary=file_receipt(summary_path),
                     inventory=file_receipt(inventory_path), frozen_frames=file_receipt(frames_path),
                     promoted_policy=release['policy']),
        pilot=dict(frame_ids=list(PILOT), verification='RGB, reference, ZIP, decoded inventory and rescored placement',
                   pooling=metrics(pilot, len(pilot)), datasets=by_dataset, frames=frames),
        carried_cohorts=dict(status='reaggregated 288 independently graded saved rows; no model rerun',
                             datasets=release['datasets']),
    )
    save(output, receipt)
    print(f'Verified {len(frames)} pilot packs and reaggregated {len(cases)} saved controls into {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REVIEW / 'controls.json')
    parser.add_argument('--workers', type=int, default=4)
    options = parser.parse_args()
    run(options.output, options.workers)


if __name__ == '__main__':
    main()
