"""Decode and score Java 26.3 builds on the frozen expanded visible cohorts."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark.camera_metrics import registration
from benchmark.expanded_baseline import COHORTS, MEASURED_REVISION, OUTPUT, ROOT, SOURCE, WORKTREES, digest, frames, read, save
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.typed_structures import SUITE, aligned_sites, geometry_score, overlap, pooled_scores, score
from calibration.world import GROUND


RELEASE = ROOT / 'benchmark/reviews'
FIXED_DATASET = 'fixed-104'
ASSIGNMENT_FIELDS = ('assigned', 'fallback', 'assigned_correct', 'assigned_positions')


@dataclass
class FrameReference:
    truth: dict
    optional: set
    pose: dict
    by_kind: dict
    hashes: dict


def load_reference(entry):
    frame = Path(entry['frame'])
    metadata = read(frame / 'frame.json')
    names = ('camera.json', 'labels.json', 'labels.npz')
    hashes = {name: digest(frame / name) for name in names}
    if (metadata['id'] != entry['id'] or metadata['files']['rgb.png'] != entry['rgb_sha256']
            or any(metadata['files'][name] != value for name, value in hashes.items())
            or not metadata['framing']['accepted'] or metadata['ground_y'] != GROUND):
        raise ValueError(f'Frozen frame provenance mismatch: {frame}')
    labels = read(frame / 'labels.json')
    with np.load(frame / 'labels.npz', allow_pickle=False) as regions:
        optional = horizontal_slivers(labels, regions['face_id'])
    truth = {}
    by_kind = defaultdict(set)
    for block in labels['blocks']:
        location = tuple(block['xyz_world'])
        truth[location] = block['block_id']
        by_kind[block.get('geometry_kind', 'full_cube')].add(location)
    if len(truth) != len(labels['blocks']) or len(truth) != metadata['counts']['visible_blocks']:
        raise ValueError(f'Visible label count or uniqueness mismatch: {frame}')
    return FrameReference(truth, optional, read(frame / 'camera.json'), dict(by_kind), hashes)


def target_breakdown(predictions, reference):
    def metric(positions):
        matched = positions & predictions.keys()
        return dict(target=len(positions), correct=len(matched), missed=len(positions - matched),
                    exact_id=sum(predictions[p] == reference.truth[p] for p in matched),
                    facade_target=len(positions - reference.optional),
                    facade_correct=len(matched - reference.optional),
                    facade_exact_id=sum(predictions[p] == reference.truth[p] for p in matched - reference.optional))

    details = {kind: metric(positions) for kind, positions in sorted(reference.by_kind.items())}
    partial = set().union(*(cells for kind, cells in reference.by_kind.items() if kind != 'full_cube'))
    if partial:
        details['partial_blocks'] = metric(partial)
    return details


def camera_fields(camera):
    scale = camera.get('camera_status', camera.get('scale_status'))
    projection = camera.get('P')
    return dict(camera_status=scale, camera_usable=scale == 'supported' and projection is not None,
                camera_source=camera.get('camera_source'), camera_reason=camera.get('reason'),
                atlas_camera_reason=camera.get('atlas_camera_reason', camera.get('atlas_reason')),
                repeat_camera_reason=camera.get('repeat_camera_reason', camera.get('repeat_reason')))


def evaluate_pack(entry, pack, camera, *, assignment=None, reference=None):
    """Score a real export from this source. Return the row and aligned world IDs.

    `pack` is a path or raw ZIP bytes. `camera` is the placement status including
    its P matrix. `assignment` maps local coordinates to material source names.
    The result is relative to one whole-building XYZ translation chosen by geometry.
    """
    payload = pack.read_bytes() if isinstance(pack, Path) else pack
    local = decoded_pack(BytesIO(payload))
    if assignment is not None and assignment.keys() != local.keys():
        raise ValueError(f'Material evidence coordinates differ from pack: {entry["id"]}')
    reference = reference if reference is not None else load_reference(entry)
    truth, optional = reference.truth, reference.optional
    projection = camera.get('P')
    camera_info = camera_fields(camera)
    if camera_info['camera_usable']:
        transform = registration(np.asarray(projection), reference.pose)
        if not np.isfinite(transform).all():
            raise ValueError(f'Nonfinite camera registration: {entry["id"]}')
        registered = dict(zip(local, aligned_sites(list(local), transform, (0, 0, 0)), strict=True))
        offset, matched = best_translation_fast(registered.values(), truth)
        aligned = {p: tuple(a + delta for a, delta in zip(world, offset, strict=True))
                   for p, world in registered.items()}
    elif local:
        raise ValueError(f'Export has {len(local)} blocks without a camera frame: {entry["id"]}')
    else:
        transform, registered, offset, aligned, matched = None, {}, (0, 0, 0), {}, 0

    def grade(mapping):
        names = {world: local[p] for p, world in mapping.items()}
        assigned = {mapping[p] for p, method in (assignment or {}).items() if method != 'fallback'}
        return geometry_score(names, truth, optional), score(names, truth, optional, assigned), names

    metrics, scores, predictions = grade(aligned)
    registered_geometry, registered_scores, _ = grade(registered)
    if matched != metrics['strict']['correct']:
        raise ValueError(f'Whole-build translation does not match the exact optimum: {entry["id"]}')
    if assignment is None:
        for row in (scores, registered_scores):
            row.update(dict.fromkeys(ASSIGNMENT_FIELDS))
    case = {name: entry[name] for name in ('id', 'dataset', 'split', 'layout', 'family', 'palette')}
    return dict(case, source_sha256=entry['rgb_sha256'], dataset_manifest_sha256=entry['dataset_manifest_sha256'],
                reference_sha256=reference.hashes, pack_sha256=sha256(payload).hexdigest(),
                geometry=metrics, scores=scores, targets=target_breakdown(predictions, reference),
                registration=dict(transform=transform.tolist() if transform is not None else None,
                                  geometry=registered_geometry, scores=registered_scores),
                translation=list(offset), exported=len(local), visible_truth=len(truth),
                required_truth=len(truth) - len(optional), **camera_info), predictions


def score_build(entry, folder):
    build = read(folder / 'build-manifest.json')
    pack = folder / 'reconstruction.zip'
    if digest(Path(entry['frame']) / 'rgb.png') != entry['rgb_sha256'] or build['image_sha256'] != entry['rgb_sha256']:
        raise ValueError(f'Source image hash differs: {entry["id"]}')
    placements = read(folder / 'placements.json')['blocks']
    assignment = {(b['x'], b['y'], b['z']): b['material_source'] for b in placements}
    local = {(b['x'], b['y'], b['z']): 'minecraft:' + b['block_id'].removeprefix('minecraft:') for b in placements}
    if len(local) != len(placements) or local != decoded_pack(pack):
        raise ValueError(f'Ordinary placement inventory differs from decoded pack: {entry["id"]}')
    row, _ = evaluate_pack(entry, pack, build['placement'], assignment=assignment)
    if row['exported'] != build['placement']['placed_blocks']:
        raise ValueError(f'Pack count differs from build manifest: {entry["id"]}')
    row['provenance'] = 'ordinary build'
    return row


def score_task(task):
    entry, root = task
    folder = root / entry['dataset'] / entry['id']
    receipt = read(folder / 'receipt.json')
    if receipt['revision'] != MEASURED_REVISION:
        raise ValueError(f'Ordinary build revision differs for {entry["id"]}')
    row = score_build(entry, folder)
    if row['pack_sha256'] != receipt['pack_sha256'] or row['source_sha256'] != receipt['source_sha256']:
        raise ValueError(f'Scored pack differs from the completed-build receipt: {entry["id"]}')
    row['build_seconds'] = receipt['build_seconds']
    return row


def fixed_cases():
    typed = RELEASE / 'typed-structures-01'
    geometry = RELEASE / 'workshop-foreground-01'
    manifest = read(typed / 'summary.json')
    geometry_manifest = read(geometry / 'summary.json')
    for name, path, expected in (
            ('typed cases', typed / 'cases.json', manifest['cases_sha256']),
            ('typed exports', typed / 'typed-exports.zip', manifest['exports_sha256']),
            ('geometry cases', geometry / 'cases.json', manifest['geometry_cases_sha256'])):
        if digest(path) != expected:
            raise ValueError(f'Saved {name} fingerprint differs')
    by_id = {row['id']: row for row in read(geometry / 'cases.json')}
    fixed = [row for row in read(typed / 'cases.json') if row['layout'] is not None]
    if len(fixed) != 104:
        raise ValueError('Published fixed cohort must contain 104 views')
    controls = {frame['id']: control for _, frame, _, control in SUITE.jobs(SUITE.framed.EVALUATION)}
    with ZipFile(typed / 'typed-exports.zip') as archive:
        for case in fixed:
            key = case['id']
            saved = by_id[key]
            control = controls[key]
            payload = archive.read(f'{key}/reconstruction.zip')
            decoded = decoded_pack(BytesIO(payload))
            strict = case['geometry']['strict']
            expected_cells = strict['correct'] + strict['extras']
            if (sha256(payload).hexdigest() != case['pack_sha256'] or len(decoded) != expected_cells
                    or case['source_hash'] != saved['source_hash']
                    or case['source_hash'] != control['rgb_sha256']
                    or case['translation'] != saved['result']['translation']):
                raise ValueError(f'Saved fixed export differs: {key}')
            yield dict(id=key, dataset=FIXED_DATASET, split=case['split'], layout=case['layout'],
                       source_sha256=case['source_hash'], pack_sha256=case['pack_sha256'], geometry=case['geometry'],
                       scores=case['result'], translation=case['translation'], exported=len(decoded),
                       visible_truth=strict['correct'] + strict['missed'],
                       required_truth=case['geometry']['facade']['correct'] + case['geometry']['facade']['missed'],
                       registration=dict(geometry={'strict': saved['result']['registered']}),
                       provenance='released decoded pack at 1ca8822',
                       historical_112_strict_f1=geometry_manifest['historical_112_strict_f1'],
                       **camera_fields(control['status']))


def pooled_typed(scores):
    if all(row['assigned'] is not None for row in scores):
        return pooled_scores(scores)
    placeholders = dict.fromkeys(ASSIGNMENT_FIELDS, 0)
    result = pooled_scores([row | placeholders for row in scores])
    result.update(dict.fromkeys(ASSIGNMENT_FIELDS))
    return result


def pooled(cases, available):
    targets = defaultdict(Counter)
    for row in cases:
        for kind, counts in row.get('targets', {}).items():
            targets[kind].update(counts)
    result = dict(tested=len(cases), available=available, empty_exports=sum(r['exported'] == 0 for r in cases),
                  visible_truth=sum(r['visible_truth'] for r in cases), required_truth=sum(r['required_truth'] for r in cases),
                  exported=sum(r['exported'] for r in cases), geometry=pooled_scores([r['geometry'] for r in cases]),
                  scores=pooled_typed([r['scores'] for r in cases]), targets=dict(targets))
    if cases and all('scores' in row['registration'] for row in cases):
        result['registration'] = dict(geometry=pooled_scores([r['registration']['geometry'] for r in cases]),
                                      scores=pooled_typed([r['registration']['scores'] for r in cases]))
    elif cases:
        result['registration'] = dict(geometry=dict(strict=overlap(*(
            sum(r['registration']['geometry']['strict'][k] for r in cases) for k in ('correct', 'missed', 'extras')))))
    unusable = [row for row in cases if not row['camera_usable']]
    supported = [row for row in cases if row['camera_usable']]
    result['camera'] = dict(usable=len(supported), unusable=len(unusable),
                            misses_strict=sum(row['geometry']['strict']['missed'] for row in unusable),
                            misses_facade=sum(row['geometry']['facade']['missed'] for row in unusable),
                            usable_geometry=pooled_scores([r['geometry'] for r in supported]),
                            usable_scores=pooled_typed([r['scores'] for r in supported]))
    return result


def summarize(cases, expected):
    groups = defaultdict(list)
    for row in cases:
        groups[row['dataset']].append(row)
    summary = {}
    for dataset, count in {FIXED_DATASET: 104, **COHORTS}.items():
        group = groups[dataset]
        split_counts = Counter(row['split'] for row in (group if dataset == FIXED_DATASET else expected)
                               if row['dataset'] == dataset)
        splits = {key: pooled([r for r in group if r['split'] == key], value)
                  for key, value in sorted(split_counts.items())}
        summary[dataset] = dict(pooled(group, count), splits=splits)
    return summary


def confirmation(cases, builds):
    from benchmark.material_seam_confirmation import pack_hash

    by_id = {row['id']: row for row in cases if row['dataset'] == 'decorative-builds-v1'}
    selected = read(RELEASE / 'material-seam-confirmation-01/decorative.json')
    comparisons = []
    for old in selected:
        row = by_id.get(old['id'])
        if row is None:
            continue
        strict, facade = row['geometry']['strict'], row['geometry']['facade']
        old_values = dict(placed=old['placed'], visible_truth=old['visible_truth'],
                          geometry_correct=old['correct_geometry'], geometry_required_correct=old['correct_geometry']-old['optional_placed'],
                          exact_ids=old['baseline'], exact_required_ids=old['facade_before'])
        new_values = dict(placed=row['exported'], visible_truth=row['visible_truth'], geometry_correct=strict['correct'],
                          geometry_required_correct=facade['correct'], exact_ids=row['scores']['strict']['correct'],
                          exact_required_ids=row['scores']['facade']['correct'])
        if row['source_sha256'] != old['rgb_sha256']:
            raise ValueError(f'Decorative confirmation source changed: {old["id"]}')
        pack = builds / row['dataset'] / row['id'] / 'reconstruction.zip'
        decoded = decoded_pack(pack)
        stable_typed_hash = pack_hash({p: name.removeprefix('minecraft:') for p, name in decoded.items()})
        previous_pack = WORKTREES / 'material-seam-confirmation/runs/decorative' / row['id'] / 'reconstruction.zip'
        equal_geometry = None
        if previous_pack.is_file():
            if digest(previous_pack) != old['pack_sha256']:
                raise ValueError(f'Previous geometry pack fingerprint differs: {row["id"]}')
            equal_geometry = decoded.keys() == decoded_pack(previous_pack).keys()
        comparisons.append(dict(id=old['id'], source_sha256=row['source_sha256'], previous=old_values, current=new_values,
                                delta={k: new_values[k]-value for k, value in old_values.items()},
                                pack_sha256=row['pack_sha256'], stable_typed_sha256=stable_typed_hash,
                                previous_stable_typed_sha256=old['baseline_typed_sha256'],
                                identical_ids_and_positions=stable_typed_hash == old['baseline_typed_sha256'],
                                identical_positions=equal_geometry))
    return comparisons


def provenance(builds, cases):
    source = read(builds / 'source_revision.json')
    selected = [row for row in cases if row['dataset'] != FIXED_DATASET]
    for row in selected:
        folder = builds / row['dataset'] / row['id']
        build = read(folder / 'build-manifest.json')
        source_vision = build['surface_finder']['boundary_settings']
        model = source_vision['recovery_inference']
        planarity = source_vision['planarity']['inference']
        yield dict(id=row['id'], dataset=row['dataset'], revision=source['revision'],
                   source_sha256=row['source_sha256'], pack_sha256=row['pack_sha256'],
                   build_manifest_sha256=digest(folder / 'build-manifest.json'),
                   model_sha256=build['model_sha256'], device=model['device'],
                   model_environment=model['model_environment'], planarity_torch=planarity['torch_version'],
                   atlas_sha256=build['placement']['material_assignment']['atlas_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--builds', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'scores')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--partial', action='store_true', help='Score completed build receipts and record pending frames')
    args = parser.parse_args()
    expected = list(frames(args.source, COHORTS))
    present = [r for r in expected if (args.builds / r['dataset'] / r['id'] / 'receipt.json').is_file()]
    missing = [r['id'] for r in expected if r not in present]
    if missing and not args.partial:
        raise ValueError(f'There are {len(missing)} incomplete builds. Use --partial for a progress snapshot.')
    with ProcessPoolExecutor(args.workers) as pool:
        measured = list(pool.map(score_task, ((r, args.builds) for r in present)))
    cases = [*fixed_cases(), *measured]
    save(args.output / 'cases.json', cases)
    save(args.output / 'decorative-confirmation.json', confirmation(cases, args.builds))
    errors = []
    for entry in expected:
        folder = args.builds / entry['dataset'] / entry['id']
        if entry['id'] in missing and (folder / 'failure.json').is_file():
            errors.append(read(folder / 'failure.json'))
    sources = read(args.builds / 'source_revision.json')
    fixed_release = RELEASE / 'typed-structures-01'
    fixed = read(fixed_release / 'summary.json')
    save(args.output / 'manifest.json', dict(schema=1, production_revision=sources['revision'],
        minecraft_version=sources['version'], environment=sources.get('environment'),
        dataset_sha256={r['dataset']: r['dataset_manifest_sha256'] for r in expected},
        fixed=dict(cases_sha256=fixed['cases_sha256'], typed_exports_sha256=fixed['exports_sha256'],
                   geometry_cases_sha256=fixed['geometry_cases_sha256']),
        frames=list(provenance(args.builds, cases))))
    save(args.output / 'summary.json', dict(schema=1, revision=MEASURED_REVISION, version='26.3',
                                            cases_sha256=digest(args.output / 'cases.json'), missing=missing,
                                            manifest_sha256=digest(args.output / 'manifest.json'),
                                            technical_failures=errors,
                                            cohorts=summarize(cases, expected)))
    print(f'Verified 104 released packs and scored {len(measured)}/{len(expected)} newly built packs')


if __name__ == '__main__':
    main()
