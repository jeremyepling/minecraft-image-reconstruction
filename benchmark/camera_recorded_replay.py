"""Replay the ordinary camera handoff with the captured Java 26.3 projection."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
from zipfile import ZipFile

os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

import numpy as np

from benchmark.camera_metrics import registration, truth_camera
from benchmark.expanded_baseline import (
    COHORTS, MEASURED_REVISION, PRODUCTION_PATHS, ROOT, SOURCE, WORKTREES,
    digest, frames, read, replay_source_fields, save,
)
from benchmark.expanded_scores import FIXED_DATASET, load_reference, pooled, score_build, summarize
from benchmark.mask_blocks import decoded_pack
from calibration.world import GROUND
from engine.geometry import project


BASELINE = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01'
RELEASE = ROOT / 'benchmark/reviews/expanded-baseline-01'
OUTPUT = ROOT / 'runs/camera-recorded-pose-01'
METHOD = 'recorded-pose'
PIXEL_TOLERANCE = 1e-6
CONTROL_COUNTS = {FIXED_DATASET: 0, 'visible-builds-v3': 108, 'decorative-builds-v1': 10}


def recorded_camera(entry):
    folder = Path(entry['frame'])
    reference = load_reference(entry)
    pose = reference.pose
    origin = np.floor(pose['eye_world']).astype(int)
    origin[1] = GROUND + 1
    world_from_local = np.eye(4)
    world_from_local[:3, 3] = origin
    projection = truth_camera(pose) @ world_from_local
    if not np.array_equal(registration(projection, pose), world_from_local):
        raise ValueError(f'Exact projection did not recover its integer origin: {entry["id"]}')

    faces = read(folder / 'labels.json')['faces']
    vertices = {
        tuple(corner): pixel
        for face in faces
        for corner, pixel, visible in zip(face['corners_world'], face['corners_px'], face['corner_visible'], strict=True)
        if visible
    }
    local = np.asarray(list(vertices)) - origin
    depth = local @ projection[2, :3] + projection[2, 3]
    errors = np.linalg.norm(project(projection, local) - list(vertices.values()), axis=1)
    if not len(errors) or np.any(depth <= 0) or not np.all(np.isfinite(errors) & (errors <= PIXEL_TOLERANCE)):
        raise ValueError(f'Recorded native corners do not match calibrated projection: {entry["id"]}')
    validation = dict(vertices=len(errors), tolerance_px=PIXEL_TOLERANCE,
                      max_reprojection_px=float(np.max(errors)), world_origin=origin.tolist())
    camera = dict(P=projection.tolist(), scale_status='supported', camera_source=METHOD,
                  reason='Recorded Java 26.3 camera projection in ground-top block coordinates')
    return camera, validation


def replay(task):
    entry, baseline_root, output_root, no_op = task
    from engine import surface_model, visible_placement

    key = entry['id']
    baseline = baseline_root / entry['dataset'] / key
    output = output_root / ('no-op' if no_op else entry['dataset']) / key
    row_file = output / 'score.json'
    image = Path(entry['frame']) / 'rgb.png'
    camera, validation = (None, None) if no_op else recorded_camera(entry)
    existing = read(row_file) if row_file.is_file() else None
    pack = output / 'reconstruction.zip'
    if existing and pack.is_file() and existing['pack_sha256'] == digest(pack):
        manifest = output / 'build-manifest.json'
        if (existing['source_sha256'] != entry['rgb_sha256'] or existing['camera_no_op'] != no_op
                or existing['build_manifest_sha256'] != digest(manifest)
                or (camera is not None and read(manifest)['placement']['P'] != camera['P'])):
            raise ValueError(f'Saved replay sources differ: {key}')
        return existing
    output.mkdir(parents=True, exist_ok=True)
    original = visible_placement.seam_camera
    calls = []

    def handoff(*args, **kwargs):
        result = original(*args, **kwargs) if no_op else camera
        calls.append(result)
        return result

    with replay_source_fields(image, baseline), patch.object(visible_placement, 'seam_camera', handoff):
        surface_model.build(image, '26.3', output, through='placement')
    manifest = read(output / 'build-manifest.json')
    status = manifest['placement']
    if not calls or status['P'] != calls[0]['P']:
        raise ValueError(f'Camera handoff did not reach exported placement: {key}')
    row = score_build(entry, output)
    baseline_manifest = read(baseline / 'build-manifest.json')
    if (manifest['model_sha256'] != baseline_manifest['model_sha256']
            or manifest['placement']['material_assignment']['atlas_sha256']
            != baseline_manifest['placement']['material_assignment']['atlas_sha256']):
        raise ValueError(f'Model or atlas fingerprint changed: {key}')
    if no_op:
        old_pack = baseline / 'reconstruction.zip'
        new_pack = output / 'reconstruction.zip'
        if decoded_pack(old_pack) != decoded_pack(new_pack) or baseline_manifest['placement'] != status:
            raise ValueError(f'No-op replay differs from the ordinary decoded export or camera status: {key}')
        row['baseline_pack_sha256'] = digest(old_pack)
        row['identical_pack_bytes'] = row['baseline_pack_sha256'] == row['pack_sha256']
    row.update(camera_no_op=no_op, camera_calls=len(calls), camera_validation=validation,
               build_manifest_sha256=digest(output / 'build-manifest.json'))
    save(row_file, row)
    return row


def load_controls(baseline, entries):
    release = read(RELEASE / 'summary.json')
    cases = RELEASE / 'cases.json'
    if release['revision'] != MEASURED_REVISION or release['cases_sha256'] != digest(cases):
        raise ValueError('Frozen ordinary release provenance differs')
    controls = read(cases)
    frame_entries = {entry['id']: entry for entry in entries}
    control_entries = {row['id'] for row in controls if row['dataset'] != FIXED_DATASET}
    if len(controls) != 104 + len(frame_entries) or control_entries != frame_entries.keys():
        raise ValueError('Frozen ordinary release frame set differs')
    fixed_archive = ROOT / 'benchmark/reviews/typed-structures-01/typed-exports.zip'
    with ZipFile(fixed_archive) as archive:
        for row in controls:
            key = row['id']
            if row['dataset'] == FIXED_DATASET:
                pack = archive.read(f'{key}/reconstruction.zip')
                actual_hash = sha256(pack).hexdigest()
                decoded_count = len(decoded_pack(BytesIO(pack)))
                if decoded_count != row['exported']:
                    raise ValueError(f'Frozen fixed pack count differs: {key}')
            else:
                entry = frame_entries[key]
                folder = baseline / row['dataset'] / key
                receipt = read(folder / 'receipt.json')
                if (receipt['revision'] != MEASURED_REVISION or receipt['pack_sha256'] != row['pack_sha256']
                        or row['source_sha256'] != entry['rgb_sha256']
                        or receipt['source_sha256'] != entry['rgb_sha256']
                        or row['dataset_manifest_sha256'] != entry['dataset_manifest_sha256']
                        or receipt['placed'] != row['exported']
                        or receipt['camera_status'] != row['camera_status']):
                    raise ValueError(f'Frozen expanded build provenance differs: {key}')
                actual_hash = digest(folder / 'reconstruction.zip')
            if row['pack_sha256'] != actual_hash:
                raise ValueError(f'Frozen export fingerprint differs: {key}')
            if not row['camera_usable'] and row['exported']:
                raise ValueError(f'Rejected view has a frozen nonempty export: {key}')
    return controls, release['cases_sha256']


def verify_no_op(row, baseline, output):
    key = row['id']
    old = baseline / row['dataset'] / key / 'reconstruction.zip'
    new = output / 'no-op' / key / 'reconstruction.zip'
    with ZipFile(old) as original, ZipFile(new) as replayed:
        names = original.namelist()
        if names != replayed.namelist() or any(original.read(name) != replayed.read(name) for name in names):
            raise ValueError(f'No-op replay changed the export payload: {key}')
    return dict(id=key, baseline_pack_sha256=row['baseline_pack_sha256'], pack_sha256=row['pack_sha256'],
                identical_pack_bytes=row['identical_pack_bytes'], identical_pack_members=True,
                identical_placement_status=True, identical_decoded_export=True)


def report(output, baseline_root, entries, controls, by_id, no_ops, cases_sha):
    paired = []
    for baseline in controls:
        key = baseline['id']
        recovered = by_id.get(key)
        if recovered is not None:
            if baseline['camera_usable'] or baseline['reference_sha256'] != recovered['reference_sha256']:
                raise ValueError(f'Invalid camera replay pair: {key}')
        source = baseline if recovered is None else recovered
        paired.append(dict(
            id=key, dataset=baseline['dataset'], split=baseline['split'], layout=baseline['layout'],
            baseline=baseline, source=source, replayed=recovered is not None,
            retained_verified_pack=recovered is None and baseline['camera_usable'],
        ))
    old = summarize(controls, entries)
    new = summarize([p['source'] for p in paired], entries)

    def group(pairs):
        rejected = [p for p in pairs if not p['baseline']['camera_usable']]
        replayed = [p for p in rejected if p['replayed']]
        original, reference = ([p[key] for p in replayed] for key in ('baseline', 'source'))
        return dict(
            ordinary_camera_rejects=len(rejected), replayed=len(replayed),
            reference_camera_accepted=sum(r['camera_usable'] for r in reference),
            recovered_nonempty=sum(r['exported'] > 0 for r in reference),
            retained_verified_packs=sum(p['retained_verified_pack'] for p in pairs),
            replayed_only=dict(baseline=pooled(original, len(rejected)), source=pooled(reference, len(rejected))),
        )

    datasets = {}
    for dataset in old:
        matching = [p for p in paired if p['dataset'] == dataset]
        datasets[dataset] = dict(baseline=old[dataset], source=new[dataset], **group(matching))
        datasets[dataset]['splits'] = {}
        for split, original in old[dataset]['splits'].items():
            pairs = [p for p in matching if p['split'] == split]
            datasets[dataset]['splits'][split] = dict(
                baseline=original, source=new[dataset]['splits'][split], **group(pairs))
    castles = [p for p in paired if p['baseline'].get('family') == 'castle']
    vertices = [r['camera_validation'] for r in by_id.values()]
    save(output / 'cases.json.gz', paired)
    save(output / 'summary.json', dict(
        schema=1, version='26.3', method=METHOD, base_revision=MEASURED_REVISION,
        baseline_cases_sha256=cases_sha, paired_cases_sha256=digest(output / 'cases.json.gz'),
        dataset_sha256={r['dataset']: r['dataset_manifest_sha256'] for r in entries},
        **group(paired), datasets=datasets,
        castle=dict(**group(castles), cases=[p['id'] for p in castles]),
        native_corner_validation=dict(frames=len(vertices), vertices=sum(v['vertices'] for v in vertices),
            tolerance_px=PIXEL_TOLERANCE, max_reprojection_px=max((v['max_reprojection_px'] for v in vertices), default=None)),
        no_op=[verify_no_op(row, baseline_root, output) for row in no_ops],
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--no-op', nargs='*', default=())
    args = parser.parse_args()
    subprocess.run(['git', 'diff', '--quiet', MEASURED_REVISION, '--', *PRODUCTION_PATHS], cwd=ROOT, check=True)
    entries = list(frames(args.source, COHORTS))
    controls, cases_sha = load_controls(args.baseline, entries)
    rejected = {r['id'] for r in controls if not r['camera_usable']}
    actual_counts = {dataset: sum(r['id'] in rejected for r in controls if r['dataset'] == dataset)
                     for dataset in CONTROL_COUNTS}
    if actual_counts != CONTROL_COUNTS or len(controls) - len(rejected) != 170:
        raise ValueError(f'Frozen camera cohort differs: {actual_counts}')
    selected = rejected if args.ids is None else set(args.ids)
    if not selected <= rejected or not set(args.no_op) <= {r['id'] for r in entries}:
        raise ValueError('Requested replay outside its applicable cohort')
    by_id = {}
    no_ops = []
    tasks = [(e, args.baseline, args.output, True) for e in entries if e['id'] in args.no_op]
    tasks += [(e, args.baseline, args.output, False) for e in entries if e['id'] in selected]
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(replay, task) for task in tasks]
        for future in as_completed(jobs):
            row = future.result()
            if row['camera_no_op']:
                no_ops.append(row)
            else:
                by_id[row['id']] = row
            print(len(by_id), len(selected), 'no-op' if row['camera_no_op'] else 'recorded',
                  row['id'], row['exported'], row['geometry']['strict']['f1'], flush=True)
    report(args.output, args.baseline, entries, controls, by_id, sorted(no_ops, key=lambda r: r['id']), cases_sha)


if __name__ == '__main__':
    main()
