"""Run and retain ordinary screenshot-only builds on the two expanded cohorts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import gzip
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from unittest.mock import patch
from zipfile import BadZipFile


ROOT = Path(__file__).resolve().parents[1]
STABLE = ROOT.parent.parent / 'minecraft-image-reconstruction' if ROOT.parent.name.endswith('-worktrees') else ROOT
WORKTREES = STABLE.with_name(STABLE.name + '-worktrees')
SOURCE = STABLE / 'runs'
OUTPUT = ROOT / 'runs/expanded-baseline-01'
COHORTS = {'visible-builds-v3': 144, 'decorative-builds-v1': 40}
MANIFEST_SHA256 = {
    'visible-builds-v3': '6e7b25fa56aa39d04a34bb3984551cf11441c11acb1a021b045d4f86e824aa28',
    'decorative-builds-v1': '09d5612ee175718b979c73363eb40fe05959d90f96aab894e82efa9548eaf064',
}
MEASURED_REVISION = '1ca88221bf2a29d4495b61a783b0c99273ee0634'
GPU_SLOTS = None
PRODUCTION_PATHS = ('engine', 'runtime', 'reconstruct.py', 'benchmark/data/count-atlas-v1',
                    'benchmark/depth_probe.py', 'benchmark/depth_pro_fields.py', 'benchmark/mask_metrics.py',
                    'benchmark/mobile_region_probe.py', 'benchmark/mobile_wall_split.py',
                    'benchmark/oblique_recess.py', 'benchmark/scalelsd_fields.py')


def read(path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).encode()
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == '.gz' else payload)
    temporary.replace(path)


def frames(source, cohorts):
    for cohort in cohorts:
        dataset = source / cohort
        manifest = read(dataset / 'manifest.json')
        rows = manifest['frames']
        if (manifest['dataset'] != cohort or manifest['minecraft_version'] != '26.3'
                or len(rows) != COHORTS[cohort] or len({row['id'] for row in rows}) != len(rows)):
            raise ValueError(f'Unexpected frame set in {dataset}')
        manifest_sha = digest(dataset / 'manifest.json')
        if manifest_sha != MANIFEST_SHA256[cohort]:
            raise ValueError(f'Packaged dataset manifest checksum mismatch: {dataset}')
        split_by_layout = {}
        for entry in rows:
            if split_by_layout.setdefault(entry['layout'], entry['split']) != entry['split']:
                raise ValueError(f'Layout {entry["layout"]} crosses frozen splits in {dataset}')
            yield {**entry, 'dataset': cohort, 'dataset_manifest_sha256': manifest_sha,
                   'frame': str(dataset / 'data' / entry['split'] / 'frames' / entry['id'])}


def initialize(gpu_slots):
    global GPU_SLOTS
    GPU_SLOTS = gpu_slots
    os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    from engine.camera_atlas_floor import _atlas
    if _atlas('26.3') is None:
        raise RuntimeError('The ordinary build needs the locally prepared Java 26.3 atlas')


def capture_handoffs(stack, folder):
    import numpy as np
    from engine import mlsd_tiny, placed_materials, planarity_entry, recovery_evidence, surface_fields, visible_placement

    folder.mkdir(parents=True, exist_ok=True)
    infer_field, predict_lines = surface_fields.infer, mlsd_tiny.detect
    infer_recovery, infer_planarity = recovery_evidence.infer, planarity_entry.infer
    place, assign = visible_placement.place, placed_materials.assign

    def field(image, kind):
        prediction = infer_field(image, kind)
        np.savez_compressed(folder / f'{kind}.npz', value=prediction)
        return prediction

    def lines(image):
        output = predict_lines(image)
        save(folder / 'mlsd.json.gz', output)
        return output

    def recovery(source, walls):
        with GPU_SLOTS:
            evidence, masks = infer_recovery(source, walls)
        save(folder / 'recovery.json.gz', {'walls': [p['support'] for p in walls], 'evidence': evidence})
        np.savez_compressed(folder / 'recovery-masks.npz', packed=np.packbits(masks, axis=2), width=masks.shape[2])
        return evidence, masks

    def planarity(source, device, *, crops=False):
        with GPU_SLOTS:
            fields, info = infer_planarity(source, device, crops=crops)
        np.savez_compressed(folder / 'planarity.npz', **fields)
        save(folder / 'planarity.json.gz', {'device': device, 'crops': crops, 'info': info})
        return fields, info

    def ranked_placement(image, version, parts, **options):
        # The ordinary caller explicitly passes the choice. An ensuing local
        # smooth-extent pass also calls place() with its defaults.
        capture = 'joint_foreground' in options
        if capture:
            save(folder / 'placement-input.json.gz', {'parts': parts, 'counts': options['counts'],
                'camera': options['camera'], 'joint_foreground': options['joint_foreground']})
        cells, status = place(image, version, parts, **options)
        if capture:
            save(folder / 'placement-output.json.gz', {'cells': sorted(cells), 'status': status})
        return cells, status

    def material(image, camera, cells, atlas):
        save(folder / 'material-input.json.gz', {'cells': sorted(cells), 'camera': camera,
                                                 'atlas': atlas.provenance if atlas is not None else None})
        return assign(image, camera, cells, atlas)

    for module, attribute, wrapper in ((surface_fields, 'infer', field), (mlsd_tiny, 'detect', lines),
                                       (recovery_evidence, 'infer', recovery), (planarity_entry, 'infer', planarity),
                                       (visible_placement, 'place', ranked_placement), (placed_materials, 'assign', material)):
        stack.enter_context(patch.object(module, attribute, wrapper))


@contextmanager
def _replay_fields(image, folder, source_sha, allow_changed_recovery_prompts, allow_missing):
    import numpy as np
    from engine import mlsd_tiny, planarity_entry, recovery_evidence, surface_fields

    if digest(image) != source_sha:
        raise ValueError('Cannot replay fields from a different source screenshot')
    infer_field, predict_lines = surface_fields.infer, mlsd_tiny.detect
    infer_recovery, infer_planarity = recovery_evidence.infer, planarity_entry.infer

    def field(image, kind):
        try:
            with np.load(folder / f'{kind}.npz', allow_pickle=False) as data:
                return data['value']
        except (FileNotFoundError, BadZipFile):
            if not allow_missing:
                raise
            return infer_field(image, kind)

    def lines(image):
        try:
            return read(folder / 'mlsd.json.gz')
        except FileNotFoundError:
            if not allow_missing:
                raise
            return predict_lines(image)

    def recovery(source, walls):
        try:
            data = read(folder / 'recovery.json.gz')
            with np.load(folder / 'recovery-masks.npz', allow_pickle=False) as masks:
                unpacked = np.unpackbits(masks['packed'], count=int(masks['width']), axis=2).astype(bool)
        except (FileNotFoundError, BadZipFile):
            if not allow_missing:
                raise
            return infer_recovery(source, walls)
        if [p['support'] for p in walls] != data['walls']:
            if allow_changed_recovery_prompts:
                return infer_recovery(source, walls)
            raise ValueError('MobileSAM prompt supports changed; enable fresh recovery inference')
        return data['evidence'], unpacked

    def planarity(source, device, *, crops=False):
        try:
            metadata = read(folder / 'planarity.json.gz')
            with np.load(folder / 'planarity.npz', allow_pickle=False) as fields:
                values = dict(fields)
        except (FileNotFoundError, BadZipFile):
            if not allow_missing:
                raise
            return infer_planarity(source, device, crops=crops)
        if (device, crops) != (metadata['device'], metadata['crops']):
            raise ValueError('Planarity device or crop choice differs from the captured settings')
        return values, metadata['info']

    with ExitStack() as stack:
        for module, attribute, producer in ((surface_fields, 'infer', field), (mlsd_tiny, 'detect', lines),
                                            (recovery_evidence, 'infer', recovery), (planarity_entry, 'infer', planarity)):
            stack.enter_context(patch.object(module, attribute, producer))
        yield


def replay_source_fields(image, baseline, *, allow_changed_recovery_prompts=False):
    """Replay frozen model evidence around an ordinary build on the same screenshot.

    Install an experiment's hooks inside this context. Changed ScaleLSD/MobileSAM
    prompts require a fresh call to the real producer, explicitly enabled here.
    No reference inputs are loaded and ordinary downstream consumers still run.
    """
    receipt = read(baseline / 'receipt.json')
    return _replay_fields(image, baseline / 'handoffs', receipt['source_sha256'],
                          allow_changed_recovery_prompts, False)


def placement_input(baseline):
    return read(baseline / 'handoffs' / 'placement-input.json.gz')


def material_input(baseline):
    return read(baseline / 'handoffs' / 'material-input.json.gz')


def build_frame(task):
    entry, output, revision, reuse_partial = task
    from engine.surface_model import build

    image = Path(entry['frame']) / 'rgb.png'
    if digest(image) != entry['rgb_sha256']:
        raise ValueError(f'Frozen source checksum mismatch for {image}')
    folder = output / entry['dataset'] / entry['id']
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'receipt.json'
    if receipt.exists():
        previous = read(receipt)
        if previous['revision'] != revision or previous['source_sha256'] != entry['rgb_sha256']:
            raise ValueError(f'Existing build is from a different revision or source: {folder}')
        if digest(folder / 'reconstruction.zip') != previous['pack_sha256']:
            raise ValueError(f'Existing export checksum mismatch: {folder}')
        return previous
    start = perf_counter()
    with ExitStack() as stack:
        log = stack.enter_context((folder / 'execution.log').open('w', encoding='utf-8'))
        stack.enter_context(redirect_stdout(log))
        stack.enter_context(redirect_stderr(log))
        if reuse_partial:
            stack.enter_context(_replay_fields(image, folder / 'handoffs', entry['rgb_sha256'], False, True))
        capture_handoffs(stack, folder / 'handoffs')
        build(image, '26.3', folder, through='placement')
    manifest = read(folder / 'build-manifest.json')
    if manifest['image_sha256'] != entry['rgb_sha256'] or manifest['through'] != 'placement':
        raise ValueError(f'Build provenance mismatch for {folder}')
    material = manifest['placement']['material_assignment']
    if manifest['placement']['placed_blocks'] and material['atlas_sha256'] is None:
        raise ValueError(f'Ordinary material atlas unavailable for {folder}')
    result = dict(id=entry['id'], dataset=entry['dataset'], revision=revision, source_sha256=entry['rgb_sha256'],
                  pack_sha256=digest(folder / 'reconstruction.zip'), atlas_sha256=material['atlas_sha256'],
                  model_sha256=manifest['model_sha256'], placed=manifest['placement']['placed_blocks'],
                  camera_status=manifest['placement']['camera_status'], build_seconds=round(perf_counter()-start, 2))
    save(receipt, result)
    (folder / 'failure.json').unlink(missing_ok=True)
    return result


def run(args):
    revision = MEASURED_REVISION
    subprocess.run(['git', 'diff', '--quiet', revision, '--', *PRODUCTION_PATHS],
                   cwd=ROOT, check=True)
    frozen = list(frames(args.source, args.datasets))
    pending = frozen
    if args.ids:
        pending = [row for row in pending if row['id'] in args.ids]
        if len(pending) != len(set(args.ids)):
            raise ValueError('An explicitly requested frame was not in the frozen manifests')
    if args.priority_ids:
        priority = {key: i for i, key in enumerate(dict.fromkeys(args.priority_ids))}
        pending = sorted(pending, key=lambda row: priority.get(row['id'], len(priority)))
    run_file = args.output / 'source_revision.json'
    previous = read(run_file) if run_file.is_file() else {}
    if previous and previous['revision'] != revision:
        raise ValueError('Existing run has a different production revision')
    by_id = {r['id']: r for r in previous.get('frames', ())}
    reused = {r['id'] for r in pending if r['id'] in by_id and r['rgb_sha256'] == by_id[r['id']]['rgb_sha256']}
    from engine.camera_atlas_floor import _atlas
    atlas = _atlas('26.3')
    if atlas is None:
        raise RuntimeError('Java 26.3 atlas is unavailable')
    environment = {'python': platform.python_version(), 'system': platform.system(),
                   'packages': {p: version(p) for p in ('numpy', 'Pillow', 'onnxruntime', 'ai-edge-litert')},
                   'device': os.environ.get('MCIR_RECOVERY_DEVICE', 'ordinary auto selection'),
                   'build_workers': args.workers, 'model_process_limit': args.gpu_slots,
                   'atlas': atlas.provenance}
    prior_environment = previous.get('environment')
    if prior_environment and any(environment[k] != prior_environment[k]
                                 for k in ('python', 'system', 'packages', 'device', 'atlas')):
        raise ValueError('Existing run has a different numerical environment or atlas')
    save(run_file, {'revision': revision, 'version': '26.3', 'frames': frozen, 'environment': environment})
    failures = []
    spawn = multiprocessing.get_context('spawn')
    gpu_slots = spawn.BoundedSemaphore(args.gpu_slots)
    with ProcessPoolExecutor(args.workers, mp_context=spawn, initializer=initialize, initargs=(gpu_slots,)) as pool:
        tasks = {pool.submit(build_frame, (row, args.output, revision, row['id'] in reused)): row for row in pending}
        for number, task in enumerate(as_completed(tasks), 1):
            row = tasks[task]
            try:
                result = task.result()
            except Exception as error:
                print(f'{number}/{len(tasks)} {row["id"]}: ERROR {error}', flush=True)
                failure = dict(id=row['id'], dataset=row['dataset'], source_sha256=row['rgb_sha256'], error=repr(error))
                failures.append(failure)
                save(args.output / row['dataset'] / row['id'] / 'failure.json', failure)
            else:
                print(f'{number}/{len(tasks)} {row["id"]}: {result["placed"]} blocks ({result["build_seconds"]}s)', flush=True)
    save(args.output / 'build_failures.json', failures)
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--datasets', nargs='+', default=list(COHORTS), choices=COHORTS)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--priority-ids', nargs='+')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu-slots', type=int, default=3)
    args = parser.parse_args()
    if args.gpu_slots < 1 or args.workers < 1:
        parser.error('Both worker limits must be positive integers')
    run(args)


if __name__ == '__main__':
    main()
