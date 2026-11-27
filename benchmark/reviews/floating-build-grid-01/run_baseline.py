"""Build and score the ordinary screenshot-only arm of the paired background test."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from importlib.metadata import version
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[4].parent / 'minecraft-image-reconstruction'
OUTPUT = ROOT / 'runs/floating-build-grid-01/baseline'
sys.path.insert(0, str(ROOT))

from benchmark import expanded_baseline as ordinary
from benchmark.expanded_scores import evaluate_pack, load_reference, pooled
from benchmark.mask_blocks import decoded_pack


REVISION = 'e3bd220188f224a0a6cf2841c8f495fd0ac9d090'
ARMS = ('pavement', 'plain', 'sky')
WORKTREES = ROOT.with_name(ROOT.name + '-worktrees')


def model_settings():
    scale = WORKTREES / 'mask-scalelsd/runs'
    planarity = WORKTREES / 'afterpass-planarity/runs/assets'
    models = WORKTREES / 'surface-strip-entry/.deps'
    paths = {
        'MCIR_RECOVERY_PYTHON': scale / 'env/Scripts/python.exe',
        'MCIR_SCALELSD_SOURCE': scale / 'upstream',
        'MCIR_SCALELSD_CHECKPOINT': scale / 'scalelsd-vitbase-v2-train-sa1b.pt',
        'MCIR_MOBILESAM_SOURCE': WORKTREES / 'surface-recovery-round/runs/mobile-sam-source',
        'MCIR_PLANARITY_SOURCE': planarity / 'PixelwisePlanarity',
        'MCIR_PLANARITY_UTILS3D': planarity / 'utils3d',
        'MCIR_PLANARITY_CHECKPOINT': planarity / 'model.pt',
        'MCIR_PLANARITY_PYTHONPATH': planarity / 'deps',
        'MINECRAFT_MOGE_ONNX': models / 'moge-2-vits.onnx',
        'MINECRAFT_DEPTH_ONNX': models / 'depth-v2-small-uint8.onnx',
        'MINECRAFT_MLSD_TFLITE': models / 'M-LSD_512_tiny_fp16.tflite',
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'{name} at {path}')
    imports = [WORKTREES / 'surface-round/runs/depth-probe/runtime',
               WORKTREES / 'surface-round/runs/mlsd-probe/lib']
    for path in imports:
        if not path.is_dir():
            raise FileNotFoundError(path)
        sys.path.insert(1, str(path))

    os.environ.update({name: str(path) for name, path in paths.items()})
    os.environ.update(PYTHONPATH=os.pathsep.join(map(str, imports)), MCIR_RECOVERY_DEVICE='cuda',
                      MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return paths


def score_variant(entry, image, expected_sha, folder):
    """Decode the variant build and keep the variant RGB and frozen truth fingerprints separate."""
    manifest = ordinary.read(folder / 'build-manifest.json')
    if ordinary.digest(image) != expected_sha or manifest['image_sha256'] != expected_sha:
        raise ValueError(f'Variant source image hash differs: {entry["id"]}')
    if ordinary.digest(Path(entry['frame']) / 'rgb.png') != entry['rgb_sha256']:
        raise ValueError(f'Original source image hash differs: {entry["id"]}')
    pack = folder / 'reconstruction.zip'
    placements = ordinary.read(folder / 'placements.json')['blocks']
    assignment = {(block['x'], block['y'], block['z']): block['material_source'] for block in placements}
    inventory = {(block['x'], block['y'], block['z']): 'minecraft:' + block['block_id'].removeprefix('minecraft:')
                 for block in placements}
    if len(inventory) != len(placements) or inventory != decoded_pack(pack):
        raise ValueError(f'Placement inventory differs from decoded pack: {entry["id"]}')
    result, predicted = evaluate_pack(entry, pack, manifest['placement'], assignment=assignment,
                                      reference=load_reference(entry))
    if result['exported'] != manifest['placement']['placed_blocks']:
        raise ValueError(f'Pack count differs from manifest: {entry["id"]}')
    result.update(source_sha256=expected_sha, original_source_sha256=entry['rgb_sha256'])
    return result, predicted


def run_arm(entry, arm):
    from engine.surface_model import build

    spec = entry['variants'][arm]
    image = Path(spec['image'])
    source_sha = spec['source_sha256']
    if ordinary.digest(image) != source_sha:
        raise ValueError(f'Variant input checksum differs: {image}')
    folder = OUTPUT / arm / entry['dataset'] / entry['id']
    folder.mkdir(parents=True, exist_ok=True)
    receipt_file = folder / 'receipt.json'
    start = perf_counter()
    if receipt_file.is_file():
        receipt = ordinary.read(receipt_file)
        if (receipt['revision'] != REVISION or receipt['source_sha256'] != source_sha
                or receipt['pack_sha256'] != ordinary.digest(folder / 'reconstruction.zip')):
            raise ValueError(f'Existing receipt cannot be replayed: {folder}')
    else:
        with ExitStack() as stack:
            log = stack.enter_context((folder / 'execution.log').open('w', encoding='utf-8'))
            stack.enter_context(redirect_stdout(log))
            stack.enter_context(redirect_stderr(log))
            ordinary.capture_handoffs(stack, folder / 'handoffs')
            build(image, '26.3', folder, through='placement')
        manifest = ordinary.read(folder / 'build-manifest.json')
        if manifest['image_sha256'] != source_sha or manifest['through'] != 'placement':
            raise ValueError(f'Ordinary build provenance differs: {folder}')
        placement = manifest['placement']
        atlas_sha = placement['material_assignment']['atlas_sha256']
        if placement['placed_blocks'] and not atlas_sha:
            raise ValueError(f'Ordinary source material atlas unavailable: {folder}')
        handoff_hashes = {path.name: ordinary.digest(path) for path in sorted((folder / 'handoffs').iterdir())}
        receipt = dict(id=entry['id'], dataset=entry['dataset'], variant=arm, revision=REVISION, image=str(image),
                       source_sha256=source_sha, pack_sha256=ordinary.digest(folder / 'reconstruction.zip'),
                       build_manifest_sha256=ordinary.digest(folder / 'build-manifest.json'), atlas_sha256=atlas_sha,
                       model_sha256=manifest['model_sha256'], placed=placement['placed_blocks'],
                       through=manifest['through'], handoffs_sha256=handoff_hashes,
                       camera_status=placement['camera_status'], build_seconds=round(perf_counter()-start, 2))
        ordinary.save(receipt_file, receipt)
    score, aligned = score_variant(entry, image, source_sha, folder)
    score.update(variant=arm, revision=REVISION, image=str(image), folder=str(folder), provenance='ordinary build',
                 build_seconds=receipt['build_seconds'])
    ordinary.save(folder / 'aligned-placements.json.gz',
                  [[*position, block_id] for position, block_id in sorted(aligned.items())])
    ordinary.save(folder / 'score.json', score)
    (folder / 'failure.json').unlink(missing_ok=True)
    return score


def summarize(rows, cases):
    groups = {'all': cases}
    for dataset in sorted({entry['dataset'] for entry in cases}):
        groups[dataset] = [entry for entry in cases if entry['dataset'] == dataset]
    result = {}
    for key, entries in groups.items():
        ids = {entry['id'] for entry in entries}
        result[key] = {arm: pooled([row for row in rows if row['id'] in ids and row['variant'] == arm], len(ids))
                       for arm in ARMS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=OUTPUT.parent / 'fixtures.json')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--gpu-slots', type=int, default=3)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    paths = model_settings()
    subprocess.run(['git', 'diff', '--quiet', REVISION, '--', *ordinary.PRODUCTION_PATHS], cwd=ROOT, check=True)
    from engine.camera_atlas_floor import _atlas
    atlas = _atlas('26.3')
    if atlas is None:
        raise RuntimeError('The ordinary build needs the locally prepared Java 26.3 atlas')
    environment = dict(python=platform.python_version(), system=platform.system(), device='cuda',
                       packages={p: version(p) for p in ('numpy', 'Pillow', 'onnxruntime', 'ai-edge-litert')},
                       paths={key: str(value) for key, value in paths.items()}, atlas=atlas.provenance)
    if args.preflight:
        print(environment)
        return

    cases = ordinary.read(args.manifest)['cases']
    frozen = {entry['id']: entry for entry in ordinary.frames(ordinary.SOURCE, ordinary.COHORTS)}
    if not cases or len({entry['id'] for entry in cases}) != len(cases):
        raise ValueError('Fixture cases must be present and unique')
    for entry in cases:
        original = frozen[entry['id']]
        if original != {key: entry[key] for key in original} or set(entry['variants']) != set(ARMS):
            raise ValueError(f'Fixture provenance mismatch: {entry["id"]}')
    metadata = dict(revision=REVISION, version='26.3', fixture_manifest=str(args.manifest),
                    fixture_manifest_sha256=ordinary.digest(args.manifest), environment=environment,
                    scorer='benchmark.expanded_scores.evaluate_pack', workers=args.workers, gpu_slots=args.gpu_slots)
    ordinary.save(OUTPUT / 'run.json', metadata)

    spawn = multiprocessing.get_context('spawn')
    gpu_slots = spawn.BoundedSemaphore(args.gpu_slots)
    scores, failures = [], []
    with ProcessPoolExecutor(args.workers, mp_context=spawn, initializer=ordinary.initialize, initargs=(gpu_slots,)) as pool:
        tasks = {pool.submit(run_arm, entry, arm): (entry, arm) for entry in cases for arm in ARMS}
        for processed, task in enumerate(as_completed(tasks), 1):
            entry, arm = tasks[task]
            try:
                score = task.result()
            except Exception as error:
                failure = dict(id=entry['id'], dataset=entry['dataset'], variant=arm,
                               source_sha256=entry['variants'][arm]['source_sha256'], error=repr(error),
                               traceback=traceback.format_exc())
                failures.append(failure)
                ordinary.save(OUTPUT / arm / entry['dataset'] / entry['id'] / 'failure.json', failure)
                print(f'{processed}/{len(tasks)} {entry["id"]} {arm}: {error}', flush=True)
            else:
                scores.append(score)
                strict = score['geometry']['strict']
                print(f'{processed}/{len(tasks)} {entry["id"]} {arm}: {score["exported"]} blocks; '
                      f'camera {score["camera_status"]}; strict {strict["f1"]:.4f} ({score["build_seconds"]} s)', flush=True)
        scores.sort(key=lambda row: (row['dataset'], row['id'], ARMS.index(row['variant'])))
    ordinary.save(OUTPUT / 'cases.json', scores)
    ordinary.save(OUTPUT / 'summary.json', metadata | {'scores': summarize(scores, cases), 'failures': failures})
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
