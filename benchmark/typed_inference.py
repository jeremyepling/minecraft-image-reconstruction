"""Run the source-only material matchers on frozen final geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

from benchmark.typed_structures import CACHE, checked, compact, digest, read, viewer
from PIL import Image


@lru_cache(maxsize=1)
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def calibrated_atlas(loader, fingerprint):
    if digest(loader) != fingerprint:
        raise ValueError('Atlas loader changed during prediction')
    return load_module('engine.atlas_count', loader).load_atlas(viewer.suite.ATLAS, '26.3')


@lru_cache(maxsize=1)
def active_predictor(snapshot, scene, scene_hash):
    if digest(scene) != scene_hash:
        raise ValueError('Source projection changed during prediction')
    sys.modules['engine.projected_appearance'] = load_module('engine.projected_appearance', scene)
    return load_module('engine.placed_materials', snapshot)


def infer(task):
    key, snapshot, fingerprint, loader, loader_hash, scene, scene_hash, out, cache, methods = task
    entry = read(cache/'inputs'/(key+'.json'))
    cells = list(map(tuple, entry['cells']))
    with Image.open(checked(Path(entry['rgb']), entry['source_hash'])) as image:
        proposals = active_predictor(snapshot, scene, scene_hash).assign_many(
            image, entry['camera'], cells, calibrated_atlas(loader, loader_hash), methods=methods)
    if set(proposals) != set(methods):
        raise ValueError('Incomplete set of material candidates')
    predictions = {}
    for name, (ids, sources) in proposals.items():
        if set(cells) != ids.keys() or ids.keys() != sources.keys():
            raise ValueError(f'Changed predicted geometry in {key}')
        predictions[name] = [[ids[c], sources[c]] for c in cells]
    compact(out/(key+'.json'), dict(source_hash=entry['source_hash'], geometry_pack_sha256=entry['geometry_pack_sha256'],
                                    predictor_sha256=fingerprint, methods=predictions))
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', type=Path, default=Path(__file__).resolve().parents[1]/'engine/placed_materials.py')
    parser.add_argument('--atlas-loader', type=Path, help='Defaults to atlas_count.py beside the predictor')
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['face', 'palette', 'auto'])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--split', choices=['train', 'all'], default='train')
    args = parser.parse_args()
    keys = read(args.cache/'manifest.json')['frames']
    if args.split == 'train':
        keys = [k for k in keys if k.startswith('train-')]
        if len(keys) != 72:
            raise ValueError('Unexpected training split')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'predictor.py'
    # Workers run a byte-identical snapshot even if a development worktree changes.
    path.write_bytes(args.predictor.read_bytes())
    fingerprint = digest(path)
    loader = args.atlas_loader.resolve() if args.atlas_loader else args.predictor.resolve().with_name('atlas_count.py')
    loader_hash = digest(loader)
    scene = loader.with_name('projected_appearance.py')
    scene_hash = digest(scene)
    tasks = [(k, path, fingerprint, loader, loader_hash, scene, scene_hash, args.out, args.cache, args.methods) for k in keys]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, key in enumerate(pool.map(infer, tasks), 1):
            if i % 20 == 0:
                print('Saved', i, 'of', len(keys), flush=True)
    compact(args.out/'run.json', dict(keys=keys, methods=args.methods, predictor_sha256=fingerprint,
                                      atlas_loader_sha256=loader_hash, projected_appearance_sha256=scene_hash))


if __name__ == '__main__':
    main()
