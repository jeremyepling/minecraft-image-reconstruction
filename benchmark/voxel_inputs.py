"""Freeze independent single-view inputs. Pose and block readback stay with the evaluator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine.voxel_space import prepare


def make(item):
    path, output = item
    try:
        scene = prepare(Image.open(path))
        np.savez_compressed(output, **scene)
        return path.stem, list(scene['hull'].shape)
    except ValueError as exc:
        return path.stem, str(exc)


def cache(output, buildings):
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for root, name in buildings:
        for direction in ('left', 'right'):
            path, = (root/'captures').glob(f'{name}-*{direction}.png')
            other, = (root/'captures').glob(f'{name}-*{"right" if direction=="left" else "left"}.png')
            cases.append(dict(id=path.stem, building=name, image=str(path), other=str(other)))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for name, result in pool.map(make, [(Path(c['image']), output/(c['id']+'.npz')) for c in cases]):
            print(name, result, flush=True)
    (output/'cases.json').write_text(json.dumps(cases, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('groups', nargs='+', help='capture_root|building_id')
    args = parser.parse_args()
    cache(args.output, [(Path(root), name) for root, name in (s.split('|') for s in args.groups)])
