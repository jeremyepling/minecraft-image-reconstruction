"""Replay camera-supported placement against the 56 frozen exterior screenshots."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from benchmark.camera_labels import exposed_faces, rasterize
from benchmark.camera_metrics import registration, truth_camera
from benchmark.camera_round import digest, read, save
from benchmark.mask_blocks import _score, _write_read, registered
from engine.visible_placement import place


SUITE = Path(__file__).resolve().parent / 'reviews/sky-gap-01'
# experiment/local-visible-placement, a11471e: camera-supported results only.
REFERENCE_SUITE = '67fe02ade765f35d5fa6f8eb85d248fd15adf1227717a9ba6209decef67ebe07'
REFERENCE_EXPORTS = 'ef361c190fb718d29afa32decb9bc40f798e4b3d4258e6283502c2fa8005fe48'
REFERENCE_REGISTERED = '403db527339c9f3d6303e7b86835ec5aad80fabb0e1d0a4b5639712646bdd231'


def signature(value):
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def projected_shape(cells, pose, target):
    """Project the exported cubes at 4-pixel sample spacing in the original camera."""
    h, w = target.shape
    projected = np.zeros((h, w), bool)
    P = truth_camera(pose)
    P[:2] /= 4
    for _, _, _, corners in exposed_faces(cells, pose['eye_world']):
        raster = rasterize(corners @ P[:, :3].T + P[:, 3], w, h)
        if raster is not None:
            box, polygon, *_ = raster
            projected[box] |= polygon
    return dict(overlap=int((projected & target).sum()), union=int((projected | target).sum()))


def replay(row, *, frames, out, hashes):
    key = row['key']
    frame = frames / ('legacy-' + key.replace('/', '-'))
    image_path, mask_path = frame / 'rgb.png', SUITE / row['predictions']
    rgb_sha, mask_sha = digest(image_path), digest(mask_path)
    if rgb_sha != row['sha256'] or mask_sha != hashes[row['predictions']]:
        raise ValueError(f'Frozen input mismatch for {key}')
    parts = json.loads(gzip.decompress(mask_path.read_bytes()))['gap']
    with Image.open(image_path) as image:
        cells, status = place(image.convert('RGB'), '26.3', parts)
    pack = out / 'packs' / (key + '.zip')
    pack.parent.mkdir(parents=True, exist_ok=True)
    emitted = _write_read(pack, cells)

    pose_path, truth_path, raster_path = (frame / name for name in ('camera.json', 'labels.json', 'labels.npz'))
    pose = read(pose_path)
    target = {tuple(b['xyz_world']) for b in read(truth_path)['blocks']}
    P = status['P']
    if P is None and emitted:
        raise ValueError(f'Unregistered output on {key}')
    world = registered(emitted, registration(np.asarray(P), pose)) if emitted else set()
    with np.load(raster_path) as labels:
        shape = projected_shape(world, pose, labels['face_id'][2::4, 2::4] > 0)
    return dict(key=key, cohort=row['cohort'], camera=status['camera_status'], score=_score(world, target),
                shape=shape, rgb_sha256=rgb_sha, masks_sha256=mask_sha, labels_sha256=digest(truth_path),
                pose_sha256=digest(pose_path), raster_sha256=digest(raster_path),
                exported=sorted(emitted), registered=sorted(world))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', required=True, type=Path, help='visible-builds-v1/data/regression/frames')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--workers', default=4, type=int)
    p.add_argument('--verify-experiment', action='store_true', help='Require exact parity with frozen supported exports')
    args = p.parse_args()
    manifest = read(SUITE / 'manifest.json')
    if digest(SUITE / 'cases.json') != manifest['artifacts']['cases.json']:
        raise ValueError('Frozen case index changed')
    run = partial(replay, frames=args.frames, out=args.out, hashes=manifest['artifacts'])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, read(SUITE / 'cases.json')))
    emitted_sha = signature({r['key']: r.pop('exported') for r in rows})
    world_sha = signature({r['key']: r.pop('registered') for r in rows})
    score, shape = Counter(), Counter()
    for r in rows:
        score.update(r['score'])
        shape.update(r['shape'])
    root = SUITE.parents[2]
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    summary = dict(version=manifest['version'], revision=revision, placement_sha256=digest(root/'engine/visible_placement.py'),
                   frames=len(rows), producing=sum(bool(r['score']['serialized']) for r in rows),
                   camera=Counter(r['camera'] for r in rows), score=score, shape=shape,
                   suite_sha256=digest(SUITE / 'manifest.json'), exported_sha256=emitted_sha,
                   registered_sha256=world_sha)
    save(args.out / 'results.json', dict(summary=summary, frames=rows))
    print(json.dumps(summary, indent=2))
    if args.verify_experiment and (summary['suite_sha256'], emitted_sha, world_sha) != (
            REFERENCE_SUITE, REFERENCE_EXPORTS, REFERENCE_REGISTERED):
        raise SystemExit('Production placements differ from the frozen camera-supported experiment.')


if __name__ == '__main__':
    os.environ['MCIR_IMAGE_ONLY'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    main()
