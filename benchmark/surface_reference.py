"""Saved, evaluation-only visible geometry for the fixed single-image benchmark."""
import argparse
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from benchmark import surface_metrics as metrics
from benchmark.camera_labels import rasterize, screen_matrix
from benchmark.camera_metrics import truth_camera
from benchmark.camera_round import ROOT, digest, read
from benchmark.exterior_set import Capture
from engine.geometry import UNIT, project


ARCHIVE = ROOT / 'benchmark/data/visible-geometry-v1.zip'
SCHEMA = 1


def _json(data):
    return json.dumps(data, separators=(',', ':'), allow_nan=False).encode('utf-8')


@lru_cache(maxsize=1)
def generator_hash():
    """Invalidate geometry changes independently of prediction and score changes."""
    functions = (screen_matrix, truth_camera, rasterize, project, metrics._components,
                 metrics._largest_rect, metrics._patch_chart, metrics._rectangle, metrics.truth)
    settings = [UNIT.tolist(), metrics.DOWNSAMPLE, metrics.MIN_PATCH_FACES, metrics.MIN_SUPPORT,
                metrics.INSET_LOW.tolist(), metrics.INSET_HIGH.tolist()]
    code = [inspect.getsource(func) for func in functions]
    return hashlib.sha256(_json([settings, code])).hexdigest()


def _inputs(case):
    return {name: digest(case.path(name)) for name in ('image', 'pose', 'labels', 'states')}


def _encode(t):
    # Most of the pixel map is background or a connected flat plane. Store consecutive
    # equal IDs as runs in row-major order. All other geometry stays ordinary JSON.
    ids = t['id_map'].ravel()
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    mask = dict(shape=list(t['id_map'].shape), lengths=np.diff(np.r_[starts, len(ids)]).tolist(),
                ids=ids[starts].tolist())
    planes = [dict(patch=p, chart=t['charts'][i].tolist(), vertices=t['vertices'][i].tolist())
              for i, p in t['patches'].items()]
    return _json(dict(image_size=t['image_size'], id_map=mask, planes=planes))


def _decode(payload):
    data = json.loads(payload)
    raster = data['id_map']
    ids = np.repeat(np.asarray(raster['ids'], dtype=np.int32), raster['lengths']).reshape(raster['shape'])
    patches, charts, vertices = {}, {}, {}
    for plane in data['planes']:
        patch = plane['patch']
        i = patch['id']
        patches[i] = patch
        charts[i] = np.asarray(plane['chart'], dtype=float)
        vertices[i] = np.asarray(plane['vertices'], dtype=float).reshape(-1, 2)
    rectangles = [p['rectangle'] for p in patches.values() if p['eligible'] and p['rectangle'] is not None]
    return dict(image_size=data['image_size'], id_map=ids, patches=patches,
                charts=charts, vertices=vertices, rectangles=rectangles)


def build(cases, path=ARCHIVE):
    """Atomically publish a complete archive with reproducible ZIP member timestamps."""
    index = dict(schema=SCHEMA, generator_sha256=generator_hash(), numpy=np.__version__,
                 sample_stride_px=metrics.DOWNSAMPLE, frames={})
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
    try:
        with ZipFile(name, 'w') as bundle:
            for case in sorted(cases, key=lambda c: (c.floor, c.id)):
                key = f'{case.floor}/{case.id}'
                if key in index['frames']:
                    raise ValueError(f'Duplicate reference case {key}')
                inputs = _inputs(case)
                if inputs['image'] != case.frame['image_sha256'] or inputs['labels'] != case.frame['label_sha256']:
                    raise ValueError(f'Fixture hashes differ for {key}')
                geometry = metrics.truth(read(case.path('pose')), read(case.path('labels'))['faces'])
                payload = _encode(geometry)
                index['frames'][key] = dict(inputs=inputs, geometry_sha256=hashlib.sha256(payload).hexdigest())
                bundle.writestr(ZipInfo(key + '.json'), payload, compress_type=ZIP_DEFLATED)
            bundle.writestr(ZipInfo('index.json'), _json(index), compress_type=ZIP_DEFLATED)
        name.replace(path)
    finally:
        name.unlink(missing_ok=True)


def recorded(case, archive=ARCHIVE):
    """Load only for evaluation. Unbundled fixtures can still use the original projector."""
    key = f'{case.floor}/{case.id}'
    pose = read(case.path('pose'))
    if archive.is_file():
        try:
            with ZipFile(archive) as bundle:
                index = json.loads(bundle.read('index.json'))
                if index['schema'] != SCHEMA or index['generator_sha256'] != generator_hash():
                    raise ValueError('incompatible geometry version')
                record = index['frames'].get(key)
                if record is not None:
                    if record['inputs'] != _inputs(case):
                        raise ValueError('source image or geometry inputs changed')
                    payload = bundle.read(key + '.json')
                    if hashlib.sha256(payload).hexdigest() != record['geometry_sha256']:
                        raise ValueError('geometry checksum differs')
                    return pose, _decode(payload)
        except (BadZipFile, KeyError, ValueError) as exc:
            raise ValueError(f'{key}: invalid saved reference ({exc}). '
                             'Rebuild with python -m benchmark.surface_reference --rebuild') from exc
    return pose, metrics.truth(pose, read(case.path('labels'))['faces'])


def review_case(run, row):
    root = (run / row['source']).resolve().parent.parent
    index = read(root / 'index.json')
    frame = next(f for f in index['frames'] if f['id'] == row['id'])
    if (index.get('floor', 'smooth_stone') != row['floor'] or frame['image_sha256'] != row['sha256']
            or (root / frame['image']).resolve() != (run / row['source']).resolve()):
        raise ValueError(f"Stored prediction refers to another screenshot: {row['key']}")
    return Capture(row['floor'], root, frame)


def verify(cases, archive=ARCHIVE):
    with ZipFile(archive) as bundle:
        keys = json.loads(bundle.read('index.json'))['frames']
    if set(keys) != {f'{c.floor}/{c.id}' for c in cases}:
        raise ValueError('Reference release and fixture membership differ')
    for case in cases:
        recorded(case, archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='Regenerate all maps from the frozen local labels')
    args = parser.parse_args()
    from benchmark.normal_round import groups
    cohorts, _, _ = groups()
    cases = [c for group in cohorts.values() for c in group]
    if args.rebuild:
        build(cases)
    verify(cases)
    print(f'Verified {len(cases)} saved reference maps in {ARCHIVE}')
