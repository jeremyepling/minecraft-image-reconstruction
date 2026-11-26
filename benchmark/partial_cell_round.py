"""Measure final-cell rigidity on the frozen 56-frame sky-gap export."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.camera_round import read, digest
from benchmark.mask_blocks import SETTINGS, _blocks, _carriers, _footprints, _score, _write_read, registered
from engine.camera_grid import predict
from engine.masked_count import count
from engine.surface_sample import sample_surface
from runtime.mode import mark_image_only


# Primary challenger and a predeclared sensitivity check. The strict run verifies
# that the frozen baseline is reproduced exactly, including camera registration.
THRESHOLDS = {'strict': 1.0, 'three_quarters': 0.75, 'half': 0.5}


def run(job):
    row, suite, frames, out, hashes, base_coords, target_coords = job
    key = row['key']
    path = frames / ('legacy-' + key.replace('/', '-'))
    pred = suite / row['predictions']
    assert digest(path / 'rgb.png') == row['sha256'] == hashes['rgb']
    assert digest(pred) == hashes['mask']
    parts = json.loads(gzip.decompress(pred.read_bytes()))['gap']
    with Image.open(path / 'rgb.png') as source:
        image = source.convert('RGB')
    camera = predict(image, '26.3', SETTINGS['camera'])
    P = np.array(camera['P']) if camera['P'] is not None else None
    masks = _footprints(parts, image.size, True)
    inputs = []
    for part in parts:
        sample = sample_surface(image, part)
        ns = count(sample)['counts']
        inputs.append(dict(id=part['id'], support=part['support'], quad=sample['quad'] if sample else None,
                           pitch=dict(counts=ns if all(n is not None for n in ns) else None)))
    carriers = _carriers(image, inputs, masks, P)
    reports = {}
    cache = {}
    base = set(map(tuple, base_coords))
    target = set(map(tuple, target_coords))
    assert digest(path/'camera.json') == hashes['pose'] and digest(path/'labels.json') == hashes['truth']
    T = np.eye(4) if P is None else registration(P, read(path/'camera.json'))
    for name, threshold in THRESHOLDS.items():
        blocks, status, details = _blocks(masks, carriers, P, cache, minimum_coverage=threshold)
        filename = out / name / 'packs' / (key + '.zip')
        filename.parent.mkdir(exist_ok=True, parents=True)
        emitted = registered(_write_read(filename, blocks), T)
        assert name != 'strict' or emitted == base, key
        delta = dict(gained=len((emitted-base)&target), lost=len((base-emitted)&target),
                     extra_added=len(emitted-base-target), extra_removed=len(base-emitted-target))
        reports[name] = dict(coords=sorted(emitted), score=_score(emitted,target),
                             delta=delta, stages=status, details=details, pack_sha256=digest(filename))
    return dict(key=key, split=row['cohort'], hashes=hashes, camera=camera['scale_status'], methods=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('suite','frames','baseline','out'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = read(args.baseline/'manifest.json')
    coords,targets = [read(args.baseline/file) for file in ('coords.json','targets.json')]
    cases = read(args.suite/'cases.json')
    assert set(coords) == set(targets) == {row['key'] for row in cases}
    jobs = [(row,args.suite,args.frames,args.out,manifest['inputs'][row['key']],coords[row['key']],targets[row['key']])
            for row in cases]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=mark_image_only) as pool:
        rows = list(pool.map(run,jobs))
    totals = {}
    for method in THRESHOLDS:
        reports = [row['methods'][method] for row in rows]
        totals[method] = {f: dict(sum((Counter(r[f]) for r in reports), Counter())) for f in ('score','delta','stages')}
        totals[method]['producing'] = sum(bool(r['coords']) for r in reports)
        coords = {row['key']:row['methods'][method].pop('coords') for row in rows}
        (args.out/method/'coords.json').write_text(json.dumps(coords))
        if method == 'three_quarters':
            (args.out/'coords.json').write_text(json.dumps(coords))
    report = dict(frames=len(rows), thresholds=THRESHOLDS, primary='three_quarters',
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  suite_sha256=digest(args.suite/'manifest.json'), baseline_sha256=digest(args.baseline/'coords.json'),
                  totals=totals, results=rows)
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(totals,indent=2))


if __name__ == '__main__':
    main()
