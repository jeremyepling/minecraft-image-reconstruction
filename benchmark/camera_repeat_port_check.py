"""Verify the guarded repeat implementation against every frozen fallback camera."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MCIR_IMAGE_ONLY='1')

from PIL import Image
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, WORKTREES, digest, frames, read, save
from engine.camera_repeat_floor import predict

PREVIOUS = WORKTREES / 'camera-horizon-floor/runs/camera-horizon-floor-01'
EXPANDED = WORKTREES / 'camera-geocalib/runs/camera-geocalib-01/matched-control/metric'
OUT = ROOT / 'runs/camera-repeat-expanded-port-check'


def expected(entry):
    camera = read(PREVIOUS / entry['id'] / 'repeat.json')['camera']
    if camera['P'] is None:
        previous = read(EXPANDED / (entry['id'] + '.json'))
        if previous['rgb_sha256'] != entry['rgb_sha256']:
            raise ValueError('Expanded source checksum changed')
        camera = previous['camera']
    return camera


def run(entry):
    source = Path(entry['frame']) / 'rgb.png'
    if digest(source) != entry['rgb_sha256']:
        raise ValueError('Source changed: ' + entry['id'])
    before = expected(entry)
    with Image.open(source) as image:
        after = predict(image)
    after = json.loads(json.dumps(after, allow_nan=False))
    case = dict(id=entry['id'], dataset=entry['dataset'], split=entry['split'],
                source_sha256=digest(source), identical=before == after,
                identical_P=before['P'] == after['P'], camera=after)
    save(OUT / (entry['id'] + '.json'), case)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--summarize', action='store_true', help='Verify saved port predictions without rerunning inference.')
    args = parser.parse_args()
    entries = [e for e in frames(SOURCE, COHORTS) if (PREVIOUS / e['id'] / 'repeat.json').is_file()]
    if len(entries) != 118:
        raise ValueError('Expected 118 frozen ordinary camera failures')
    cases = []
    if args.summarize:
        for entry in entries:
            path = OUT / (entry['id'] + '.json')
            row = read(path)
            before = expected(entry)
            row['identical'] = before == row['camera']
            row['identical_P'] = before['P'] == row['camera']['P']
            save(path, row)
            cases.append(row)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            for row in pool.map(run, entries):
                cases.append(row)
                print(row['id'], row['identical'], flush=True)
    result = dict(tested=len(cases), identical=sum(r['identical'] for r in cases),
                  identical_camera=sum(r['identical_P'] for r in cases),
                  supported=sum(r['camera']['P'] is not None for r in cases))
    save(OUT / 'summary.json', result)
    print(result, flush=True)
    if result['identical'] != len(cases):
        raise ValueError('Guarded repeat port changed a frozen fallback')


if __name__ == '__main__':
    main()
