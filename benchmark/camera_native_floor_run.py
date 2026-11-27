"""Run both source proposal arms on frozen incumbent abstentions, then export the selected camera."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from benchmark.camera_geocalib_infer import WEIGHT_SHA
from benchmark.camera_horizon_run import BASE
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, WORKTREES, digest, frames, read, save
from benchmark.camera_native_floor_search import solve

REVIEW = ROOT / 'benchmark/reviews/camera-native-floor-inverse-01'
CONTROL = ROOT / 'benchmark/reviews/camera-model-controls-01'
RUN = ROOT / 'runs/camera-native-floor-inverse-01'


def predict(entry, image, arm):
    seed = None
    if arm == 'geocalib':
        path = WORKTREES / 'camera-geocalib/runs/camera-geocalib-01/proposals' / (entry['id'] + '.json')
        seed = read(path)
        if seed['rgb_sha256'] != entry['rgb_sha256'] or seed['weights_sha256'] != WEIGHT_SHA:
            raise ValueError('Mismatched GeoCalib input')
    result = solve(image, seed)
    return dict(camera=result, rgb_sha256=entry['rgb_sha256'], arm=arm,
                weights_sha256=WEIGHT_SHA if seed else None)


def run(job):
    import cv2
    cv2.setNumThreads(1)
    entry, stage = job
    filename = entry['id'] + '.json'
    selected = RUN / 'metric' / filename
    if stage == 'predict':
        path = Path(entry['frame']) / 'rgb.png'
        if digest(path) != entry['rgb_sha256']:
            raise ValueError('Source RGB digest mismatch')
        with Image.open(path) as image:
            base = predict(entry, image, 'boundary')
            learned = predict(entry, image, 'geocalib')
        for data in (base, learned):
            save(RUN / data['arm'] / 'metric' / filename, data)
        choice = learned if base['camera']['P'] is None and learned['camera']['P'] is not None else base
        save(selected, choice)
    else:
        from benchmark.camera_horizon_replay import replay
        choice = read(selected)
        if choice['camera']['P'] is not None:
            build = Path(entry['dataset']) / entry['id']
            case = replay(entry, choice['camera'], BASE / build, RUN / 'builds' / build)
            save(RUN / 'scores' / filename, case)
    print(entry['id'], choice['arm'], choice['camera']['P'] is not None, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['predict', 'build'])
    parser.add_argument('split', choices=['train', 'validation', 'test'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    failures = {r['id'] for r in read(CONTROL / 'independent-matched-cases.json.gz') if not r['camera_usable']}
    jobs = [(e, args.stage) for e in frames(SOURCE, COHORTS) if e['split'] == args.split and e['id'] in failures]
    with ProcessPoolExecutor(args.workers) as workers:
        list(workers.map(run, jobs))
