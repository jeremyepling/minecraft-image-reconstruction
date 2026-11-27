"""Run RGB-only floor fallback hypotheses, then export confirmed camera handoffs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

from PIL import Image

from benchmark.camera_horizon_floor import solve
from benchmark.camera_horizon_repeat import solve_repeat
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, digest, frames, read, save

CONTROL = ROOT.parent / 'full-expanded-controls'
BASE = CONTROL / 'runs/expanded-baseline-01'
REVIEW = ROOT / 'benchmark/reviews/camera-horizon-floor-01'
OUT = ROOT / 'runs/camera-horizon-floor-01'
STAGES = {
    'predict': ('boundary', 'free'), 'build': ('boundary', 'free'),
    'repeat-predict': ('repeat',), 'repeat-build': ('repeat',),
    'grid-predict': ('grid',), 'grid-build': ('grid',),
}


def run(task):
    entry, stage = task
    frame_id = entry['id']
    folder = OUT / frame_id
    rgb = Path(entry['frame']) / 'rgb.png'
    if digest(rgb) != entry['rgb_sha256']:
        raise ValueError(f'Source changed for {frame_id}')
    for mode in STAGES[stage]:
        filename = folder / f'{mode}.json'
        if stage.endswith('predict') and not filename.exists():
            with Image.open(rgb) as image:
                if mode in ('repeat', 'grid'):
                    camera = solve_repeat(image, use_boundary=mode=='repeat')
                else:
                    camera = solve(image, mode)
            save(filename, dict(rgb_sha256=entry['rgb_sha256'], camera=camera))
        cached = read(filename)
        if cached['rgb_sha256'] != entry['rgb_sha256']:
            raise ValueError(f'Camera cache source mismatch for {frame_id}')
        prediction = cached['camera']
        score = folder / f'{mode}-score.json'
        if stage.endswith('predict') or prediction['P'] is None or score.exists():
            continue
        # The two repeat arms can converge to the exact same registered camera.
        repeat_score = folder / 'repeat-score.json'
        if mode=='grid' and repeat_score.exists() and read(folder/'repeat.json')['camera']['P']==prediction['P']:
            case = read(repeat_score)
        else:
            from benchmark.camera_horizon_replay import replay
            case = replay(entry, prediction, BASE / entry['dataset'] / frame_id, folder / mode)
        save(score, case)
    return frame_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    cases = read(CONTROL / 'benchmark/reviews/expanded-baseline-01/cases.json')
    failed = {r['id'] for r in cases if not r['camera_usable']}
    entries = [e for e in frames(SOURCE, COHORTS) if e['id'] in failed]
    if len(entries) != 118:
        raise ValueError('Frozen set of ordinary abstentions differs')
    entries.sort(key=lambda e: (e['split']!='train', e['dataset'], e['id']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for frame_id in pool.map(run, [(entry, args.stage) for entry in entries[:args.limit]]):
            print(frame_id, flush=True)


if __name__=='__main__':
    main()
