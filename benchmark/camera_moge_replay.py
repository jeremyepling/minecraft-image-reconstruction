"""Replay accepted cameras and score orientation of all frozen hypotheses independently."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark.camera_moge_surface import BASE,RUN,failures,read,save
from benchmark.camera_structure_run import orientation
from benchmark.camera_horizon_replay import replay


def evaluate(entry):
    base = BASE/entry['dataset']/entry['id']
    out = RUN/entry['dataset']/entry['id']
    proposal = read(out/'proposal.json')
    angle = orientation(proposal['linear'],read(Path(entry['frame'])/'camera.json'))
    save(out/'angle.json',angle)
    if proposal['camera']:
        (out/'build').mkdir(exist_ok=True)
        from contextlib import redirect_stderr,redirect_stdout
        with (out/'build.log').open('w',encoding='utf8') as log,redirect_stderr(log),redirect_stdout(log):
            case = replay(entry,proposal['camera'],base,out/'build')
        save(out/'case.json',case)
        return entry['id'],case['geometry']['strict']


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train','held'],required=True)
    args=parser.parse_args()
    entries=[e for e in failures() if (e['split']=='train')==(args.split=='train')]
    with ProcessPoolExecutor(max_workers=5) as pool:
        for result in pool.map(evaluate,entries):
            if result: print(result,flush=True)
