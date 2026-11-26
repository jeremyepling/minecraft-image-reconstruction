"""Test whether switching among existing image-only cameras can recover metric scale."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_metrics import score
from benchmark.camera_round import digest, read, save
from benchmark.normal_round import groups
from engine.camera_grid import DEFAULT, METHODS, predict
from runtime.mode import mark_image_only


def evaluate(case):
    with Image.open(case.path('image')) as source:
        rgb = Image.fromarray(np.asarray(source.convert('RGB')))
    predictions = {name: predict(rgb, '26.3', name) for name in METHODS}
    pose = read(case.path('pose'))
    faces = read(case.path('labels'))['faces']
    scores = {}
    for name, prediction in predictions.items():
        row = score(prediction, pose, faces)
        scores[name] = dict(supported=prediction['scale_status'] == 'supported',
                            vertices=row['vertices'], within_2px=row['within_2px'])
    return dict(frame=f'{case.floor}/{case.id}', layout=case.frame['layout'],
                image_sha256=digest(case.path('image')), scores=scores)


def summarize(rows):
    baseline = [r['scores'][DEFAULT] for r in rows]
    any_scale = [any(m['supported'] for m in r['scores'].values()) for r in rows]
    best = [max(m['within_2px'] for m in r['scores'].values()) for r in rows]
    return dict(frames=len(rows), default_supported=sum(m['supported'] for m in baseline),
                any_method_supported=sum(any_scale), visible_corners=sum(m['vertices'] for m in baseline),
                default_within_2px=sum(m['within_2px'] for m in baseline), oracle_within_2px=sum(best),
                new_supported_frames=[r['frame'] for r,b,a in zip(rows,baseline,any_scale)
                                      if a and not b['supported']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    cohorts, _, hashes = groups()
    cases = [c for group in cohorts.values() for c in group]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=mark_image_only) as pool:
        rows = list(pool.map(evaluate, cases))
    development = {f'{c.floor}/{c.id}' for c in cohorts['development']}
    out = dict(fixture_sha256=hashes, methods=list(METHODS), default=DEFAULT,
               oracle='Select one camera per screenshot using its count of visible vertices within two pixels after compass and integer origin registration. Diagnostic only.',
               all=summarize(rows), development=summarize([r for r in rows if r['frame'] in development]),
               regression=summarize([r for r in rows if r['frame'] not in development]), frames=rows)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    save(args.out, out)
    print(out['all'])


if __name__ == '__main__':
    main()
