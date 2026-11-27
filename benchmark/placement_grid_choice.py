"""Evaluate original versus accepted retry charts through ordinary typed exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import json
from pathlib import Path
import pickle
from unittest.mock import patch

from benchmark import expanded_fixed_handoffs as fixed
from benchmark import expanded_scores as grade
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, frames, placement_input, read, replay_source_fields, save
from benchmark.placement_grid_options import choices, distinct_options, private_charts
from engine import surface_model, visible_placement
from PIL import Image

CONTROL = ROOT.parent / 'full-expanded-controls/runs/expanded-baseline-01'
RUN = ROOT / 'runs/placement-grid-choice-01'
REVIEW = ROOT / 'benchmark/reviews/placement-grid-choice-01'


def fixed_input(entry):
    path = RUN / 'fixed' / (entry['id'] + '.pickle')
    if path.exists():
        with path.open('rb') as stream:
            return pickle.load(stream)
    inputs = fixed.inference_inputs(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    with temporary.open('wb') as stream:
        pickle.dump(inputs, stream)
    temporary.replace(path)
    return inputs


def expanded_build(entry, folder, saved, overrides, selected):
    path = Path(entry['frame']) / 'rgb.png'
    output = RUN / entry['dataset'] / entry['id'] / ('choice-' + '-'.join(map(str, selected)))
    output.mkdir(parents=True, exist_ok=True)
    ordinary = visible_placement.place
    seen = []
    with ExitStack() as stack:
        stack.enter_context(replay_source_fields(path, folder))

        def apply(image, version, parts, **options):
            if 'joint_foreground' in options:
                handoff = options['counts']
                if json.loads(json.dumps(handoff)) != saved['counts'] or json.loads(json.dumps(parts)) != saved['parts']:
                    save(output/'replay-input.json.gz', dict(parts=parts, counts=handoff))
                assert json.loads(json.dumps(handoff)) == saved['counts'], 'counter handoff differs'
                assert json.loads(json.dumps(parts)) == saved['parts'], 'surface inputs differ'
                seen.append(True)
                stack.enter_context(private_charts(handoff, overrides))
            return ordinary(image, version, parts, **options)

        stack.enter_context(patch.object(visible_placement, 'place', apply))
        surface_model.build(path, '26.3', output, through='placement')
    assert len(seen) == 1
    return grade.score_build(entry, output)


def compact(row):
    return {k: row[k] for k in ('pack_sha256', 'geometry', 'scores', 'translation')}


def process(task):
    entry, base = task
    ds, key = entry['dataset'], entry['id']
    path = RUN / 'cases' / ds / (key + '.json')
    if path.exists():
        return
    baseline = compact(base)
    record = dict(dataset=ds, id=key, split=entry['split'], layout=entry['layout'],
                  source_sha256=entry['rgb_sha256'], dataset_manifest_sha256=entry['dataset_manifest_sha256'],
                  reference_sha256=base.get('reference_sha256'), camera_usable=base['camera_usable'],
                  options=[], variants=[dict(selections=[[]], result=baseline)], baseline=baseline,
                  source=baseline, oracle=baseline, original=baseline)
    if not base['camera_usable']:
        record['reason'] = 'camera rejected before chart use'
        save(path, record)
        return
    if ds == grade.FIXED_DATASET:
        inp = fixed_input(entry)
        image, parts, counts, camera = inp.image, inp.parts, inp.counts, inp.camera
    else:
        folder = CONTROL / ds / key
        saved = placement_input(folder)
        parts, counts, camera = (saved[k] for k in ('parts', 'counts', 'camera'))
        with Image.open(Path(entry['frame']) / 'rgb.png') as src:
            image = src.convert('RGB')
    if camera is None:
        camera = visible_placement.seam_camera(image, '26.3', atlas_path=fixed.suite.ATLAS)
    options = choices(image, parts, counts)
    record['options'] = options
    if options:
        groups = distinct_options(image, parts, counts, camera, options)
        source = tuple(i for i, row in enumerate(options) if row['source'])
        variants = []
        for group in groups:
            selected = group[0]
            overrides = {options[i]['id']: options[i]['original'] for i in selected}
            if not selected:
                result = baseline
            elif ds == grade.FIXED_DATASET:
                with private_charts(counts, overrides):
                    replay = fixed.replay_placement(inp)
                result = compact(fixed.score_result(entry, replay)[0])
            else:
                result = compact(expanded_build(entry, folder, saved, overrides, selected))
            variants.append(dict(selections=group, result=result))
            if source in group:
                record['source'] = result
            if tuple(range(len(options))) in group:
                record['original'] = result
        record['variants'] = variants
        record['oracle'] = max(variants, key=lambda v: v['result']['geometry']['facade']['f1'])['result']
        record['reason'] = 'equal carrier and ground-corner signatures' if len(groups) == 1 else 'replayed distinct signatures'
    else:
        record['reason'] = 'no changed verified retry charts'
    save(path, record)
    if len(record['variants']) > 1:
        print(ds, key, len(options), len(record['variants']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args()
    baselines = {(r['dataset'], r['id']): r for r in read(CONTROL / 'scores/cases.json')}
    entries = [*fixed.fixed_frames(), *frames(SOURCE, COHORTS)]
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(process, (e, baselines[e['dataset'], e['id']])): e for e in entries
                if not args.ids or e['id'] in args.ids}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception:
                print('Failed:', jobs[job]['id'], flush=True)
                raise


if __name__ == '__main__':
    main()
