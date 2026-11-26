"""Replay a single camera method on the fixed, paired exterior regression set."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform

import numpy as np
from PIL import __version__ as pillow_version

from benchmark import camera_metrics, ratio_metrics
from benchmark.camera_round import ROOT, digest, infer, read, revision, save
from benchmark.exterior_set import RELEASE, load_suite
from engine.camera_grid import DEFAULT, METHODS
from runtime.mode import image_only, mark_image_only


PROTOCOL = 'camera-floor-ratios-v1'
REFERENCE = ROOT / 'benchmark/reviews/ground-floor-01'


def aggregate(rows):
    result = camera_metrics.summarize(rows)
    result['axes_frames'] = sum(r['axis_mae_deg'] is not None for r in rows)
    for kind in ('endpoint_diagnostic', 'automatic_charts'):
        total = sum(r[kind]['total'] for r in rows)
        measured = sum(r[kind]['measured'] for r in rows)
        good = sum(1 for r in rows for c in r[kind]['cases']
                   if c['abs_error_pct'] is not None and c['abs_error_pct'] <= 10)
        errors = []
        for layout in sorted({r['layout'] for r in rows}):
            values = [r[kind]['mean_abs_error_pct'] for r in rows
                      if r['layout'] == layout and r[kind]['mean_abs_error_pct'] is not None]
            if values:
                errors.append(np.mean(values))
        result[kind] = dict(total=total, measured=measured, abstained=total-measured,
                            within_10pct=good, returned_frames=sum(r[kind]['measured'] > 0 for r in rows),
                            conditional_layout_mean_abs_error_pct=float(np.mean(errors)) if errors else None)
    return result


def score_capture(case, prediction):
    pose = read(case.path('pose'))
    faces = read(case.path('labels'))['faces']
    return dict(id=case.id, floor=case.floor, layout=case.frame['layout'], split=case.frame['split'],
                condition=case.frame['condition'], metric=prediction['scale_status'] == 'supported',
                **camera_metrics.score(prediction, pose, faces), **ratio_metrics.score(prediction, pose, faces))


def relative(path, base):
    return os.path.relpath(path, base).replace('\\', '/')


def compare(reference, scores, suite, split, out):
    """Keep coverage and conditional error separate in score changes."""
    if not reference.is_file() or reference.parent.resolve() == out.resolve():
        return None
    baseline = read(reference)
    if baseline['suite_sha256'] != digest(RELEASE) or baseline['protocol'] != PROTOCOL:
        raise ValueError('Reference run belongs to a different suite or scoring protocol')
    deltas = {}
    for floor, groups in scores.items():
        for key, current in groups.items():
            if key == 'all' and split != 'all':
                continue
            prior = baseline['scores'][floor][key]
            result = {k: current[k] - prior[k] for k in ('metric_images', 'within_2px', 'axes_frames')}
            for k in ('endpoint_diagnostic', 'automatic_charts'):
                result[k] = {j: current[k][j] - prior[k][j] for j in ('measured','within_10pct')}
                err = 'conditional_layout_mean_abs_error_pct'
                a, b = current[k][err], prior[k][err]
                result[k][err] = a - b if a is not None and b is not None else None
            deltas.setdefault(floor, {})[key] = result
    return dict(reference=relative(reference, out), reference_method=baseline['methods'][0],
                suite=suite['id'], current_minus_reference=deltas)


@image_only()
def run(out, method=DEFAULT, floor=None, split='all', workers=3):
    suite, pairs = load_suite()
    floors = [floor] if floor else list(suite['conditions'])
    chosen = [c for pair in pairs.values() for f, c in pair.items()
              if f in floors and (split == 'all' or c.frame['split'] == split)]
    if not chosen or workers < 1:
        raise ValueError('Requested selection has no frames or workers')
    out.mkdir(parents=True, exist_ok=False)
    def prediction_path(c):
        return out / 'predictions' / c.floor / (c.id + '.json')
    for f in floors:
        (out / 'predictions' / f).mkdir(parents=True)
    jobs = [(c.path('image'), suite['version'], method, prediction_path(c)) for c in chosen]
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        list(pool.map(infer, jobs))
    rows, examples = [], {}
    cases_dir = out / 'scores'
    cases_dir.mkdir()
    for case in chosen:
        prediction = read(prediction_path(case))
        row = score_capture(case, prediction)
        save(cases_dir / (case.floor + '--' + case.id + '.json'), row)
        rows.append(row)
        example = examples.setdefault(case.id, dict(id=case.id, split=case.frame['split'],
                                      images={}, predictions={}, metrics={}))
        example['images'][case.floor] = case.path('image')
        example['predictions'][case.floor] = prediction
        example['metrics'][case.floor] = row
    summaries = {}
    for f in floors:
        condition = [r for r in rows if r['floor'] == f]
        summaries[f] = {'all': aggregate(condition)}
        for group in sorted({r['split'] for r in condition}):
            summaries[f][group] = aggregate([r for r in condition if r['split'] == group])
    artifacts = dict(predictions='predictions/', per_image='scores/', visual_feedback='feedback.jsonl')
    (out / artifacts['visual_feedback']).touch()
    if len(floors) == 2:
        from benchmark.ground_sheet import sheet
        regions = sheet(out, list(examples.values()))
        save(out / 'regions.json', regions)
        artifacts.update(sheet='comparison.png', regions='regions.json')
    comparison = compare(REFERENCE / 'manifest.json', summaries, suite, split, out)
    if comparison is not None:
        save(out / 'delta.json', comparison)
        artifacts['reference_delta'] = 'delta.json'
    indexes = {f: relative((RELEASE.parent / suite['conditions'][f]['index']).resolve(), out) for f in floors}
    manifest = dict(schema=2, run_id=out.name, suite=suite['id'], suite_sha256=digest(RELEASE),
                    suite_path=relative(RELEASE, out), protocol=PROTOCOL, version=suite['version'],
                    stage='camera', methods=[method], settings=prediction['settings'], split=split,
                    capture_conditions=suite['capture_conditions'],
                    cohort_status=suite['cohort_status'], revision=revision(), frames=len(rows),
                    fixture=dict(indexes=indexes), environment=dict(python=platform.python_version(),
                    numpy=np.__version__, pillow=pillow_version), artifacts=artifacts, scores=summaries)
    save(out / 'manifest.json', manifest)
    for f in floors:
        s = summaries[f]['all']
        ratio = s['endpoint_diagnostic']
        print(f"{f}: {s['metric_images']}/{s['images']} metric cameras, "
              f"{ratio['measured']}/{ratio['total']} supplied-endpoint ratios")
    print(out.resolve() / 'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default=DEFAULT)
    parser.add_argument('--floor', choices=['smooth_stone', 'light_gray_concrete'])
    parser.add_argument('--split', choices=['dev', 'holdout', 'all'], default='all')
    parser.add_argument('--out', type=Path, help='New directory, whose name becomes the run ID')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run(args.out or ROOT / 'runs/exterior' / stamp, args.method, args.floor, args.split, args.workers)


if __name__ == '__main__':
    main()
