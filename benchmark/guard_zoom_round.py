"""Reproduce and compare protected wall boundaries and closer views of the same plane model."""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import numpy as np

from benchmark.afterpass_round import COHORTS, picture_table
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.ledge_join_round import index_candidate, score_dimensions, summary
from benchmark.mask_metrics import footprint, measure
from benchmark.normal_round import groups
from benchmark.plane_acceptance_round import BASE
from benchmark.split_round import load_predictions, save_and_score
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.surface_topology import opening_mask
from benchmark.wall_join_test import checked
from runtime.mode import image_only, mark_image_only


NAMES = {'current': 'Main default', 'guard': 'Protected edges', 'zoom': 'Closer views'}
MODULES = {'guard': 'benchmark.boundary_guard_round', 'zoom': 'benchmark.zoom_planes_round'}
REVIEW = ROOT / 'benchmark/reviews/guard-zoom-01'


def explain_zoom(capture, predictions, result, events):
    """Attribute measured changes to the actual emitted regions, after scoring."""
    gained = result['scopes']['zoom']['gained']
    scores = result['scores']
    delta = scores['zoom']['opening_filled'] - scores['current']['opening_filled']
    attribution = {'gained': [], 'opening_change': delta, 'additions_in_newly_covered_openings': {}}
    if not gained and not delta:
        return attribution
    _, ref = recorded(capture)
    zoom = predictions['zoom']
    if gained:
        children = {p for event in events if event['action']=='split' for p in event['children']}
        additions = {event['id'] for event in events if event['action']=='add'}
        _, matches = measure(zoom, ref, use_mask=True)
        for index, rid in matches:
            if rid in gained:
                pid = zoom[index]['id']
                action = 'split' if pid in children else 'addition' if pid in additions else 'retained region'
                attribution['gained'].append(dict(reference=rid, proposal=pid, action=action))
    if delta:
        free = opening_mask(ref['id_map'])
        for part in predictions['current']:
            free &= ~footprint(part, free.shape, ref['image_size'], True)
        new_ids = {event['id'] for event in events if event['action']=='add'}
        for part in zoom:
            if part['id'] in new_ids:
                hits = int(np.count_nonzero(free & footprint(part, free.shape, ref['image_size'], True)))
                if hits:
                    attribution['additions_in_newly_covered_openings'][part['id']] = hits
    return attribution


def grade(job):
    capture, row, original_sha, entries, out = job
    checked(capture.path('image'), row['sha256'])
    predictions = {'current': load_predictions(BASE, row['predictions'], original_sha)['planarity']}
    times = {'current': 0.0}
    for method, (directory, path, sha, record) in entries.items():
        models = load_predictions(directory, path, sha)
        if models['current'] != predictions['current']:
            raise ValueError(f"Different starting masks for {row['key']} in {method}")
        predictions[method] = models[method]
        times[method] = record['cached_processing_s'][method]
    result = save_and_score(capture, row, predictions, times, out, 'planarity')
    score_dimensions(capture, row, predictions, result, baseline_method='planarity')
    for method, (_, _, _, record) in entries.items():
        for field in ('scores', 'scopes', 'sizes', 'size_changes'):
            if result[field][method] != record[field][method]:
                raise ValueError(f"Independent {field} mismatch for {row['key']} in {method}")
    result['zoom_diagnostic'] = explain_zoom(capture, predictions, result, entries['zoom'][3]['events'])
    return result


def run_candidate(method, directory, workers, fields):
    command = [sys.executable, '-B', '-m', MODULES[method], '--out', str(directory), '--workers', str(workers)]
    if method == 'zoom' and fields is not None:
        command.extend(['--fields', str(fields)])
    subprocess.run(command, cwd=ROOT, check=True)


def raw_opportunities(zoom_rows):
    """Use scored reference IDs only after prediction to trace which earlier misses became visible."""
    meta = read(BASE / 'manifest.json')
    earlier = read(checked(BASE / 'recovered.json', meta['artifacts']['recovered.json']))
    previous = {(r['frame'], r['reference']): r for r in earlier if 'planarity' not in r['recovered_by']}
    trace = []
    for key, (_, _, _, row) in zoom_rows.items():
        for miss in row['raw_baseline_misses']:
            old = previous.pop((key, miss['patch']))
            trace.append(dict(frame=key, reference=miss['patch'], orientation=old['orientation'],
                              mobile_iou=old['earlier_mobilesam_iou'], whole_iou=miss['global_iou'],
                              crop_iou=miss['crop_iou'], finally_found=miss['recovered']))
    if previous or len(trace) != 85:
        raise ValueError('Different baseline miss roster')
    unseen = [r for r in trace if max(r['mobile_iou'], r['whole_iou']) < .5]
    return dict(previously_without_matching_outline=len(unseen),
                now_with_matching_crop=sum(r['crop_iou'] >= .5 for r in unseen),
                threshold=.5, misses=trace)


@image_only()
def run(out, workers, existing, fields):
    baseline = read(BASE / 'manifest.json')
    rows = read(checked(BASE / 'cases.json', baseline['artifacts']['cases.json']))
    checked(ARCHIVE, baseline['reference_sha256'])
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}': c for cohort in cohorts.values() for c in cohort}
    roster = {r['key']: r['sha256'] for r in rows}
    if len(rows) != 56 or roster.keys() != captures.keys() or fixtures != baseline['fixtures']:
        raise ValueError('Frozen screenshot roster differs')
    out.mkdir(parents=True, exist_ok=False)
    stamp = revision()
    candidates = {m: out / 'candidate-runs' / m for m in MODULES}
    if existing:
        for method, directory in candidates.items():
            shutil.copytree(existing[method], directory)
    else:
        with ThreadPoolExecutor(max_workers=len(MODULES)) as pool:
            futures = [pool.submit(run_candidate, m, directory, workers, fields) for m, directory in candidates.items()]
            for future in futures:
                future.result()

    indexed = {}
    records = {}
    for method, directory in candidates.items():
        meta, indexed[method] = index_candidate(directory, fixtures, baseline['reference_sha256'], roster)
        records[method] = dict(run_id=meta['run_id'], manifest=(directory / 'manifest.json').relative_to(out).as_posix(),
                               manifest_sha256=digest(directory / 'manifest.json'), code=meta['code'])
    jobs = [(captures[r['key']], r, baseline['artifacts'][r['predictions']],
             {m: index[r['key']] for m, index in indexed.items()}, out) for r in rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        results = list(pool.map(grade, jobs))

    table = {'all': summary(results, NAMES)}
    for cohort in (*COHORTS, *sorted({r['layout'] for r in results})):
        table[cohort] = summary([r for r in results if cohort in (r['cohort'], r['layout'])], NAMES)
    save(out / 'cases.json', results)
    save(out / 'scores.json', table)
    save(out / 'raw-opportunities.json', raw_opportunities(indexed['zoom']))
    (out / 'pictures.md').write_text(picture_table(results, NAMES, NAMES), encoding='utf8')
    from benchmark.guard_zoom_sheet import render
    render(out, results, captures, table, candidates['zoom'])
    save(out / 'manifest.json', dict(
        run_id=out.name, code=stamp, version=suite['version'], protocol=baseline['protocol'],
        capture=baseline['capture'], fixtures=fixtures, frames=baseline['frames'],
        splits=baseline['splits'], reference_sha256=digest(ARCHIVE),
        methods=NAMES, baseline='plane-recovery', counter='masked-repeat',
        comparison='Each candidate starts from the same 513-surface prediction. Both use the frozen Block Counter.',
        baseline_source='benchmark/reviews/afterpass-models-01/manifest.json',
        baseline_manifest_sha256=digest(BASE / 'manifest.json'), candidates=records,
        score_file='scores.json', per_picture='pictures.md', sheet='comparison.jpg',
        reproduce=r'.\Invoke-SurfaceRound.ps1 -Round guard-zoom',
        artifacts={p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    for method, score in table['all'].items():
        print(f"{NAMES[method]}: {score['matched']}/{score['eligible']} surfaces, {score['sizes']['correct']} correct sizes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs' / f'guard-zoom-{uuid4().hex[:12]}')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--guard-run', type=Path)
    parser.add_argument('--zoom-run', type=Path)
    parser.add_argument('--fields', type=Path)
    args = parser.parse_args()
    if bool(args.guard_run) != bool(args.zoom_run) or args.workers < 1:
        parser.error('Supply both finished candidate runs or neither, and at least one worker.')
    sources = {'guard': args.guard_run.resolve(), 'zoom': args.zoom_run.resolve()} if args.guard_run else None
    fields = args.fields or REVIEW / 'candidate-runs/zoom/crop-fields'
    run(args.out.resolve(), args.workers, sources, fields.resolve())
