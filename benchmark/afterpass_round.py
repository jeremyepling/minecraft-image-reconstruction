"""Regrade frozen second-pass predictions and run the ordinary Block Counter on every outline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from uuid import uuid4

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.ledge_join_round import BASE, index_candidate, score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL
from benchmark.normal_round import groups
from benchmark.recovery_overview import checked
from benchmark.split_round import load_predictions, save_and_score
from benchmark.surface_reference import ARCHIVE
from runtime.mode import image_only, mark_image_only


NAMES = {'current':'Current Surface Finder', 'planarity':'Pixel-wise Planarity', 'ect':'ECT'}
ARCHIVED = ROOT/'benchmark/reviews/afterpass-models-01'
COHORTS = ('development', 'previously_reviewed_regression')
SELECTION = ('Freeze each model configuration on the 20 development views. Evaluate both against the '
             'same ordinary Surface Finder on the six previously reviewed buildings. Favor gains in matched '
             'surfaces and surface F1 on both splits, preserve openings and check both block dimensions. '
             'Record raw opportunities separately from recovery by the automatic pipeline.')


def grade(job):
    case, row, control_sha, files, out = job
    checked(case.path('image'), row['sha256'])
    predictions = {'current': load_predictions(BASE, row['predictions'], control_sha)['single-v2']}
    times = {'current':0.}
    for method, (directory, file, sha, record) in files.items():
        predictions[method] = load_predictions(directory, file, sha)
        times[method] = record['cached_processing_s']
    result = save_and_score(case, row, predictions, times, out, 'single-v2')
    score_dimensions(case, row, predictions, result)
    return result


def picture_table(rows, methods):
    lines = ['# Surface and block counts by screenshot', '',
             'An R number identifies a recorded visible surface in that picture. Each candidate starts from the same default. '
             'Both dimensions must be correct for a block-count match.', '',
             '| View | Method | Found / scored | Gained | Lost | Extra guesses | Sizes correct |',
             '| --- | --- | ---: | --- | --- | ---: | ---: |']
    for row in sorted(rows, key=lambda r: (r['layout'], r['key'])):
        for m in methods:
            scope = row['scopes'][m]
            stats = row['scores'][m]
            ids = [' '.join(f'R{i}' for i in scope[k]) or '—' for k in ('gained','lost')]
            cells = [row['key'], NAMES[m], f'{stats["matched"]} / {stats["eligible"]}', *ids,
                     scope['unpaired'], row['sizes'][m]['correct']]
            lines.append('| '+' | '.join(map(str,cells))+' |')
    return '\n'.join(lines)+'\n'


def recovery_table(rows, methods):
    """Which baseline misses were recovered by each automatic method, including overlap across methods."""
    audit = ROOT/'benchmark/reviews/afterpass-headroom-01'
    manifest = read(audit/'manifest.json')
    checked(audit/'per-miss.json', manifest['artifacts']['per-miss.json'])
    by_key = {r['key']:r for r in rows}
    results = []
    for miss in read(audit/'per-miss.json'):
        row = by_key[miss['frame']]
        rid = miss['reference']
        if rid in row['scopes']['current']['matched']:
            raise ValueError('Historical miss does not match replayed baseline')
        results.append(dict(frame=row['key'], layout=row['layout'], reference=rid,
                            orientation=miss['orientation'], earlier_mobilesam_iou=miss['raw'],
                            recovered_by=[m for m in methods[1:] if rid in row['scopes'][m]['gained']]))
    baseline_misses = sum(r['scores']['current']['eligible']-r['scores']['current']['matched'] for r in rows)
    if len({(v['frame'],v['reference']) for v in results}) != baseline_misses:
        raise ValueError('Historical miss roster differs')
    return results


@image_only()
def run(out, candidate_dirs, workers):
    from benchmark.afterpass_sheet import render
    baseline = read(BASE/'manifest.json')
    checked(ARCHIVE, baseline['reference_sha256'])
    checked(BASE/'cases.json', baseline['artifacts']['cases.json'])
    rows = read(BASE/'cases.json')
    groups_by_split, suite, fixtures = groups()
    cases = {f'{c.floor}/{c.id}':c for cohort in groups_by_split.values() for c in cohort}
    roster = {r['key']:r['sha256'] for r in rows}
    if fixtures != baseline['fixtures'] or len(rows) != 56 or roster.keys() != cases.keys():
        raise ValueError('Frozen screenshot suite differs')
    manifests, indexed = {}, {}
    for m, directory in candidate_dirs.items():
        meta, indexed[m] = index_candidate(directory, fixtures, digest(ARCHIVE), roster)
        manifests[m] = {'manifest_sha256':digest(directory/'manifest.json'), 'method':meta.get('method', m)}
    out.mkdir(parents=True, exist_ok=False)
    methods = ('current', *candidate_dirs)
    jobs = [(cases[r['key']], r, baseline['artifacts'][r['predictions']],
             {m:indexed[m][r['key']] for m in candidate_dirs}, out) for r in rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        rows = list(pool.map(grade, jobs))
    table = {'all':summary(rows,methods)}
    for group in (*COHORTS,*sorted({r['layout'] for r in rows})):
        table[group] = summary([r for r in rows if group in (r['cohort'],r['layout'])], methods)
    save(out/'cases.json', rows)
    save(out/'scores.json', table)
    (out/'pictures.md').write_text(picture_table(rows,methods), encoding='utf8')
    save(out/'recovered.json', recovery_table(rows, methods))
    render(out, rows, cases, table, fields={m:d/'fields' for m,d in candidate_dirs.items()})
    files = [p for p in out.rglob('*') if p.is_file()]
    save(out/'manifest.json', dict(run_id=out.name, code=revision(), version=suite['version'], capture=baseline['capture'],
         protocol=PROTOCOL, selection=SELECTION, splits=baseline['splits'], fixtures=fixtures,
         reference_sha256=digest(ARCHIVE), methods={m:NAMES[m] for m in methods},
         baseline='wall-recovery-single-v2', baseline_manifest_sha256=digest(BASE/'manifest.json'),
         candidates=manifests, frames=[{k:r[k] for k in ('key','sha256','cohort','layout')} for r in rows],
         artifacts={p.relative_to(out).as_posix():digest(p) for p in files},
         reproduce=r'.\Invoke-SurfaceRound.ps1 -Round afterpass'))
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'afterpass-{uuid4().hex[:12]}')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--planarity',type=Path,default=ARCHIVED/'candidate-runs/planarity')
    parser.add_argument('--ect',type=Path,default=ARCHIVED/'candidate-runs/ect')
    args = parser.parse_args()
    run(args.out.resolve(), {m:getattr(args,m).resolve() for m in ('planarity','ect')}, args.workers)
