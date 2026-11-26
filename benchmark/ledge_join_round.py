"""Compare two independent Surface Finder refinements with the frozen single-outline default."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import PROTOCOL, footprint
from benchmark.normal_round import groups
from benchmark.recovery_overview import checked
from benchmark.split_count_check import count_frame
from benchmark.split_round import load_predictions, pooled, save_and_score
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only, mark_image_only


BASE = ROOT/'benchmark/reviews/single-wall-depth-01'
METHODS = ('current', 'ledge', 'join')
NAMES = dict(zip(METHODS, ('Current Surface Finder', 'Ledge cuts', 'Wall-piece joins')))
MODULES = {'ledge': 'benchmark.ledge_cut_test', 'join': 'benchmark.wall_join_test'}
RULE = ('Compare each refinement to the same baseline. Prefer more matched surfaces on development '
        'and previously reviewed regression, with no decrease in surface F1 or correct block dimensions '
        'and no increase in opening fill. Examine exact reference IDs gained and lost and false cuts.')


def index_candidate(directory, fixtures, ref_sha, expected):
    manifest = read(directory/'manifest.json')
    if manifest['fixtures'] != fixtures or manifest['reference_sha256'] != ref_sha:
        raise ValueError(f'Fixture mismatch in {directory}')
    checked(directory/'cases.json', manifest['artifacts']['cases.json'])
    frames = read(directory/'cases.json')
    if len(frames) != len(expected) or {r['key']: r['sha256'] for r in frames} != expected:
        raise ValueError(f'Screenshot roster mismatch in {directory}')
    indexed = {}
    for row in frames:
        name = row['predictions']
        sha = row.get('prediction_sha256') or manifest['artifacts'][name]
        indexed[row['key']] = (directory, name, sha, row)
    return manifest, indexed


def audit_edits(capture, row, predictions, events, evidence_sha):
    """Use the recorded geometry only after prediction, to inspect the proposed cuts and joins."""
    _, ref = recorded(capture)
    ids = ref['id_map']
    counts = Counter()
    original = {}
    for method in METHODS:
        masks = {p['id']:footprint(p, ids.shape, ref['image_size'], True) for p in predictions[method]}
        coverage = np.logical_or.reduce(list(masks.values()), axis=0) if masks else np.zeros(ids.shape, bool)
        if method == 'current':
            baseline = coverage
            original = masks
        elif not np.array_equal(coverage, baseline):
            raise ValueError(f'Visible support changed in {row["key"]}/{method}')
    reference_file = load_predictions(BASE,row['boundary_evidence'],evidence_sha)
    labels = {(v['parent'],v['proposal']):v for v in reference_file['reference_diagnostics']}
    for cut in events['ledge']['added_cuts']:
        label = labels[(cut['parent'],cut['proposal'])]
        counts['cuts'] += 1
        counts['cuts_'+label['kind'].replace(' ','_')] += 1
        counts['cuts_flush_material_seam'] += int(label['material_seam'])
    for join in events['join']['joins']:
        dominant = []
        for name in join['parts']:
            hits = np.bincount(ids[original[name]])
            rid = int(np.argmax(hits))
            dominant.append(rid if rid and hits[rid]*2 > hits.sum() else None)
        kind = ('uncertain' if None in dominant else 'same_surface' if len(set(dominant)) == 1
                else 'different_surfaces')
        counts['joins'] += 1
        counts['joins_'+kind] += 1
    return dict(counts)


def evaluate(job):
    capture, row, base_sha, evidence_sha, candidates, out = job
    checked(capture.path('image'), row['sha256'])
    predictions = {'current': load_predictions(BASE, row['predictions'], base_sha)['single-v2']}
    times = {'current': 0.0}
    events = {}
    for method, (directory, name, sha, record) in candidates.items():
        predictions[method] = load_predictions(directory, name, sha)
        times[method] = record['cached_processing_s']
        events[method] = {k: record[k] for k in ('added_cuts', 'joins') if k in record}
    result = save_and_score(capture, row, predictions, times, out, 'single-v2')
    for method in METHODS[1:]:
        for key in ('opening_filled', 'background_claimed', 'unclaimed_pixels', 'gaps_half_filled'):
            if result['scores'][method][key] != result['scores']['current'][key]:
                raise ValueError(f'Foreground coverage changed in {row["key"]}/{method}/{key}')
    score_dimensions(capture, row, predictions, result)
    result['events'] = events
    result['edit_diagnostics'] = audit_edits(capture,row,predictions,events,evidence_sha)
    return result


def score_dimensions(capture, row, predictions, result, baseline_method='single-v2'):
    if result['scopes']['current']['matched'] != row['scopes'][baseline_method]['matched']:
        raise ValueError('Baseline matches changed')
    with Image.open(capture.path('image')) as source:
        sizes, _ = count_frame(capture, source.convert('RGB'), result, predictions, tuple(predictions))
    result['sizes'] = {m: {k: sizes[m]['totals'][k] for k in ('correct','incorrect','missing','unmatched_returned')}
                       for m in predictions}
    if result['sizes']['current'] != row['sizes'][baseline_method]:
        raise ValueError('Baseline block counts changed')
    result['size_changes'] = {m: sizes[m]['comparisons'] for m in predictions}


def summary(rows, methods=METHODS):
    table = {}
    for m in methods:
        counts = Counter()
        for row in rows:
            counts.update(row['sizes'][m])
        table[m] = dict(pooled(rows, m), sizes=dict(counts),
                        cached_extra_seconds=sum(row['cached_processing_s'][m] for row in rows))
    return table


def picture_table(rows):
    lines = ['# Surface Finder results by picture', '',
             'Each score is a count of matched visible surfaces. Gained and lost IDs refer to this picture. '
             'The last column counts surfaces whose width and height in blocks were both correct.', '',
             '| Picture | Scored | Baseline | Ledge cuts | Ledge gained / lost | Wall joins | Join gained / lost | Both dimensions right (base / cut / join) |',
             '| --- | ---: | ---: | ---: | --- | ---: | --- | --- |']
    for row in sorted(rows, key=lambda r: (r['layout'], r['key'])):
        s = row['scores']
        cells = [row['key'], s['current']['eligible'], s['current']['matched']]
        for method in METHODS[1:]:
            scope = row['scopes'][method]
            changes = [' '.join(f'R{i}' for i in scope[k]) or '—' for k in ('gained','lost')]
            cells.extend((s[method]['matched'], ' / '.join(changes)))
        cells.append(' / '.join(str(row['sizes'][m]['correct']) for m in METHODS))
        lines.append('| '+' | '.join(map(str,cells))+' |')
    return '\n'.join(lines)+'\n'


def predict(method, out):
    subprocess.run([sys.executable, '-B', '-m', MODULES[method], '--source-root', str(ROOT),
                    '--out', str(out)], check=True, cwd=ROOT)


@image_only()
def run(out, workers, candidates):
    from benchmark.ledge_join_sheet import render
    baseline = read(BASE/'manifest.json')
    hashes = baseline['artifacts']
    checked(BASE/'cases.json', hashes['cases.json'])
    checked(ARCHIVE, baseline['reference_sha256'])
    rows = read(BASE/'cases.json')
    cohorts, _, fixtures = groups()
    captures = {f'{c.floor}/{c.id}': c for group in cohorts.values() for c in group}
    roster = {r['key']: r['sha256'] for r in rows}
    if len(rows) != 56 or captures.keys() != roster.keys() or fixtures != baseline['fixtures']:
        raise ValueError('Frozen screenshot suite differs')
    out.mkdir(parents=True, exist_ok=False)
    directories = {m: out/'candidate-runs'/m for m in MODULES}
    if candidates:
        for m, directory in candidates.items():
            shutil.copytree(directory, directories[m])
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(predict,m,directory) for m,directory in directories.items()]
            for f in futures:
                f.result()
    indexed = {}
    sources = {}
    for method, directory in directories.items():
        _, indexed[method] = index_candidate(directory, fixtures, baseline['reference_sha256'], roster)
        sources[(directory/'manifest.json').relative_to(out).as_posix()] = digest(directory/'manifest.json')
    jobs = [(captures[r['key']], r, hashes[r['predictions']], hashes[r['boundary_evidence']],
             {m: indexed[m][r['key']] for m in MODULES}, out)
            for r in rows]
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        rows = list(pool.map(evaluate, jobs))
    table = {'all': summary(rows)}
    for group in ('development','previously_reviewed_regression',*sorted({r['layout'] for r in rows})):
        table[group] = summary([r for r in rows if group in (r['layout'],r['cohort'])])
    save(out/'scores.json', table)
    save(out/'cases.json', rows)
    (out/'pictures.md').write_text(picture_table(rows), encoding='utf8')
    render(out, rows, captures, table)
    files = [p for p in out.rglob('*') if p.is_file()]
    save(out/'manifest.json', dict(run_id=out.name, code=revision(), methods=NAMES, protocol=PROTOCOL,
         selection=RULE, version=baseline['version'], fixtures=fixtures, reference_sha256=digest(ARCHIVE),
         baseline_id='wall-recovery-single-v2', baseline_manifest_sha256=digest(BASE/'manifest.json'),
         baseline_models={k:baseline[k] for k in ('models','line_models','mobile_model')},
         capture=baseline['capture'], splits=baseline['splits'], source_manifests=sources,
         timing='Extra cached refinement time per picture, summed. Baseline adds zero. Shared model inference reused.',
         edit_diagnostics=dict(sum((Counter(r['edit_diagnostics']) for r in rows),Counter())),
         frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in rows],
         artifacts={p.relative_to(out).as_posix():digest(p) for p in files},
         reproduce=r'.\Invoke-SurfaceRound.ps1 -Round ledge-join'))
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs'/f'ledge-join-{uuid4().hex[:12]}')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--ledge', type=Path)
    parser.add_argument('--join', type=Path)
    args = parser.parse_args()
    if bool(args.ledge) != bool(args.join):
        parser.error('Supply both private candidate directories, or omit both to regenerate predictions.')
    run(args.out.resolve(), args.workers, {m:getattr(args,m).resolve() for m in MODULES} if args.ledge else None)
