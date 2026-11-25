"""Test the sequential interaction of frozen junction cuts followed by underside recovery."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import reference_planes
from benchmark.mask_metrics import measure
from benchmark.normal_fields import cached_fields
from benchmark.normal_round import groups
from benchmark.overhang_matches import score_parts
from benchmark.perimeter_round import downstream_counts
from benchmark.split_round import load_predictions
from benchmark.split_tiny_lines import ARCHIVE as TINY
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import ARCHIVE, recorded
from engine.surface_strip_split import refine
from runtime.mode import image_only


JUNCTION = 'junction-cuts'
UNDERSIDE = 'wall-and-underside'
BOTH = 'junction-then-underside'
CONTROL = (JUNCTION, UNDERSIDE)
TOTALS = ('eligible', 'proposed', 'matched', 'correct_pixels', 'eligible_pixels', 'opening_filled',
          'opening_pixels', 'gaps', 'gaps_half_filled', 'correct_connections', 'proposed_connections', 'real_connections')
COUNTS = ('correct', 'incorrect', 'missing', 'unmatched_returned')


def compact(scores, scope, counts, correct):
    return dict(totals={**{k: scores[k] for k in TOTALS}, 'false_proposals': scope['unpaired'],
                        'small_surfaces': len(scope['small']), 'undersides': len(scope['undersides']),
                        'vertical': len(scope['vertical']), **{k: counts[k] for k in COUNTS}},
                found=scope['matched'], small=scope['small'], sizes_right=sorted(correct))


def evaluate(job):
    source, case, row, counted, sha, depth_sha, tiny, field_cache, out = job
    parts = load_predictions(source, row['predictions'], sha)
    if digest(case.path('image')) != row['sha256'] or row['sha256'] != tiny['image_sha256']:
        raise ValueError(f"Input mismatch for {row['key']}")
    with Image.open(case.path('image')) as src:
        image = src.convert('RGB')
    maps = cached_fields(row['sha256'], field_cache)
    if maps is None:
        raise FileNotFoundError(f"Restored normal and depth fields required for {row['key']}")
    if tiny['image_size'] != list(image.size) or hashlib.sha256(maps['depth'].tobytes()).hexdigest() != depth_sha:
        raise ValueError(f"Line coordinates or numerical depth changed for {row['key']}")
    start = perf_counter()
    stack = refine(image, parts[JUNCTION], {'mlsd': tiny}, maps['normal'], maps['depth'])
    elapsed = perf_counter() - start
    same = next((name for name in CONTROL if stack == parts[name]), None)
    if len({p['id'] for p in stack}) != len(stack):
        raise ValueError('Repeated proposal ID')
    relative = 'predictions/' + row['key'] + '.json.gz'
    path = out/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps({BOTH: stack}, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    # Evaluate after the combined prediction is frozen. Control sizes were previously measured
    # with the same counter. Only complete identical proposal records can reuse their readings.
    results = {}
    references = {}
    known = {}
    for name in CONTROL:
        control = counted['methods'][name]
        references[name] = {int(k): v for k, v in control['references'].items()}
        by_id = {p['id']: p for p in parts[name]}
        for r in references[name].values():
            if r['proposal'] is not None:
                signature = json.dumps(by_id[r['proposal']], sort_keys=True, separators=(',', ':'))
                if signature in known and known[signature] != r['guess']:
                    raise ValueError('Identical counter inputs produced different sizes')
                known[signature] = r['guess']
        correct = [k for k, v in references[name].items() if v['status'] == 'correct']
        results[name] = compact(row['scores'][name], row['scopes'][name], control['totals'], correct)
    fresh = 0
    count_start = perf_counter()
    if same:
        results[BOTH] = results[same]
        references[BOTH] = references[same]
    else:
        guesses = []
        for part in stack:
            key = json.dumps(part, sort_keys=True, separators=(',', ':'))
            if key not in known:
                reading = downstream_counts(image, [part])[0]
                known[key] = reading['measures']['masked-repeat']['dimensions']['blocks']
                fresh += 1
            guesses.append(known[key])
        pose, ref = recorded(case)
        faces = read(case.path('labels'))['faces']
        scores, pairs = measure(stack, ref, use_mask=True)
        native = _native_reference(pose, ref, faces)
        pixels = np.bincount(native.ravel(), minlength=max(ref['patches']) + 1)
        detail = [dict(id=i, normal=p['normal'], eligible=p['eligible'], samples=p['support_px'],
                       pixels=int(pixels[i]), methods={}) for i, p in ref['patches'].items()]
        check = score_parts(stack, BOTH, ref, native, detail)
        if check != {pid: stack[i]['id'] for i, pid in pairs}:
            raise ValueError('Independent exact identity scoring disagrees')
        small = sorted(r['id'] for r in detail if not r['eligible'] and r['methods'][BOTH]['proposal'] is not None)
        scope = dict(matched=sorted(check), small=small, unpaired=len(stack) - len(check) - len(small),
                     undersides=[k for k in check if ref['patches'][k]['normal'][1] == -1],
                     vertical=[k for k in check if ref['patches'][k]['normal'][1] == 0])
        indices = {pid: i for i, pid in pairs}
        sizes = reference_planes(ref, faces)
        truth = {}
        for pid, target in ref['patches'].items():
            if not target['eligible']:
                continue
            i = indices.get(pid)
            guess = guesses[i] if i is not None else [None, None]
            expected = sizes[pid]['size']
            truth[pid] = dict(proposal=check.get(pid), expected=expected, guess=guess,
                              status='missing' if None in guess else 'correct' if guess == expected else 'incorrect')
        references[BOTH] = truth
        counts = Counter(r['status'] for r in truth.values())
        used = set(indices.values())
        counts['unmatched_returned'] = sum(i not in used and None not in guess for i, guess in enumerate(guesses))
        results[BOTH] = compact(scores, scope, counts, [k for k, v in truth.items() if v['status'] == 'correct'])
    delta = {}
    for name in CONTROL:
        difference = {}
        for label in ('found', 'small', 'sizes_right'):
            before, after = set(results[name][label]), set(results[BOTH][label])
            difference[label] = dict(gained=sorted(after - before), lost=sorted(before - after))
        delta[name] = difference
    size_changes = {pid: {name: references[name][pid] for name in (*CONTROL, BOTH)}
                    for pid in references[BOTH]
                    if any(references[BOTH][pid] != references[name][pid] for name in CONTROL)}
    return dict(key=row['key'], sha256=row['sha256'], cohort=row['cohort'], layout=row['layout'], predictions=relative,
                methods=results, changes_vs=delta, changed_sizes=size_changes,
                identical_to=same, refinement_seconds=elapsed, freshly_counted_proposals=fresh,
                count_and_score_seconds=perf_counter() - count_start,
                stack_sha256=hashlib.sha256(json.dumps(stack, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def summarize(rows):
    totals = {}
    for name in (*CONTROL, BOTH):
        sums = Counter()
        for row in rows:
            sums.update(row['methods'][name]['totals'])
        totals[name] = dict(sums)
    changes = {}
    for name in CONTROL:
        changes[name] = {key: {change: sum(len(r['changes_vs'][name][key][change]) for r in rows)
                               for change in ('gained', 'lost')} for key in ('found', 'small', 'sizes_right')}
    return dict(frames=len(rows), methods=totals, changes_vs=changes)


@image_only()
def run(source, out, field_cache, workers):
    prior = read(source/'manifest.json')
    roster = source/'cases.json'
    count_manifest = read(source/'counts/manifest.json')
    count_cases = source/'counts/cases.json'
    if digest(roster) != prior['artifacts']['cases.json'] or digest(count_cases) != count_manifest['artifacts']['cases.json']:
        raise ValueError('Source case or counter results changed')
    if digest(TINY) != prior['line_input_sha256']['mlsd']:
        raise ValueError('M-LSD Tiny inputs changed')
    cohorts, suite, fixture = groups()
    if fixture != prior['fixtures'] or digest(ARCHIVE) != prior['reference_sha256']:
        raise ValueError('Reference or fixture identity mismatch')
    shots = {f'{c.floor}/{c.id}': c for members in cohorts.values() for c in members}
    rows = read(roster)
    counts = {r['key']: r for r in read(count_cases)}
    if len(rows) != 56 or {r['key'] for r in rows} != shots.keys() or counts.keys() != shots.keys():
        raise ValueError('Expected the fixed 56-picture suite')
    with gzip.open(TINY, 'rt', encoding='utf8') as f:
        tiny = json.load(f)
    earlier = ROOT/'benchmark/reviews/perimeters-01'
    if digest(earlier/'cases.json') != read(earlier/'manifest.json')['artifacts']['cases.json']:
        raise ValueError('Numerical depth identifiers changed')
    depth = {r['key']: r['numerical_depth_sha256'] for r in read(earlier/'cases.json')}
    out.mkdir(parents=True, exist_ok=False)
    jobs = [(source, shots[r['key']], r, counts[r['key']], prior['artifacts'][r['predictions']],
             depth[r['key']], tiny['frames'][r['sha256']], field_cache, out) for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        report = list(pool.map(evaluate, jobs))
    groups_to_score = ('all', *sorted({r['cohort'] for r in rows}), *sorted({r['layout'] for r in rows}))
    scores = {name: summarize([r for r in report if name in ('all', r['cohort'], r['layout'])])
              for name in groups_to_score}
    save(out/'cases.json', report)
    save(out/'scores.json', scores)
    save(out/'manifest.json', dict(run=out.name, type='Frozen sequential interaction, no additional tuning',
         code=revision(), fixture=fixture, minecraft=suite['version'], reference_sha256=digest(ARCHIVE),
         source_run=source.name, source_manifest_sha256=digest(source/'manifest.json'),
         count_manifest_sha256=digest(source/'counts/manifest.json'), models=prior['models'], lines=prior['line_models'],
         source_settings=prior['settings'], method_order=['wall-depth-anything', JUNCTION, 'strip-mlsd'],
         counter='masked-repeat with visible support extent, both dimensions must match',
         false_proposals='Unmatched after pairing eligible surfaces, then smaller visible references.',
         pixels='Stride-four reference samples. Opening counts include uncovered clear sightlines.',
         source_inputs=['Same screenshot, archived junction prediction, M-LSD Tiny lines, MoGe-2 normals, Depth Anything V2 depth.'],
         per_picture='cases.json', scores='scores.json', input_ids=[{k:r[k] for k in ('key','sha256','cohort','layout')} for r in report],
         artifacts={p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file()},
         source_files={p:digest(ROOT/p) for p in ('engine/surface_junction_split.py','engine/surface_strip_split.py',
                       'benchmark/junction_underside.py','engine/masked_count.py','engine/surface_dimensions.py')},
         runtime=dict(workers=workers, inference_seconds_this_run=0, cached_refinement_s=sum(r['refinement_seconds'] for r in report),
                      count_and_score_s=sum(r['count_and_score_seconds'] for r in report),
                      elapsed_note='Summed worker time. Archived full prediction sets may reuse their entire frozen evaluation.'),
         reproduce='python -m benchmark.junction_underside'))
    return scores['all']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'benchmark/reviews/recovery-01')
    parser.add_argument('--fields', type=Path, default=ROOT.parent/'normal-round/runs/normal-fields')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/junction-underside-replay')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.out.resolve(), args.fields, args.workers), indent=2))
