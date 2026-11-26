"""Compare crop windows with the same frozen Surface Finder masks and Block Counter."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import reference_planes
from benchmark.mask_metrics import PROTOCOL, measure
from benchmark.normal_round import groups
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import SETTINGS as COUNT_SETTINGS, count
from engine.surface_dimensions import dimensions
from engine.surface_perimeter import _native_alpha
from engine.surface_sample import SETTINGS as CURRENT_SETTINGS, sample_surface
from runtime.mode import image_only, mark_image_only


BASE = ROOT / 'benchmark/reviews/sky-gap-01'
BASE_SHA = '67fe02ade765f35d5fa6f8eb85d248fd15adf1227717a9ba6209decef67ebe07'
METHODS = ('current', 'full', 'guarded')
RULE = ('Prefer greater native support coverage among methods with at least as many correct '
        'sizes and no more incorrect sizes than current on development. Retain current on ties. '
        'Check that the size criteria also hold on reviewed regression before promotion.')


def predictors():
    from benchmark.crop_full_fit import sample as full
    from benchmark.crop_guarded_fit import sample as guarded
    return dict(current=sample_surface, full=full, guarded=guarded)


def predict(image, parts):
    results = [{} for part in parts]
    elapsed = {}
    for method, sample_fn in predictors().items():
        start = perf_counter()
        for result, part in zip(results, parts):
            sample = sample_fn(image, part)
            quad = None if sample is None else sample['quad']
            counts = count(sample)['counts']
            guess = dimensions(part, quad, image.size, counts)['blocks']
            result[method] = dict(quad=quad, guess=guess,
                valid_output_pixels=0 if sample is None else int(sample['mask'].sum()))
        elapsed[method] = perf_counter() - start
    return results, elapsed


def chart_contains(x, y, quad):
    if quad is None:
        return np.zeros(x.size, bool)
    corners = np.asarray(quad)
    signs = np.array([(b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0])
                      for a,b in zip(corners, np.roll(corners,-1,axis=0))])
    return np.all(signs >= -1.e-8, axis=0) | np.all(signs <= 1.e-8, axis=0)


def score_crop(result, part, rid, size, native, truth):
    alpha, (left, top, _, _) = _native_alpha(part['support'], size)
    y, x = np.nonzero(np.asarray(alpha))
    x, y = x + left, y + top
    same = native[y, x] == rid
    x, y = x + .5, y + .5
    for method, rec in result.items():
        inside = chart_contains(x, y, rec['quad'])
        if method == 'current':
            original = inside
        gained = inside & ~original
        lost = original & ~inside
        guess = rec['guess']
        rec.update(status='missing' if None in guess else 'correct' if guess == truth else 'incorrect',
                   covered=int(inside.sum()), correct_pixels=int((inside & same).sum()),
                   other_pixels=int((inside & ~same).sum()), gained_correct_pixels=int((gained & same).sum()),
                   gained_other_pixels=int((gained & ~same).sum()), lost_correct_pixels=int((lost & same).sum()),
                   lost_other_pixels=int((lost & ~same).sum()))
    return dict(reference_id=rid, proposal=part['id'], expected=truth, source_mask_pixels=int(x.size),
                methods=result)


def evaluate(job):
    capture, row, sha = job
    with Image.open(checked(capture.path('image'), row['sha256'])) as src:
        image = src.convert('RGB')
    parts = read_gzip(checked(BASE / row['predictions'], sha))['gap']
    results, timing = predict(image, parts)

    pose, ref = recorded(capture)
    faces = read(capture.path('labels'))['faces']
    native = _native_reference(pose, ref, faces)
    sizes = reference_planes(ref, faces)
    _, pairs = measure(parts, ref, use_mask=True)
    if sorted(rid for i,rid in pairs) != row['scopes']['gap']['matched']:
        raise ValueError(f"Surface matches changed for {row['key']}")
    matched = [score_crop(results[i], parts[i], rid, image.size, native, sizes[rid]['size'])
               for i, rid in pairs]
    indices = {i for i,rid in pairs}
    unmatched = {m:sum(None not in r[m]['guess'] for i,r in enumerate(results) if i not in indices)
                 for m in METHODS}
    frame = dict(key=row['key'], sha256=row['sha256'], layout=row['layout'],
                 cohort='development' if row['cohort']=='development' else 'regression',
                 eligible=row['scores']['gap']['eligible'], matched=matched,
                 unmatched_returned=unmatched, processing_s=timing)
    frame['scores'] = totals([frame])
    for k,v in row['sizes']['gap'].items():
        if frame['scores']['current'][k] != v:
            raise ValueError(f"Baseline {k} changed on {row['key']}")
    return frame


def totals(rows):
    eligible = sum(f['eligible'] for f in rows)
    surfaces = [s for f in rows for s in f['matched']]
    table = {}
    for method in METHODS:
        records = [s['methods'][method] for s in surfaces]
        usable = [bool(r['valid_output_pixels']) for r in records]
        counter = Counter(r['status'] for r in records)
        sources = sum(s['source_mask_pixels'] for s in surfaces)
        total = dict(eligible=eligible, matched=len(surfaces), usable=sum(usable),
            below_90=sum(ok and r['covered'] < .9*s['source_mask_pixels']
                         for ok,r,s in zip(usable,records,surfaces)),
            no_chart=sum(r['quad'] is None for r in records),
            correct=counter['correct'], incorrect=counter['incorrect'], missing=eligible-counter['correct']-counter['incorrect'],
            wins=sum(s['methods']['current']['status']!='correct' and r['status']=='correct' for s,r in zip(surfaces,records)),
            losses=sum(s['methods']['current']['status']=='correct' and r['status']!='correct' for s,r in zip(surfaces,records)),
            unmatched_returned=sum(f['unmatched_returned'][method] for f in rows), source_mask_pixels=sources,
            processing_s=sum(f['processing_s'][method] for f in rows))
        for name in ('covered','correct_pixels','other_pixels','gained_correct_pixels','gained_other_pixels',
                     'lost_correct_pixels','lost_other_pixels'):
            total[name] = sum(r[name] for r in records)
        total['coverage'] = total['covered']/sources if sources else None
        total['usable_fraction'] = sum(usable)/len(surfaces) if surfaces else None
        table[method] = total
    return table


@image_only()
def run(out, workers):
    from benchmark import crop_guarded_fit
    from benchmark.crop_fit_sheet import render
    source = read(checked(BASE/'manifest.json', BASE_SHA))
    rows = read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    checked(ARCHIVE,source['reference_sha256'])
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}':c for cohort in cohorts.values() for c in cohort}
    frames = {r['key']:r['sha256'] for r in rows}
    if (len(rows)!=56 or set(captures)!=frames.keys() or fixtures!=source['fixtures'] or source['protocol']!=PROTOCOL or
            frames!={r['key']:r['sha256'] for r in source['frames']} or suite['version']!=source['version']):
        raise ValueError('Frozen fixtures, version or scoring protocol changed')
    out.mkdir(parents=True, exist_ok=False)
    jobs = [(captures[r['key']],r,source['artifacts'][r['predictions']]) for r in rows]
    code = revision()
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results = list(pool.map(evaluate,jobs))
    scores = {group:totals([r for r in results if group in ('all',r['cohort'],r['layout'])])
              for group in ('all','development','regression',*sorted({r['layout'] for r in results}))}
    if (scores['all']['current']['correct'],scores['all']['current']['incorrect'])!=(94,51):
        raise ValueError('Expected sky-gap-01 block counts')
    save(out/'cases.json',results)
    save(out/'scores.json',scores)
    render(out)
    paths = dict(current='engine/surface_sample.py',full='benchmark/crop_full_fit.py',guarded='benchmark/crop_guarded_fit.py')
    save(out/'manifest.json',dict(run_id=out.name,code=code,version=source['version'],fixtures=fixtures,
        frames=source['frames'],splits=source['splits'],capture=source['capture'],protocol=PROTOCOL,
        reference_sha256=source['reference_sha256'],source_manifest='benchmark/reviews/sky-gap-01/manifest.json',
        source_manifest_sha256=BASE_SHA,source_code=source['code'],predictions='gap',selection=RULE,
        source_model_timing=source['source_model_timing'],new_inference_s=0,workers=workers,
        timing='Crop + unchanged Block Counter + dimensions, per method across all proposals. I/O and reference scoring excluded.',
        methods=paths,method_sha256={m:digest(ROOT/p) for m,p in paths.items()},
        settings=dict(current=CURRENT_SETTINGS,full='Full native support pixel edges in original chart axes',
                      shared_expansion=dict(horizon_relative_epsilon=1e-6,minimum_corner_change_px=.5,
                                            allowed_bounds_in_canvas_spans=[-1,2]),
                      guarded=crop_guarded_fit.SETTINGS,block_counter=COUNT_SETTINGS),
        coverage='Native matched mask pixel centers inside chart / all matched mask pixels, including charts unavailable. '
                 'Below 90 counts usable samples only. Correct pixels belong to the matched true surface. '
                 'This grades window extent before the unchanged four-tap sampling filter.',
        size_scope='Matched size estimates scored against actual full connected surface width and height in blocks. '
                   'Missing includes undetected eligible surfaces. Unmatched guesses counted separately.',
        reproduce='python -m benchmark.crop_fit_round --out runs/crop-fit-replay',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(scores['all'],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=4)
    args=cli.parse_args()
    run(args.out.resolve(),args.workers)
