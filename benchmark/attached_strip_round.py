"""Compare attached strip cleanup methods on frozen exterior screenshots."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark import attached_strip, strip_evidence
from benchmark.afterpass_round import COHORTS
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL, measure
from benchmark.normal_round import groups
from benchmark.split_round import score_regions
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from runtime.mode import image_only, mark_image_only

METHODS = ('current', 'shape', 'evidence')


def refine_shape(image, parts):
    return attached_strip.refine(image, parts)


def refine_evidence(image, parts):
    return attached_strip.refine(image, parts, strip_evidence.gate)


def evaluate(job, refiners=(('shape', refine_shape), ('evidence', refine_evidence))):
    case,row,sha = job
    with Image.open(checked(case.path('image'),row['sha256'])) as src:
        photo = src.convert('RGB')
    baseline = read_gzip(checked(BASE/row['predictions'],sha))['gap']
    proposals = {'current':baseline}
    changes = {}
    timing = {'current':0.}
    for name, refine in refiners:
        start = perf_counter()
        proposals[name], changes[name] = refine(photo,baseline)
        timing[name] = perf_counter()-start

    # Predictions are final before any recorded camera or reference mask is opened.
    scores, scopes = score_regions(case,proposals,row['key'])
    if scores['current'] != row['scores']['gap']:
        raise ValueError('Frozen baseline mismatch')
    pose,ref = recorded(case)
    native = _native_reference(pose,ref,read(case.path('labels'))['faces'])
    _,pairs = measure(baseline,ref,use_mask=True)
    matched = {baseline[i]['id']:rid for i,rid in pairs}
    removed = {}
    for method, edits in changes.items():
        details = []
        for pid,(l,t,r,b),mask in edits:
            rid = matched.get(pid)
            n = int(mask.sum())
            correct = int((native[t:b,l:r][mask]==rid).sum()) if rid is not None else None
            details.append(dict(proposal=pid,reference=rid,removed=n,removed_correct=correct,
                                removed_background=int((native[t:b,l:r][mask]==0).sum())))
        removed[method] = details
    result = dict(key=row['key'],sha256=row['sha256'],cohort=row['cohort'],layout=row['layout'],
                  proposals=len(baseline),scopes=scopes,scores=scores,cached_processing_s=timing,removed=removed)
    score_dimensions(case,row,proposals,result,'gap')
    return result


def summarize(results, methods):
    table = {}
    for group in ('all',*COHORTS,*sorted({r['layout'] for r in results})):
        subset = [r for r in results if group in ('all',r['cohort'],r['layout'])]
        table[group] = summary(subset,methods)
        for m in methods[1:]:
            details = [d for r in subset for d in r['removed'][m]]
            matched = [d for d in details if d['reference'] is not None]
            table[group][m]['removed'] = dict(
                changed_frames=sum(bool(r['removed'][m]) for r in subset),
                changed_proposals=len(details),matched_changed=len(matched),
                correct=sum(d['removed_correct'] for d in matched),
                wrong=sum(d['removed']-d['removed_correct'] for d in matched),
                all_wrong=sum(d['removed_correct']==0 for d in matched),
                mixed=sum(0<d['removed_correct']<d['removed'] for d in matched),
                all_correct=sum(d['removed_correct']==d['removed'] for d in matched))
    return table


@image_only()
def run(out, workers):
    source = read(checked(BASE/'manifest.json',BASE_SHA))
    rows = read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    checked(ARCHIVE,source['reference_sha256'])
    cohorts,suite,fixtures = groups()
    if fixtures!=source['fixtures'] or source['protocol']!=PROTOCOL or suite['version']!=source['version']:
        raise ValueError('Fixture or scorer changed')
    captures = {c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    frames = {r['key']:r['sha256'] for r in source['frames']}
    if frames!={r['key']:r['sha256'] for r in rows} or frames.keys()!=captures.keys() or len(rows)!=56:
        raise ValueError('Unexpected fixture inventory')
    out.mkdir(parents=True,exist_ok=False)
    jobs = [(captures[r['key']],r,source['artifacts'][r['predictions']]) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results = list(pool.map(evaluate,jobs))
    table = summarize(results, METHODS)
    save(out/'cases.json',results)
    save(out/'scores.json',table)
    from benchmark.strip_review_sheet import render
    from benchmark.strip_failure_sheet import render as render_failure
    render(out)
    render_failure(out)
    modules = ['attached_strip','strip_evidence','attached_strip_round','strip_review_sheet','strip_failure_sheet']
    save(out/'manifest.json',dict(run_id=out.name,code=revision(),version=suite['version'],fixtures=fixtures,
         frames=source['frames'],splits=source['splits'],capture=source['capture'],protocol=PROTOCOL,
         reference_sha256=source['reference_sha256'],source_manifest='benchmark/reviews/sky-gap-01/manifest.json',
         source_manifest_sha256=BASE_SHA,source_code=source['code'],source_model_timing=source['source_model_timing'],
         predictions='gap',new_inference_s=0,workers=workers,methods=METHODS,
         settings=dict(shape=attached_strip.SETTINGS,evidence=strip_evidence.SETTINGS),
         module_sha256={m:digest(ROOT/'benchmark'/(m+'.py')) for m in modules},
         rule='Freeze without suite tuning. Reject default if existing matches or exact sizes lost, new incorrect sizes or openings covered, or native ownership worsens. Review collateral removals.',
         native_metric='Per baseline-matched proposal, measure removed native pixel centers against its matched reference. Unmatched proposal changes are recorded separately. Opening fill uses the established sampled scorer.',
         reproduce='python -m benchmark.attached_strip_round --out runs/attached-strip-replay',
         artifacts={f.name:digest(f) for f in out.iterdir() if f.is_file()}))
    for method in METHODS:
        rec = table['all'][method]
        print(method,rec['sizes'],rec.get('removed',{}),flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=6)
    args = cli.parse_args()
    run(args.out.resolve(),args.workers)
