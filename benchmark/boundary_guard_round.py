"""Replay the frozen two-model proposals and protect independently supported existing seams."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.afterpass_round import COHORTS, picture_table
from benchmark.boundary_guard import PRESETS, SETTINGS, guard
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL, aggregate, measure
from benchmark.normal_round import groups
from benchmark.plane_acceptance_round import qualifies
from benchmark.single_wall_round import MODEL_DIRS
from benchmark.split_round import load_predictions, save_and_score
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked
from runtime.mode import image_only, mark_image_only


SOURCE = ROOT/'benchmark/reviews/plane-acceptance-01'
DEPTH = MODEL_DIRS['single-v2']
NAMES = {'current':'Six-model Surface Finder', 'consensus':'Earlier two-model agreement',
         'guard':'Two-model agreement with protected wall boundaries'}


def inputs(job):
    row, capture, prediction_hash, depth_hash = job
    with Image.open(checked(capture.path('image'),row['sha256'])) as src:
        photo = src.convert('RGB')
    masks = load_predictions(SOURCE,row['predictions'],prediction_hash)
    path = DEPTH/'fields'/(row['sha256']+'.npz')
    with np.load(checked(path,depth_hash),allow_pickle=False) as data:
        depth = data['inverse_depth']
    return photo,masks['current'],masks['consensus'],depth


def development(job):
    parts = inputs(job)
    predictions = {key:guard(*parts,key)[0] for key in PRESETS}
    _, ref = recorded(job[1])
    return {key:measure(value,ref,use_mask=True)[0] for key,value in predictions.items()}


def evaluate(args):
    job, selected, out = args
    row, capture, _, _ = job
    photo, old, consensus, depth = inputs(job)
    start = perf_counter()
    candidate, events = guard(photo,old,consensus,depth,selected)
    elapsed = perf_counter()-start
    predicted = dict(current=old,consensus=consensus,guard=candidate)
    timings = dict(current=0.,consensus=0.,guard=elapsed)
    result = save_and_score(capture,row,predicted,timings,out,'current')
    score_dimensions(capture,row,predicted,result,baseline_method='current')
    for field in ('scores','scopes','sizes'):
        if result[field]['consensus']!=row[field]['consensus']:
            raise ValueError(f"Frozen two-model result differs at {row['key']} / {field}")
    result['events'] = {'guard':events}
    return result


@image_only()
def run(out, workers):
    base = read(SOURCE/'manifest.json')
    source = read(checked(SOURCE/'cases.json',base['artifacts']['cases.json']))
    depth = read(DEPTH/'manifest.json')
    cohorts,suite,fixtures = groups()
    captures = {f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    roster = {r['key']:r['sha256'] for r in source}
    if (len(source)!=56 or set(captures)!=set(roster) or roster!=depth['frames']
            or roster!={r['key']:r['sha256'] for r in base['frames']}
            or fixtures!=base['fixtures'] or fixtures!=depth['fixtures']):
        raise ValueError('The expected 56-frame suite differs between inputs')
    checked(ARCHIVE,base['reference_sha256'])
    jobs = [(r,captures[r['key']],base['artifacts'][r['predictions']],
             depth['artifacts']['fields/'+r['sha256']+'.npz']) for r in source]
    dev = [j for j in jobs if j[0]['cohort']=='development']
    if len(dev)!=20 or {j[0]['layout'] for j in dev}!={'shopfront','wing'}:
        raise ValueError('Unexpected development split')
    out.mkdir(parents=True,exist_ok=False)
    code = revision()
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        dev_results = list(pool.map(development,dev))
        metrics = {key:aggregate([r[key] for r in dev_results]) for key in PRESETS}
        baseline = aggregate([j[0]['scores']['current'] for j in dev])
        def rank(key):
            s = metrics[key]
            return qualifies(s,baseline),s['matched'],s['f1'],s['correct_pixels'],-PRESETS[key]
        selected = max(metrics,key=rank)
        save(out/'development-freeze.json',dict(
            rule='Apply the previous acceptance round development guardrails, then rank by matched count, F1, '
                 'correctly assigned pixels and, for exact ties, protect the smallest parent share.',
            preceding_consensus=base['selected']['consensus'], selected=selected,
            presets=PRESETS,settings=SETTINGS,development=metrics,baseline=baseline,
            frames=[j[0]['key'] for j in dev]))
        print('Development freeze:', selected, {k:v['matched'] for k,v in metrics.items()},flush=True)
        rows = list(pool.map(evaluate,[(j,selected,out) for j in jobs]))
    methods = tuple(NAMES)
    scores = {'all':summary(rows,methods)}
    for scope in (*COHORTS,*sorted({r['layout'] for r in rows})):
        scores[scope] = summary([r for r in rows if scope in (r['cohort'],r['layout'])],methods)
    save(out/'scores.json',scores)
    save(out/'cases.json',rows)
    (out/'pictures.md').write_text(picture_table(rows,methods,NAMES),encoding='utf8')
    save(out/'manifest.json',dict(run_id=out.name,code=code,version=suite['version'],fixtures=fixtures,
         reference_sha256=base['reference_sha256'],capture=base['capture'],splits=base['splits'],
         frames=base['frames'],methods=NAMES,baseline='plane-recovery',protocol=PROTOCOL,
         selected=selected,preceding_consensus=base['selected']['consensus'],settings=SETTINGS,
         upstream_manifests={d.name:digest(d/'manifest.json') for d in (SOURCE,DEPTH)},
         inherited_model_manifests=base['upstream_manifests'],depth_checkpoint=depth['model'],
         new_inference_s=0,guard_seconds=scores['all']['guard']['cached_extra_seconds'],
         timing='Baseline and previously frozen consensus masks are reused. Guard time excludes scoring and loading.',
         stages={'source':'per-image frozen consensus and current masks, relative depth on the same screenshot',
                 'outlines':'Depth and orientation guard for existing walls','counter':'masked-repeat'},
         reproduce='python -m benchmark.boundary_guard_round --out runs/boundary-guard-repeat',
         artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))
    print({k:(v['matched'],v['lost'],v['unpaired'],v['sizes']['correct'])
           for k,v in scores['all'].items()},flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'boundary-guard-{uuid4().hex[:10]}')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    run(args.out.resolve(),args.workers)
