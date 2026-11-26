"""Compare one supported wall outline with the same vote from two depth models."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import PROTOCOL
from benchmark.normal_round import groups
from benchmark.single_wall_metrics import boundary_truth, grade_votes
from benchmark.split_count_check import count_frame
from benchmark.split_round import load_predictions, pooled, save_and_score
from benchmark.surface_reference import ARCHIVE
from runtime.mode import image_only, mark_image_only


REVIEWS=ROOT/'benchmark/reviews'
BASE=REVIEWS/'recovery-summary-01'
REGIONS=REVIEWS/'mobile-regions-01'
METHODS=('current','single-v2','single-pro')
LABELS=dict(zip(METHODS,('Wall Recovery','One outline + Depth Anything V2','One outline + Depth Pro')))
MODEL_DIRS={'single-v2':REVIEWS/'single-wall-v2-fields-01','single-pro':REVIEWS/'depth-pro-fields-01'}
RULE=('Select on Shopfront and Wing: require more exact surface matches and higher pooled surface F1 '
      'than Wall Recovery, preserve all opening pixels, and rank by matches, then F1, then correctly '
      'assigned pixels. On reviewed regression require at least as many matches and no lower F1. '
      'Speed and number of models are excluded from selection.')


def verify(path,expected):
    if digest(path)!=expected:
        raise ValueError(f'File differs from the pinned run: {path}')
    return path


def evaluate(job):
    from engine.single_wall_gate import prepare, apply
    case,row,inputs,out=job
    key,sha=row['key'],row['sha256']
    with Image.open(verify(case.path('image'),sha)) as src:
        image=src.convert('RGB')
    path,expected=row['resources']['combined']
    baseline=load_predictions(REVIEWS,path,expected)
    with np.load(verify(*inputs['masks']),allow_pickle=False) as data:
        masks=np.unpackbits(data['packed'],count=int(data['shape'][2]),axis=2).astype(bool)
    with gzip.open(verify(*inputs['proposals']),'rt',encoding='utf8') as f:
        meta=json.load(f)
    start=perf_counter()
    inventory=prepare(image,baseline,masks,meta['candidates'],meta['selected'])
    times={'prepare':perf_counter()-start}
    predictions={'current':baseline}
    events={}
    for method in METHODS[1:]:
        with np.load(verify(*inputs[method]),allow_pickle=False) as data:
            grid=data['inverse_depth']
        if grid.shape!=masks.shape[1:] or not np.isfinite(grid).all():
            raise ValueError(f'Invalid {method} depth lattice for {key}')
        start=perf_counter()
        predictions[method],events[method]=apply(image,baseline,masks,meta['candidates'],inventory,grid)
        times[method]=perf_counter()-start
    result=save_and_score(case,row,predictions,times,out,'combined')
    baseline_matches=result['scopes']['current']['matched']
    if baseline_matches!=row['scopes']['combined']['matched']:
        raise ValueError(f'Control match IDs differ for {key}')
    for method in METHODS[1:]:
        for metric in ('gaps_half_filled','opening_filled','unclaimed_pixels','background_claimed'):
            if result['scores'][method][metric]!=result['scores']['current'][metric]:
                raise ValueError(f'{method}: original coverage changed for {key}')
    remaining=set(row['remaining']['in_filtered_sam'])
    result['opportunities']=sorted(remaining)
    result['recovered_opportunities']={m:sorted(remaining.intersection(result['scopes'][m]['gained'])) for m in METHODS}
    truth=boundary_truth(case,inventory)
    result['vote_grades']={m:grade_votes(truth,events[m]) for m in METHODS[1:]}
    result['boundary_evidence']='boundaries/'+key+'.json.gz'
    inventory_sha=hashlib.sha256(json.dumps(inventory,separators=(',',':'),sort_keys=True).encode()).hexdigest()
    evidence=dict(inventory=inventory,inventory_sha256=inventory_sha,events=events,reference_diagnostics=truth)
    path=out/result['boundary_evidence']
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(evidence,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    result['inventory_sha256']=inventory_sha
    sizes,_=count_frame(case,image,result,predictions,METHODS)
    result['sizes']={m:{k:d['totals'][k] for k in ('correct','incorrect','missing','unmatched_returned')} for m,d in sizes.items()}
    result['size_changes']={m:d['comparisons'] for m,d in sizes.items()}
    return result


def summary(rows):
    table={}
    for m in METHODS:
        t=pooled(rows,m)
        votes,counts=Counter(),Counter()
        for r in rows:
            counts.update(r['sizes'][m])
            votes.update(r['vote_grades'].get(m,{}))
        t.update(sizes=dict(counts),votes=dict(votes),opportunities=sum(len(r['opportunities']) for r in rows),
                 recovered_opportunities=sum(len(r['recovered_opportunities'][m]) for r in rows))
        table[m]=t
    return table


def choose(dev):
    base=dev['current']
    eligible=[m for m in METHODS[1:] if dev[m]['matched']>base['matched'] and dev[m]['f1']>base['f1']
              and dev[m]['opening_filled']==base['opening_filled']]
    return max(eligible,key=lambda m:(dev[m]['matched'],dev[m]['f1'],dev[m]['correct_pixels']),default='current')


def picture_table(rows):
    lines=['# Wall outlines by screenshot','',
       'The first three results count matched surfaces. Correct sizes require both dimensions in blocks.','',
       '| Screenshot | Scored | Wall Recovery | One outline + V2 | One outline + Depth Pro | Both sizes right (control / V2 / Pro) |',
       '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in sorted(rows,key=lambda r:(r['layout'],r['key'])):
        cells=[r['key'],r['scores']['current']['eligible'],*(r['scores'][m]['matched'] for m in METHODS),
               ' / '.join(str(r['sizes'][m]['correct']) for m in METHODS)]
        lines.append('| '+' | '.join(map(str,cells))+' |')
    return '\n'.join(lines)+'\n'


@image_only()
def run(out,workers):
    from engine.single_wall_gate import SETTINGS
    from benchmark.single_wall_opportunities import write_diagnostic
    from benchmark.single_wall_sheet import render
    source=read(BASE/'manifest.json')
    baseline=read(verify(BASE/'cases.json',source['artifacts']['cases.json']))
    cohorts,suite,fixtures=groups()
    cases={f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    if len(baseline)!=56 or set(cases)!={r['key'] for r in baseline} or fixtures!=source['fixtures'] or digest(ARCHIVE)!=source['reference_sha256']:
        raise ValueError('Frozen suite differs from the Wall Recovery control')
    original=read(REVIEWS/'recovery-01/manifest.json')
    sam=read(verify(REGIONS/'manifest.json',source['sources']['mobile-regions-01/manifest.json']))
    maps={m:read(p/'manifest.json') for m,p in MODEL_DIRS.items()}
    out.mkdir(parents=True,exist_ok=False)
    code=revision()
    save(out/'protocol.json',dict(selection=RULE,gate=SETTINGS,scoring=PROTOCOL))
    rows,scores=[],{}
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        for group in ('development','previously_reviewed_regression'):
            jobs=[]
            for row in baseline:
                if row['cohort']!=group:
                    continue
                sha=row['sha256']
                inputs={}
                for key,ext in [('masks','.npz'),('proposals','.json.gz')]:
                    name='proposals/'+sha+ext
                    inputs[key]=(REGIONS/name,sam['artifacts'][name])
                for m,path in MODEL_DIRS.items():
                    name='fields/'+sha+'.npz'
                    inputs[m]=(path/name,maps[m]['artifacts'][name])
                jobs.append((cases[row['key']],row,inputs,out))
            cohort=list(pool.map(evaluate,jobs))
            rows.extend(cohort)
            scores[group]=summary(cohort)
            if group=='development':
                selected=choose(scores[group])
                save(out/'development-selection.json',dict(selected=selected,rule=RULE,scores=scores[group]))
            print(group,{m:scores[group][m]['matched'] for m in METHODS},flush=True)
    reg=scores['previously_reviewed_regression']
    recommended=selected if reg[selected]['matched']>=reg['current']['matched'] and reg[selected]['f1']>=reg['current']['f1'] else 'current'
    scores['all']=summary(rows)
    for layout in sorted({r['layout'] for r in rows}):
        scores[layout]=summary([r for r in rows if r['layout']==layout])
    save(out/'cases.json',rows)
    save(out/'scores.json',scores)
    (out/'pictures.md').write_text(picture_table(rows),encoding='utf8')
    sources={d.relative_to(REVIEWS).as_posix():digest(d/'manifest.json') for d in
             (BASE,REGIONS,REVIEWS/'single-wall-selection-01',*MODEL_DIRS.values())}
    manifest=dict(run_id=out.name,code=code,version=suite['version'],fixtures=fixtures,
                  reference_sha256=digest(ARCHIVE),capture=original['capture'],splits=original['splits'],
                  per_picture='cases.json',sources=sources,methods=LABELS,settings=SETTINGS,protocol=PROTOCOL,
                  selected_on_development=selected,recommended=recommended,selection_rule=RULE,models=original['models'],
                  line_models=original['line_models'],mobile_model=dict(source_revision=sam['source_revision'],
                        checkpoint_sha256=sam['checkpoint_sha256']),depth_pro_sha256=maps['single-pro']['checkpoint_sha256'],
                  model_sources={m:MODEL_DIRS[m].name+'/manifest.json' for m in METHODS[1:]},
                  frames=[{k:r[k] for k in ('key','sha256','cohort','layout','inventory_sha256')} for r in rows],
                  cached_seconds={k:sum(r['cached_processing_s'][k] for r in rows) for k in ('prepare',*METHODS[1:])},
                  scores='scores.json',sheet='comparison.jpg',reproduce=r'.\Invoke-SurfaceRound.ps1 -Round single-wall',
                  artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    save(out/'manifest.json',manifest)
    render(out,out)
    write_diagnostic(out,out/'opportunities.json')
    for name in ('comparison.jpg','sheet-views.json','opportunities.json'):
        manifest['artifacts'][name]=digest(out/name)
    save(out/'manifest.json',manifest)
    print(out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'runs'/f'single-wall-{uuid4().hex[:12]}')
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    run(args.out.resolve(),args.workers)
