"""Compare texture-spacing methods on identical crops from the shipped Surface Finder."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import reference_planes
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only, mark_image_only


SOURCE = ROOT/'benchmark/reviews/detail-01'
FINDERS = {'neighbor':(SOURCE,'current'),
           'photo':(ROOT/'benchmark/reviews/edge-detail-stable-01','edge_detail:native4-stable-counts')}
MODULES = {'repeat':'portico_repeat','atlas':'portico_atlas','rectify':'repeat_rectify'}


def scalar(x):
    if isinstance(x,np.ndarray):
        return x.tolist()
    if isinstance(x,np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def evaluate(job):
    capture,row,pred_hash,methods,out,finder = job
    source,method = FINDERS[finder]
    parts = read_gzip(checked(source/row['predictions'],pred_hash))[method]
    with Image.open(checked(capture.path('image'),row['sha256'])) as opened:
        photo = opened.convert('RGB')
    functions = {m:importlib.import_module('benchmark.'+MODULES[m]) for m in methods}
    preds,records = [],[]
    timing = Counter()
    for part in parts:
        sample = sample_surface(photo,part)
        start = perf_counter()
        base = count(sample)
        timing['current'] += perf_counter()-start
        evidence = {'current':base}
        for name,module in functions.items():
            start = perf_counter()
            if name=='rectify':
                evidence[name] = module.predict_surface(photo,part,sample,deepcopy(base))
            else:
                evidence[name] = module.predict(sample,deepcopy(base))
            timing[name] += perf_counter()-start
        quad = sample['quad'] if sample else None
        flags = {m:dimensions(part,res.get('quad',quad),photo.size,res['counts'])['blocks'] for m,res in evidence.items()}
        records.append(dict(proposal=part['id'],guess=flags))
        preds.append(dict(proposal=part['id'],chart=quad,chart_wh=list(sample['mask'].shape[::-1]) if sample else None,
                          valid=int(sample['mask'].sum()) if sample else 0,methods=evidence))
    # The predictor has finished before any reference face, material or dimension is read.
    _,ref = recorded(capture)
    faces = read(capture.path('labels'))['faces']
    targets = reference_planes(ref,faces)
    _,pairs = measure(parts,ref,use_mask=True)
    for i,rid in pairs:
        target=targets[rid]
        records[i].update(reference=rid,actual=target['size'],
                          materials=dict(Counter(f['block_id'] for f in target['faces'].values())))
    rel = 'predictions/'+row['key']+'.json.gz'
    path=out/rel
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(preds,allow_nan=False,default=scalar,separators=(',',':')).encode(),mtime=0))
    frame=dict(key=row['key'],layout=row['layout'],cohort=row['cohort'],sha256=row['sha256'],
               eligible=row['scores'][method]['eligible'],predictions=rel,surfaces=records,processing_s=dict(timing))
    frame['scores']=summarize([frame],['current',*methods])
    if any(frame['scores']['current'][k]!=v for k,v in row['sizes'][method].items()):
        raise ValueError('Baseline size scores changed: '+row['key'])
    return frame


def summarize(rows,methods):
    scores={}
    for method in methods:
        total=Counter(dict(correct=0,incorrect=0,unmatched_returned=0,axis_correct=0,axis_wrong=0,
                           prior_correct_lost=0,new_correct=0,new_wrong=0))
        for frame in rows:
            total['eligible']+=frame['eligible']
            for rec in frame['surfaces']:
                guess=rec['guess'][method]
                if 'actual' not in rec:
                    total['unmatched_returned']+=int(None not in guess)
                    continue
                actual,old=rec['actual'],rec['guess']['current']
                total['matched']+=1
                total['correct']+=int(guess==actual)
                total['incorrect']+=int(None not in guess and guess!=actual)
                total['prior_correct_lost']+=int(old==actual and guess!=actual)
                total['new_correct']+=int(old!=actual and guess==actual)
                total['new_wrong']+=int((None in old or old==actual) and None not in guess and guess!=actual)
                for g,a in zip(guess,actual):
                    total['axis_correct']+=int(g==a)
                    total['axis_wrong']+=int(g is not None and g!=a)
        total['missing']=total['eligible']-total['correct']-total['incorrect']
        scores[method]=dict(total)
    return scores


@image_only()
def run(out,methods,split,workers,sheet=False,finder='neighbor'):
    source,source_method = FINDERS[finder]
    manifest=read(source/'manifest.json')
    rows=read(checked(source/'cases.json',manifest['artifacts']['cases.json']))
    cohorts,suite,fixtures=groups()
    cases={c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    if len(cases)!=56 or fixtures!=manifest['fixtures'] or set(cases)!={r['key'] for r in rows}:
        raise ValueError('Fixture inventory changed')
    rows=[r for r in rows if split=='all' or r['cohort']==split]
    out.mkdir(parents=True,exist_ok=False)
    code=revision()
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results=list(pool.map(evaluate,[(cases[r['key']],r,manifest['artifacts'][r['predictions']],methods,out,finder) for r in rows]))
    names=['current',*methods]
    scores={k:summarize([r for r in results if k in ('all',r['cohort'],r['layout'])],names)
            for k in dict.fromkeys(['all',*[r['cohort'] for r in rows],*[r['layout'] for r in rows]])}
    save(out/'scores.json',scores)
    save(out/'cases.json',results)
    if sheet:
        from benchmark.texture_count_sheet import render
        render(out,finder)
    atlas_source = {}
    if 'atlas' in methods:
        from benchmark.portico_atlas import atlas
        atlas_source = atlas().provenance
    save(out/'manifest.json',dict(run_id=out.name,atlas=atlas_source,code=code,version=suite['version'],fixtures=fixtures,
         frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in rows],
         source_manifest_sha256=digest(source/'manifest.json'),source_manifest=(source/'manifest.json').relative_to(ROOT).as_posix(),
         source_method=source_method,
         reference_sha256=manifest['reference_sha256'],methods=names,split=split,workers=workers,
         settings={m:getattr(importlib.import_module('benchmark.'+MODULES[m]),'SETTINGS',{}) for m in methods},
         reproduce='python -m benchmark.texture_count_round '+' '.join('--method '+m for m in methods)
                   +' --finder '+finder+' --split '+split+' --out runs/my-texture-count'+(' --sheet' if sheet else ''),
         method_sha256={m:digest(ROOT/'benchmark'/(MODULES[m]+'.py')) for m in methods},
         timing='Per-frame CPU wall time per method. Candidate processing follows the shared baseline.',
         artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(scores['all'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',action='append',choices=MODULES,default=[])
    parser.add_argument('--split',choices=['development','previously_reviewed_regression','all'],default='all')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--finder',choices=FINDERS,default='neighbor')
    parser.add_argument('--sheet',action='store_true')
    args=parser.parse_args()
    if args.sheet and (set(args.method)!=set(MODULES) or args.split!='all'):
        parser.error('--sheet needs both methods on all views')
    run(args.out.resolve(),args.method,args.split,args.workers,args.sheet,args.finder)
