"""Fit a local chooser only after the constrained whole-section oracle has been recorded."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.thin_face_fit import boost,fit
from benchmark.texture_camera_eval import read,save,digest
from engine import wall_section_grid as grid


def train(method,rows):
    x,y=[],[]
    for row in rows:
        for choices in row['sections']:
            for opt in choices:
                x.append(opt['features'])
                y.append(opt['positive']/len(opt['cells']))
    model=method(np.array(x),np.array(y))
    model['features']=list(grid.FEATURES)
    return model


def counts(model,threshold,rows):
    result=Counter()
    for row in rows:
        selected=grid.select(row['sections'],model,threshold)
        for opt in selected:
            result['true']+=opt['positive']
            result['false']+=len(opt['cells'])-opt['positive']
    return dict(true=result['true'],false=result['false'])


def run(args):
    oracle=read(args.source/'oracle.json')
    assert oracle['groups']['train']['actual_oracle_unique_correct']>0
    rows=[read(p) for p in sorted((args.source/'train').glob('*.json'))]
    assert len(rows)==72
    families=sorted({r['summary']['family'] for r in rows})
    assert len(families)==6
    traces={}
    limits=(.5,.65,.75,.8,.85,.88,.9,.92,.94,.96,.97,.98,.99,.995)
    best={}
    for label,method in [('logistic',fit),('boosted',boost)]:
        folds={}
        for family in families:
            model=train(method,[r for r in rows if r['summary']['family']!=family])
            held=[r for r in rows if r['summary']['family']==family]
            folds[family]={str(t):counts(model,t,held) for t in limits}
        totals={t:{k:sum(f[str(t)][k] for f in folds.values()) for k in ('true','false')} for t in limits}
        traces[label]=dict(folds=folds,totals=totals)
        for rule in ['strict','balanced']:
            for threshold in limits:
                tp,fp=(totals[threshold][k] for k in ('true','false'))
                limit=0 if rule=='strict' else .03*tp
                if not tp or fp>limit:
                    continue
                utility=tp-10*fp
                if rule not in best or utility>best[rule]['utility']:
                    best[rule]=dict(method=label,threshold=threshold,true=tp,false=fp,utility=utility)
    for mode in ['strict','balanced']:
        if mode not in best:
            best[mode]=dict(method='logistic',threshold=1.,true=0,false=0,utility=0)
    # Keep one shared model in production and set both thresholds on that family's OOF data.
    # Also freeze the operating point with best OOF net F1 utility, including its measured false positives.
    _,family,operating=max((row['true']-.84*row['false'],label,t)
                             for label,head in traces.items() for t,row in head['totals'].items())
    if best['strict']['true']:
        family=best['strict']['method']
        operating=max(limits,key=lambda t:traces[family]['totals'][t]['true']-.84*traces[family]['totals'][t]['false'])
    if best['balanced']['method']!=family:
        allowed=[(row['true']-10*row['false'],t,row) for t,row in traces[family]['totals'].items()
                 if row['true'] and row['false']<=.03*row['true']]
        if allowed:
            utility,t,row=max(allowed)
            best['balanced']=dict(method=family,threshold=t,utility=utility,**row)
        else:
            best['balanced']=best['strict']
    final=train(fit if family=='logistic' else boost,rows)
    final['thresholds']={k:v['threshold'] for k,v in best.items()}
    final['thresholds']['operating']=operating
    args.out.mkdir(exist_ok=True,parents=True)
    save(grid.MODEL,final)
    save(args.out/'training.json',dict(oracle_sha256=digest(args.source/'oracle.json'),approach=family,best=best,
                                        comparisons=traces,development_ids=[r['summary']['id'] for r in rows]))
    print(best)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
