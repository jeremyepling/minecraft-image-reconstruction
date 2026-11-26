"""Fit and compare a masked visible-plane ranker under building-family exclusion."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path

from benchmark.canopy_depth_choices import DEST
from benchmark.texture_camera_eval import ROOT,read,save
from engine.atlas_plane_depth import frozen_weights
from engine.plane_repeat_evidence import feature_rows,pick,pick_with_fit
import numpy as np


def examples(frames):
    items=[]
    for frame in frames:
        for row in frame['rows']:
            rows=row['choices']
            y=np.array([row['truth'].get(str(r['level']),0) for r in rows],float)
            if len(rows)<2 or y.sum()<max(24,row['n']*.2):continue
            items.append((feature_rows(rows,row['baseline']),y/y.sum(),min(y.sum()**.5,60.)))
    return items


def fit(items):
    total=sum(a for _,_,a in items)
    w=np.zeros(items[0][0].shape[1])
    for _ in range(30):
        grad=.002*w
        hess=.002*np.eye(len(w))
        for x,y,a in items:
            v=x@w
            prob=np.exp(v-v.max());prob/=prob.sum()
            centered=x-prob@x
            grad+=a/total*((prob-y)@x)
            hess+=a/total*(centered.T@(prob[:,None]*centered))
        step=np.linalg.solve(hess,grad)
        w-=step/max(1.,np.max(abs(step))/2.)
        if abs(step).max()<1.e-5:break
    return w


def analyze(frames,w):
    groups=defaultdict(Counter)
    for f in frames:
        for row in f['rows']:
            q,b,t=row['choices'],row['baseline'],row['truth']
            if not q: continue
            narrow=[r for r in q if min(abs(r['level']-b),r['scene_delta'])<=6]
            wide=[r for r in q if min(abs(r['level']-b),r['scene_delta'])<=18]
            votes=lambda level:t.get(str(level),0)
            ceiling=lambda seq:max([votes(r['level']) for r in seq]+[0])
            old=pick(q,b,frozen_weights(),'old')
            best=max(t,key=t.get,default=b)
            orientation='vertical' if row['axis']!=1 else 'top' if int(best)<row['eye'] else 'underside'
            m=dict(samples=row['n'],masks=1,wrong_axis_or_background=row['n']-sum(t.values()),old=votes(old),
                wide=votes(pick(q,b,frozen_weights(),'wide')),oracle_narrow=ceiling(narrow),
                oracle_wide=ceiling(wide),oracle_near=ceiling(q),oracle_any=max(t.values(),default=0),
                learned=votes(pick_with_fit(q,b,old,w,'learned')),
                guarded=votes(pick_with_fit(q,b,old,w,'guarded')))
            for k in ('all',orientation):groups[k].update(m)
    return {k:dict(v) for k,v in groups.items()}


def main():
    train=[read(f) for f in sorted((DEST/'train').glob('*.json'))]
    assert len(train)==72
    groups=sorted({f['family'] for f in train})
    oof,folds={},{}
    for group in groups:
        w=fit(examples([f for f in train if f['family']!=group]))
        folds[group]=w.tolist()
        oof[group]=analyze([f for f in train if f['family']==group],w)
    w=fit(examples(train))
    transfer=[read(f) for f in sorted((DEST/'transfer').glob('*.json'))]
    results=dict(transfer=analyze(transfer,w),oof=oof,folds=folds)
    dest=ROOT/'benchmark/reviews/canopy-plane-chooser-01'
    dest.mkdir(parents=True,exist_ok=True)
    save(dest/'development.json',results)
    save(ROOT/'engine/repeat_plane_weights.json',dict(weights=w.tolist(),train_frames=72))
    total=Counter()
    for v in oof.values(): total.update(v['all'])
    print('LOFO',dict(total),'transfer',results['transfer'])


if __name__=='__main__':main()
