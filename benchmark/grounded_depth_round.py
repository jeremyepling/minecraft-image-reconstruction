"""Fit and compare grounded depth rejection using whole-building-family holdouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
from benchmark import nearby_face_trial as trial
from benchmark.nearby_face_fit import train
from benchmark.texture_camera_eval import ROOT, STABLE, checked, read, save
from engine import grounded_depth_envelope as envelope
from engine.nearby_face_decision import predict_forest


BASE=STABLE.parent/(STABLE.name+'-worktrees')/'nearby-thin-geometry/runs/release'
ATLAS=STABLE/'runs/26.3/atlas'
CACHE=ROOT/'runs/grounded-depth'
REPORT=ROOT/'benchmark/reviews/grounded-depth-envelope-01'


def jobs(split):
    frames=trial.cases(split)
    source=BASE/split
    indexed={r['id']:r for r in read(source/'cases.json')}
    return [(s,f,sources,source,indexed[f['id']]) for s,f,sources,_,_ in frames]


def train_example(job):
    split,f,*_=job
    image,parts,camera,prior=trial.load_prediction(job)
    obs=envelope.measure(image,parts,camera,prior,atlas_path=ATLAS)
    T,truth,world=trial.labels(job,camera['P'])
    if obs is None:
        pts=np.empty((0,3),int); X=np.empty((0,len(envelope.FEATURES)))
    else:
        pts,X=obs
    coords=np.floor((pts+.5)@T[:3,:3].T+T[:3,3]).astype(int)+job[-1]['translation']
    false=np.array([tuple(p) not in truth for p in coords],bool)
    hidden=np.array([tuple(p) in world and tuple(p) not in truth for p in coords],bool)
    dest=CACHE/split/(f['id']+'.npz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,points=pts,features=X,false=false,hidden=hidden)
    return len(X)


def selection():
    rows=jobs('train')
    samples=[dict(np.load(CACHE/'train'/(j[1]['id']+'.npz'))) for j in rows]
    groups=np.array([j[1]['family'] for j in rows])
    X=np.concatenate([s['features'] for s in samples]); y=np.concatenate([s['false'] for s in samples])
    hidden=np.concatenate([s['hidden'] for s in samples])
    G=np.repeat(groups,[len(s['points']) for s in samples])
    print('rows',len(y),'false',y.sum(),'shown',len(samples),flush=True)
    variants={}
    for padding in (2,4,8):
        for confidence in (0,1):
            keep=(X[:,0]>padding) & (X[:,11]>1) & (X[:,19]<(.5 if confidence else .8))
            variants[f'rule-{padding}-{confidence}']=stats(keep,y,hidden,G)
    for depth in [2,3]:
        p=np.zeros(len(y))
        for hold in sorted(set(G)):
            left=G!=hold
            model=train(X[left],y[left],rounds=72,depth=depth)
            p[~left]=predict_forest(model,X[~left])
        for t in (.50,.60,.70,.75,.80,.85,.90,.95,.98):
            for local in (False,True):
                gate=(X[:,0]>.5)|(X[:,11]>2) if local else np.ones(len(y),bool)
                variants[f'model-{depth}-{t}-{int(local)}']=stats((p>=t)&gate,y,hidden,G)
    REPORT.mkdir(parents=True,exist_ok=True)
    save(REPORT/'family-holdouts.json',variants)
    for k,v in variants.items():
        print(k,v['removed_extras'],v['lost_correct'],v['hidden_removed'],v['family'])
    robust=[k for k,v in variants.items() if k.startswith('model') and v['removed_extras']>0 and
            v['removed_extras']>=5*v['lost_correct'] and min(m[0]-m[1]*2 for m in v['family'].values())>=0]
    if not robust:
        raise ValueError('No candidate passed family and retention checks')
    best=max(robust,key=lambda k: variants[k]['removed_extras']-3*variants[k]['lost_correct'])
    print('selected',best)
    _,depth,t,local=best.split('-')
    model=train(X,y,rounds=72,depth=int(depth))
    obj=dict(features=envelope.FEATURES,forest=model,threshold=float(t),local=bool(int(local)))
    envelope.WEIGHTS.write_text(json.dumps(obj,separators=(',',':'))+'\n')
    save(REPORT/'selection.json',dict(selected=best,rows=len(y),incorrect=int(y.sum()),family_holdouts=sorted(set(G))))


def stats(mask,y,hidden,G):
    return dict(removed_extras=int(sum(mask&y)),lost_correct=int(sum(mask&~y)),hidden_removed=int(sum(mask&hidden)),
                family={g:[int(sum(mask&y&(G==g))),int(sum(mask&~y&(G==g)))] for g in sorted(set(G))})


def apply(image,parts,counts,camera,prior):
    return envelope.recover(image,parts,camera,prior,atlas_path=ATLAS)


def neighbors(image,parts,counts,camera,prior):
    return envelope.recover(image,parts,camera,prior,atlas_path=ATLAS,threshold=.80,neighbors=True)


def conservative(image,parts,counts,camera,prior):
    return envelope.recover(image,parts,camera,prior,atlas_path=ATLAS,threshold=.98)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['features','fit'])
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.mode=='features':
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            print(sum(pool.map(train_example,jobs('train'))))
    else:
        selection()
