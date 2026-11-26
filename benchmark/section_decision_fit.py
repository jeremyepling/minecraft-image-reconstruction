"""Fit and freeze the section acceptance/removal heads using only framed training images."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json

import numpy as np
from benchmark.nearby_face_fit import train
from benchmark.plane_section_trial import jobs, source, ATLAS,RUN
from benchmark.texture_camera_eval import save
from engine import section_decision as policy
from engine.nearby_face_decision import predict_forest


def sample(job, include_photo=False):
    split,f,*_=job
    with np.load(RUN/'evidence'/split/(f['id']+'.npz'),allow_pickle=False) as z:
        data=(z['points'],z['owner'],z['P'],z['eye'],z['rows'].tolist()) if 'points' in z else None
    obs=policy.observations(data)
    image,parts,camera,prior=source.load_prediction(job)
    T,truth,occupied=source.labels(job,camera['P'])
    xyz=np.array(list(obs))
    world=np.floor((xyz+.5)@T[:3,:3].T+T[:3,3]).astype(int) if len(xyz) else []
    labels=np.array([tuple(v) in truth for v in world],bool)
    X=np.array([o[1] for o in obs.values()]).reshape(-1,30)
    exists=np.array([o[0] for o in obs.values()],bool)
    return f['id'],f['family'],X,exists,labels


def fit(rows):
    fitted={}
    for head,want in [('add',False),('remove',True)]:
        X=np.concatenate([X[e==want] for _,_,X,e,y in rows])
        y=np.concatenate([y[e==want] for _,_,X,e,y in rows])
        fitted[head]=train(X,1-y if want else y,rounds=32,depth=2)
    return fitted


def main(args):
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows=list(ex.map(sample,jobs('train')))
    groups=sorted({r[1] for r in rows})
    totals={k:[] for k in ('add','remove')}
    cutoffs=(.50,.60,.70,.78,.85,.90,.94)
    folds=[]
    for group in groups:
        model=fit([r for r in rows if r[1]!=group])
        result={}
        for head,want in [('add',False),('remove',True)]:
            test=[r for r in rows if r[1]==group]
            X=np.concatenate([X[e==want] for _,_,X,e,y in test])
            y=np.concatenate([y[e==want] for _,_,X,e,y in test])
            target=(~y if want else y)
            prob=predict_forest(model[head],X)
            r=[(int(np.sum((prob>=t)&target)),int(np.sum((prob>=t)&~target))) for t in cutoffs]
            result[head]=r
            totals[head].append(r)
        folds.append(dict(family=group,counts=result))
    chosen={}
    for head in totals:
        a=np.array(totals[head])
        # Resist large numbers of false placements and removal of valid blocks.
        weights=1.5 if head=='add' else 2.
        values=a[:,:,0]-weights*a[:,:,1]
        score=values.sum(axis=0)+2*values.min(axis=0)
        chosen[head+'_threshold']=cutoffs[int(np.argmax(score))]
        print(head,chosen[head+'_threshold'],a.sum(axis=0).tolist(),flush=True)
    model=fit(rows)
    model.update(chosen)
    policy.WEIGHTS.write_text(json.dumps(model,separators=(',',':'))+'\n')
    save(RUN/'loo.json',dict(cutoffs=cutoffs,folds=folds,selected=chosen))


def postprocess(image,parts,counts,camera,prior):
    return policy.recover(image,parts,camera,prior,atlas_path=ATLAS)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=4)
    main(ap.parse_args())

