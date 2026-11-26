"""Fit a regularized thin-face decision rule, holding out entire building families."""
import argparse
from pathlib import Path
from collections import Counter

from benchmark.texture_camera_eval import TRAINING_RELEASE,read,save
from engine.thin_atlas_faces import FEATURES,features
import numpy as np


def fit(X,y):
    center,scale=X.mean(0),np.maximum(.015,X.std(0))
    design=np.column_stack((np.ones(len(X)),(X-center)/scale))
    weights=np.zeros(design.shape[1])
    penalty=np.diag([.001]+[12.]*(len(weights)-1))
    for _ in range(18):
        z=np.clip(design@weights,-30,30)
        prob=1/(1+np.exp(-z))
        hess=design.T@((prob*(1-prob))[:,None]*design)+penalty
        step=np.linalg.solve(hess,design.T@(y-prob)-penalty@weights)
        weights+=step
        if np.max(abs(step))<1e-5:
            break
    return dict(features=list(FEATURES),mean=center.tolist(),scale=scale.tolist(),
                weights=weights[1:].tolist(),bias=float(weights[0]))


def boost(X,y):
    # Shallow additive trees use histogram logits, shrinkage and a minimum
    # leaf size so a rare texture on one training building cannot select a cell.
    thresholds=[np.unique(np.quantile(x,np.linspace(.05,.95,17))) for x in X.T]
    bins=[np.searchsorted(t,x,side='left') for t,x in zip(thresholds,X.T)]
    bias=float(np.log(y.mean()/(1-y.mean())))
    logits=np.full(len(y),bias)
    forest=[]
    for _ in range(45):
        p=1/(1+np.exp(-logits))
        g,h=y-p,p*(1-p)
        def tree(ind,depth):
            G,H=g[ind].sum(),h[ind].sum()
            if depth and len(ind)>=120:
                best=(0,None,None)
                for k,b in enumerate(bins):
                    args=b[ind]
                    size=len(thresholds[k])+1
                    n=np.bincount(args,minlength=size).cumsum()[:-1]
                    grad=np.bincount(args,weights=g[ind],minlength=size).cumsum()[:-1]
                    hes=np.bincount(args,weights=h[ind],minlength=size).cumsum()[:-1]
                    gain=grad**2/(hes+6)+(G-grad)**2/(H-hes+6)-G**2/(H+6)
                    gain[(n<45)|((len(ind)-n)<45)]=-np.inf
                    j=int(np.argmax(gain))
                    if gain[j]>best[0]:
                        best=(gain[j],k,j)
                gain,k,j=best
                if gain>1:
                    lower=bins[k][ind]<=j
                    return [k,float(thresholds[k][j]),tree(ind[lower],depth-1),tree(ind[~lower],depth-1)]
            update=float(.18*np.clip(G/(H+6),-3.5,3.5))
            logits[ind]+=update
            return update
        forest.append(tree(np.arange(len(y)),2))
    return dict(features=list(FEATURES),bias=bias,forest=forest)


def main(args):
    files=sorted((args.source/'train').glob('*/trace.json'))
    admitted = {r['id'] for r in read(TRAINING_RELEASE)['frames'] if r['split'] == 'train'}
    if {p.parent.name for p in files} != admitted:
        raise ValueError('Training traces must match the whole-structure training split')
    X,ys,groups,eligible,candidates=[],[],[],[],Counter()
    for path in files:
        group=path.parent.name.split('-')[1].rstrip('0123456789')
        rows={tuple(r['cell']):r for r in read(path)}
        for c,r in rows.items():
            X.append(features(c,r,rows))
            ys.append(r['target'])
            groups.append(group)
            eligible.append(r['split']>=.25 and r['phase']>.02 and r['texels']>=8)
            candidates[group]+=int(r['target'])
    X,y,groups=np.asarray(X),np.asarray(ys),np.asarray(groups)
    eligible=np.asarray(eligible)
    oof=np.zeros(len(y))
    from engine.thin_atlas_faces import model_probabilities
    train=boost if args.boost else fit
    for group in sorted(set(groups)):
        hold=groups==group
        model=train(X[~hold],y[~hold])
        oof[hold]=model_probabilities(X[hold],model)
    rows={}
    for threshold in (.5,.65,.75,.85,.92):
        take=(oof>=threshold)&eligible
        by_group={}
        for g in sorted(set(groups)):
            a=take&(groups==g)
            by_group[g]=[int((a&y).sum()),int((a&~y).sum())]
        rows[str(threshold)]=by_group
        print(threshold,int((take&y).sum()),int((take&~y).sum()),by_group)
    model=train(X,y)
    save(args.weights,model)
    save(args.report,dict(groups=sorted(set(groups)),train_frames=len(files),labelled_candidates=len(y),
                          candidate_headroom=dict(candidates),group_out_predictions=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--boost',action='store_true')
    main(p.parse_args())
