"""Group-held-out boosting for nearby visible face and depth decisions."""
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np

from benchmark.nearby_face_trial import RUN, PUBLIC, cases, frozen, art_sample, method, save


def train(X,y,rounds=36,depth=3):
    # Shallow logistic gradient boosting with 16 quantile split candidates
    # per feature and a minimum of 24 training rays per leaf.
    cuts=np.quantile(X,np.arange(1,16)/16,axis=0).T.astype('float32')
    bins=np.stack([np.searchsorted(row,col,side='left') for row,col in zip(cuts,X.T)])
    bins=bins.astype(np.int32)
    start=float(np.log(y.mean()/(1-y.mean())))
    logits=np.full(len(y),start)
    trees=[]
    for _ in range(rounds):
        prob=1/(1+np.exp(-logits))
        grad=y-prob
        weight=prob*(1-prob)
        nodes=np.zeros(len(y),int)
        levels=[]
        for layer in range(depth):
            size=2**layer
            best=np.full(size,-np.inf)
            idx=np.zeros(size,int)
            thresholds=np.zeros(size,int)
            spans=np.arange(16)[None,:]
            for i,b in enumerate(bins):
                ids=nodes*16+b
                g=np.bincount(ids,grad,minlength=size*16).reshape(size,16).cumsum(1)
                h=np.bincount(ids,weight,minlength=size*16).reshape(size,16).cumsum(1)
                n=np.bincount(ids,minlength=size*16).reshape(size,16).cumsum(1)
                gain=g*g/(h+2)+(g[:,-1:]-g)**2/(h[:,-1:]-h+2)
                gain[(n<24)|(n[:,-1:]-n<24)|(spans==15)]=-np.inf
                t=gain.argmax(1)
                vals=gain[np.arange(size),t]
                improved=vals>best
                idx[improved],thresholds[improved],best[improved]=i,t[improved],vals[improved]
            values=cuts[idx,thresholds]
            levels.append([[int(i),float(t)] for i,t in zip(idx,values)])
            nodes=nodes*2+(bins[idx[nodes],np.arange(len(y))]>thresholds[nodes])
        n=2**depth
        update=.18*np.bincount(nodes,grad,minlength=n)/(2+np.bincount(nodes,weight,minlength=n))
        logits+=update[nodes]
        trees.append(dict(levels=levels,leaves=update.tolist()))
    return dict(bias=start,trees=trees)


def fold(job):
    held,depth,atlas=job
    items=[(f[1]['family'],(art_sample if atlas else frozen)(f)) for f in cases('train')]
    features=[]
    source,dest=[],[]
    for family,p in items:
        if family in held:continue
        edges=p['edges']
        features.append(p['features'])
        source.extend(p['correct'][edges[:,0]])
        dest.extend(p['correct'][edges[:,1]])
    X=np.concatenate(features)
    # Use the same reproducible row sample for both classifier heads.
    rng=np.random.default_rng(71)
    take=rng.choice(len(X),min(len(X),48000),replace=False)
    X=X[take]
    model=dict(source=train(X,np.asarray(source)[take],depth=depth),
               destination=train(X,np.asarray(dest)[take],depth=depth))
    if not held:
        return model
    probs=[]
    for i,(family,p) in enumerate(items):
        if family not in held:continue
        x=p['features']
        probs.append((i,method.predict_forest(model['destination'],x),method.predict_forest(model['source'],x)))
    return probs


def changes(data,probs,add,drop,margin):
    s,t,_=data['edges'].T
    new,old=probs
    accepted=new>=add
    proposed=np.unique(t[accepted])
    removed=np.unique(s[accepted & (old<=drop) & ((new-old)>=margin)])
    truth=data['correct']
    return np.array([truth[proposed].sum(),(~truth[proposed]).sum(),truth[removed].sum(),(~truth[removed]).sum()])


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',action='store_true')
    args=parser.parse_args()
    items=[(f[1]['family'],frozen(f)) for f in cases('train')]
    families=sorted({k for k,p in items})
    groups=[families[i::3] for i in range(3)]
    configs=[(add,drop,.2) for add,drop in product((.50,.60,.70,.78,.85,.92),(0,.10,.25,.4))]
    best=None
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for depth in (2,3):
            results=pool.map(fold,[(g,depth,args.atlas) for g in groups])
            predictions={i:(new,old) for result in results for i,new,old in result}
            prefix='atlas' if args.atlas else 'photo'
            np.savez_compressed(RUN/f'held-{prefix}-{depth}.npz',**{str(i):np.stack(pair) for i,pair in predictions.items()})
            for add,drop,margin in configs:
                counts=np.array([changes(p,predictions[i],add,drop,margin) for i,(_,p) in enumerate(items)])
                gained,extras,lost,removed=counts.T
                # Choose on held-family performance. A loss that trades away an
                # already correct cube is slightly more costly than an abstention.
                score=gained-1.4*lost-.9*extras+.6*removed
                per=[float(sum(score[i] for i,(name,_) in enumerate(items) if name==g)) for g in families]
                rank=float(score.sum()+2*min(per))
                total=counts.sum(0)
                row=dict(depth=depth,add=add,drop=drop,margin=margin,gained=int(total[0]),extra_added=int(total[1]),
                         lost=int(total[2]),extra_removed=int(total[3]),family_utility=per,rank=rank)
                rows.append(row)
                if best is None or rank>best['rank']:
                    best=row
            print('best so far',best,flush=True)
    model=fold(([],best['depth'],args.atlas))
    model['atlas']=args.atlas
    model.update({k:best[k] for k in ('drop','add','margin')})
    method.MODEL.write_text(json.dumps(model,separators=(',',':'))+'\n',encoding='utf8')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    oracle=sum(int(p['correct'][int(p['base_count']):].sum()) for _,p in items)
    save(PUBLIC/'training.json',dict(selected=best,families=families,held_groups=groups,oracle_additions=oracle,variants=rows))
    print('oracle proposals',oracle, 'frozen',best,flush=True)


if __name__=='__main__':
    main()

