"""Compare neighborhood agreement on probabilities from six family-held folds."""
from itertools import product

import numpy as np
from benchmark.grounded_depth_round import CACHE, REPORT, jobs, stats, train, predict_forest, save


from engine.grounded_depth_envelope import smooth


def main():
    records=jobs('train')
    rows=[dict(np.load(CACHE/'train'/(j[1]['id']+'.npz'))) for j in records]
    X=np.concatenate([r['features'] for r in rows])
    y=np.concatenate([r['false'] for r in rows])
    h=np.concatenate([r['hidden'] for r in rows])
    lengths=[len(r['false']) for r in rows]
    groups=np.repeat([j[1]['family'] for j in records],lengths)
    prob=np.zeros(len(y))
    for hold in sorted(set(groups)):
        left=groups!=hold
        model=train(X[left],y[left],rounds=72,depth=3)
        prob[~left]=predict_forest(model,X[~left])
    results={}
    for hor in [False,True]:
        q=np.concatenate([smooth(row['points'],part,horizontal_only=hor)
                          for row,part in zip(rows,np.split(prob,np.cumsum(lengths)[:-1]))])
        for weight,t in product((.15,.3,.5,.7),(.65,.7,.75,.8,.82,.85,.90)):
            reject=(1-weight)*prob+weight*q>=t
            results[f'{int(hor)}-{weight}-{t}']=stats(reject,y,h,groups)
    selected={k:v for k,v in results.items() if v['removed_extras']>=443 and v['lost_correct']<=36
              and all(a>=2*b for a,b in v['family'].values())}
    save(REPORT/'component-train.json',dict(results=results,dominating=selected))
    print([(k,v['removed_extras'],v['lost_correct']) for k,v in selected.items()])
    print(sorted([(v['removed_extras']-3*v['lost_correct'],k,v['removed_extras'],v['lost_correct'])
                  for k,v in results.items()],reverse=True)[:6])


if __name__=='__main__':
    main()
