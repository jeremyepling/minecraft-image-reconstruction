"""Score acceptance cutoffs after actual joint visibility on held-out training families."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np

from benchmark.nearby_face_trial import RUN, PUBLIC, cases, frozen, load_prediction, method, save, read
from benchmark.nearby_face_fit import fold

CONFIGS=[(d,a,r) for d in (2,3) for a,r in ((.50,.35),(.55,.50),(.60,.30),(.65,.15),(.70,.30),(.78,.20),(.85,.15),(.70,0))]


def evaluate(job):
    i,case=job
    data=frozen(case)
    image,parts,camera,prior=load_prediction(case)
    src,tgt,axis=data['edges'].T
    pts=[tuple(p) for p in data['points']]
    labels=dict(zip(pts,data['correct']))
    results=[]
    cache={}
    for d in (2,3):
        with np.load(RUN/f'held-atlas-{d}.npz') as output:
            new,old=output[str(i)]
        for depth,add,drop in CONFIGS:
            if depth!=d:continue
            accept=new>=add
            remove=accept&(old<=drop)&(new-old>=.2)
            decision=set(zip(src[remove],tgt[remove])),set(tgt[accept]),set(zip(tgt[accept],axis[accept]))
            key=tuple(frozenset(s) for s in decision)
            if key not in cache:
                placed=method.apply(image,parts,camera,prior,data,decision)
                plus,minus=placed-prior,prior-placed
                cache[key]=[sum(labels[p] for p in plus),sum(not labels[p] for p in plus),
                            sum(labels[p] for p in minus),sum(not labels[p] for p in minus)]
            results.append(cache[key])
    return results


def main():
    jobs=cases('train')
    with ProcessPoolExecutor(max_workers=6) as pool:
        values=np.array(list(pool.map(evaluate,enumerate(jobs))))
    families=sorted({c[1]['family'] for c in jobs})
    best=None
    result=[]
    for j,(dep,a,d) in enumerate(CONFIGS):
        add,extra,lost,deleted=values[:,j].T
        value=add-1.4*lost-.9*extra+.6*deleted
        totals=[float(sum(v for c,v in zip(jobs,value) if c[1]['family']==f)) for f in families]
        counts=values[:,j].sum(0).tolist()
        row=dict(depth=dep,add=a,drop=d,gained=counts[0],extra_added=counts[1],lost=counts[2],extra_removed=counts[3],
                 family_utility=totals,rank=float(value.sum()+2*min(totals)),
                 families={f:values[[c[1]['family']==f for c in jobs],j].sum(0).tolist() for f in families})
        result.append(row)
        if best is None or row['rank']>best['rank']:
            best=row
    save(PUBLIC/'held-visible.json',dict(variants=result,selected=best,folds=[families[i::3] for i in range(3)]))
    print(best,flush=True)
    model=fold(([],best['depth'],True))
    model.update(atlas=True,drop=best['drop'],add=best['add'],margin=.2)
    import json
    method.MODEL.write_text(json.dumps(model,separators=(',',':'))+'\n',encoding='utf8')


if __name__=='__main__':
    main()

