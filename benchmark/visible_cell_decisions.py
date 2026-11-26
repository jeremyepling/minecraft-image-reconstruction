"""Train and evaluate cell abstention rules with whole-building-family cross-validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from functools import lru_cache
from unittest.mock import patch

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MCIR_IMAGE_ONLY='1')
import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.mask_blocks import _write_read, registered
from benchmark.texture_camera_eval import STABLE, checked
from engine import visible_placement
from engine.visible_cell_decision import FEATURES as CELL_FEATURES, cell_features
from engine.visible_floor_evidence import FEATURES as FLOOR_FEATURES, floor_features
from engine.visible_ray_filter import fully_occluded_cells

FEATURES=CELL_FEATURES+FLOOR_FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA = STABLE/'runs/visible-builds-v1/data'
CACHE = STABLE/'runs/visible-depth-rounds/inputs'
OUT = ROOT/'runs/cell-decisions'
MODEL = ROOT/'benchmark/reviews/visible-cell-decisions-01/model.json'


def read(path):
    return json.loads(path.read_bytes())


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=1,allow_nan=False)+'\n')


def write_samples(path,**arrays):
    staging=path.with_suffix('.partial')
    with staging.open('wb') as file:
        np.savez_compressed(file,**arrays)
    staging.replace(path)


def baseline(image,version,parts,counts,camera):
    with patch.object(visible_placement,'seam_camera',return_value=camera):
        return visible_placement.place(image,version,parts,counts=counts)


def features(path):
    source=read(path)
    key=source['id']
    d=DATA/source['split']/'frames'/key
    with Image.open(checked(d/'rgb.png',source['rgb_sha256'])) as img:
        rgb=img.convert('RGB')
    camera=source['camera']
    start=time.perf_counter()
    cells,_=baseline(rgb,'26.3',source['parts'],source['counts'],camera)
    elapsed_base=time.perf_counter()-start
    start=time.perf_counter()
    xyz,values,available=cell_features(rgb,source['parts'],camera['P'],cells)
    elapsed_features=time.perf_counter()-start
    # Open the source world and pose only after extracting each image's features.
    target={tuple(b['xyz_world']) for b in read(d/'labels.json')['blocks']}
    world=read(d/'world-readback.json')
    ground_y=read(d/'frame.json')['ground_y']
    occupied={(b['x'],b['y'],b['z']) for b in world if b['block_id'].removeprefix('minecraft:')!='air' and b['y']>ground_y}
    if len(xyz):
        T=registration(np.asarray(camera['P']),read(d/'camera.json'))
        mapped=np.floor((xyz+.5) @ T[:3,:3].T+T[:3,3]).astype(int)
    else:
        mapped=xyz
    visible=np.array([tuple(cell) in target for cell in mapped],dtype=bool)
    hidden=np.array([tuple(cell) in occupied and tuple(cell) not in target for cell in mapped],dtype=bool)
    dest=OUT/'features'/f'{key}.npz'
    write_samples(dest,xyz=xyz,features=values,y=visible,hidden=hidden,target=len(target),
                        atlas=available,family=source['family'],split=source['split'],layout=source['layout'],
                        rgb=source['rgb_sha256'],input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        baseline_seconds=elapsed_base,feature_seconds=elapsed_features)
    return key,len(xyz)


def prepare(args):
    (OUT/'features').mkdir(parents=True,exist_ok=True)
    allowed={r['id'] for r in read(ROOT/'training/visible-builds-v1.json')['frames']
             if r['split'] in args.splits}
    while True:
        todo=[path for path in sorted(CACHE.glob('*.json'))
              if path.stem in allowed and not (OUT/'features'/f'{path.stem}.npz').exists()]
        if todo:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for key,n in pool.map(features,todo):
                    print(key,n,flush=True)
        remaining=sum(not (OUT/'features'/f'{key}.npz').exists() for key in allowed)
        if not args.follow or remaining==0:
            return
        time.sleep(4)


def floor_measure(path):
    with np.load(path) as z:
        xyz=z['xyz']
        split=str(z['split'])
    source=read(CACHE/f'{path.stem}.json')
    with Image.open(checked(DATA/split/'frames'/path.stem/'rgb.png',source['rgb_sha256'])) as img:
        t=time.perf_counter()
        X=floor_features(img,source['camera']['P'],xyz)
        elapsed=time.perf_counter()-t
    write_samples(OUT/'floor'/path.name,x=X,seconds=elapsed)


def floor_prepare(args):
    (OUT/'floor').mkdir(parents=True,exist_ok=True)
    allowed={r['id'] for r in read(ROOT/'training/visible-builds-v1.json')['frames']
             if r['split'] in args.splits}
    while True:
        todo=[path for path in sorted((OUT/'features').glob('*.npz'))
              if path.stem in allowed and not (OUT/'floor'/path.name).exists()]
        if todo:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(floor_measure,todo))
        if not args.follow or all((OUT/'floor'/f'{k}.npz').exists() for k in allowed):
            return
        time.sleep(4)


def ray_measure(path):
    with np.load(path) as z:
        xyz=z['xyz']
        split=str(z['split'])
    source=read(CACHE/f'{path.stem}.json')
    with Image.open(checked(DATA/split/'frames'/path.stem/'rgb.png',source['rgb_sha256'])) as img:
        t=time.perf_counter()
        occluded=fully_occluded_cells(img.size,source['camera']['P'],set(map(tuple,xyz)))
        elapsed=time.perf_counter()-t
    write_samples(OUT/'rays'/path.name,drop=np.array([tuple(p) in occluded for p in xyz],bool),
                        seconds=elapsed)


def ray_prepare(args):
    (OUT/'rays').mkdir(parents=True,exist_ok=True)
    allowed={r['id'] for r in read(ROOT/'training/visible-builds-v1.json')['frames'] if r['split'] in args.splits}
    while True:
        todo=[path for path in sorted((OUT/'features').glob('*.npz'))
              if path.stem in allowed and not (OUT/'rays'/path.name).exists()]
        if todo:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(ray_measure,todo))
        if not args.follow or all((OUT/'rays'/f'{k}.npz').exists() for k in allowed):
            return
        time.sleep(4)


def feature_rows(splits):
    rows=[]
    for file in sorted((OUT/'features').glob('*.npz')):
        with np.load(file) as z:
            split=str(z['split'])
            if split in splits:
                if not (OUT/'floor'/file.name).exists():
                    continue
                with np.load(OUT/'floor'/file.name) as f:
                    x=np.column_stack((z['features'],f['x']))
                    seconds=float(f['seconds'])
                rows.append(dict(id=file.stem,split=split,family=str(z['family']),layout=str(z['layout']),
                                 xyz=z['xyz'],x=x,y=z['y'],hidden=z['hidden'],target=int(z['target']),
                                 rgb=str(z['rgb']),input_sha256=str(z['input_sha256']),seconds=seconds+float(z['feature_seconds'])))
    return rows


def arrays(rows):
    return np.vstack([r['x'] for r in rows]),np.concatenate([r['y'] for r in rows])


def fit_tree(X,Y,depth,min_leaf):
    """Small CART using binary Gini reduction and development quantiles."""
    count=len(Y)
    probability=float((Y.sum()+1)/(count+2))
    leaf=dict(probability=probability,n=count)
    if depth==0 or count<2*min_leaf or np.all(Y==Y[0]):
        return leaf
    best=0
    choice=None
    for col in range(X.shape[1]):
        values=X[:,col]
        for threshold in np.unique(np.quantile(values,np.linspace(.04,.96,16))):
            left=values<=threshold
            a=left.sum()
            b=count-a
            if min(a,b)<min_leaf:
                continue
            p=Y[left].mean()
            q=Y[~left].mean()
            gain=a*b/count*(p-q)**2
            if gain>best:
                best=gain
                choice=(col,float(threshold),left)
    if choice is None:
        return leaf
    col,v,left=choice
    return dict(feature=FEATURES[col],value=v,left=fit_tree(X[left],Y[left],depth-1,min_leaf),
                right=fit_tree(X[~left],Y[~left],depth-1,min_leaf))


def probabilities(tree,X):
    if 'probability' in tree:
        return np.full(len(X),tree['probability'])
    mask=X[:,FEATURES.index(tree['feature'])]<=tree['value']
    out=np.empty(len(X))
    out[mask]=probabilities(tree['left'],X[mask])
    out[~mask]=probabilities(tree['right'],X[~mask])
    return out


def fit_rule(X,Y,families=None):
    best=(0,None)
    for i,col in enumerate(FEATURES):
        supported=X[:,i][X[:,i]>0]
        if not len(supported):
            continue
        thresholds=np.r_[np.quantile(X[:,i],np.linspace(.025,.975,39)),
                         np.quantile(supported,[0,.002,.005,.01,.025,.05])]
        for t in np.unique(thresholds):
            for sense in (-1,1):
                reject=X[:,i]*sense<t*sense
                removed=(~Y & reject).sum()
                lost=(Y & reject).sum()
                score=removed-5*lost
                if score<=best[0] or lost>.03*Y.sum():
                    continue
                if families is not None and any(np.sum(reject&Y&group)>.005*np.sum(Y&group)
                         or np.sum(reject&~Y&group)<4*np.sum(reject&Y&group) for group in families):
                    continue
                best=(score,dict(feature=col,value=float(t),sense=sense))
    return best[1]


def fit_conjunctions(rows):
    """Mine short rejection rules under separate loss limits in each training family."""
    X,Y=arrays(rows)
    family=np.concatenate([np.repeat(r['family'],len(r['y'])) for r in rows])
    bits=lambda group:int.from_bytes(np.packbits(group,bitorder='little').tobytes(),'little')
    bad,good=bits(~Y),bits(Y)
    budgets=[]
    for g in np.unique(family):
        valid=family==g
        true=bits(valid & Y)
        budgets.append((true,bits(valid & ~Y),max(1,int(.01*true.bit_count()))))
    conditions=[]
    seen=set()
    for i,name in enumerate(FEATURES):
        v=X[:,i]
        ts=np.unique(np.quantile(v,[.025,.05,.10,.20,.35,.50,.65,.80,.90,.95,.975]))
        for threshold in ts:
            for sense in (-1,1):
                cover=bits(v*sense<threshold*sense)
                if cover.bit_count()>=20 and cover not in seen:
                    seen.add(cover)
                    conditions.append((cover,dict(feature=name,value=float(threshold),sense=sense)))
    clauses=[]
    pruned=0
    for _ in range(2):
        best_score=0
        best=None
        for a,(cover,first) in enumerate(conditions):
            for other,second in conditions[a:]:
                if first['feature']==second['feature'] and first!=second:
                    continue
                pair=cover & other
                new=pair & ~pruned
                score=(new&bad).bit_count()-6*(new&good).bit_count()
                if score<=best_score:
                    continue
                combined=pruned | pair
                costs=[((combined&t).bit_count(),(combined&f).bit_count(),budget) for t,f,budget in budgets]
                if any(loss>budget or gain<3*loss for loss,gain,budget in costs):
                    continue
                # An entire family can lack the trial failure. The evidence must still
                # cover at least half the independently grouped training families.
                if sum((new&f).bit_count()>=4 for _,f,_ in budgets)<(len(budgets)+1)//2:
                    continue
                best_score=score
                best=(pair,[first] if first==second else [first,second])
        if best is None:
            break
        pruned |= best[0]
        clauses.append(best[1])
    return dict(kind='conjunctions',clauses=clauses)


def rejects(model,X):
    if model['kind']=='conjunctions':
        mask=np.zeros(len(X),bool)
        for clause in model['clauses']:
            match=np.ones(len(X),bool)
            for r in clause:
                match &= X[:,FEATURES.index(r['feature'])]*r['sense']<r['value']*r['sense']
            mask |= match
        return mask
    if model['kind'] in ('rule','family_rule'):
        r=model['rule']
        return np.zeros(len(X),bool) if r is None else X[:,FEATURES.index(r['feature'])]*r['sense']<r['value']*r['sense']
    votes=np.stack([probabilities(t,X) for t in model['trees']])
    confidence=votes.max(0) if model['kind']=='consensus' else votes.mean(0)
    return confidence<model['cutoff']


@lru_cache(maxsize=1)
def weights():
    return read(MODEL)['models']


def after_placement(image,parts,camera,cells,model,atlas_path=None):
    """Filter the actual cells supplied by any placement route, in their local grid."""
    if not cells or camera['P'] is None:
        return set(cells)
    if model=='occluded':
        return set(cells)-fully_occluded_cells(image.size,camera['P'],cells)
    xyz,X,atlas=cell_features(image,parts,camera['P'],cells,atlas_path=atlas_path)
    if not atlas:
        return set(cells)
    floors=floor_features(image,camera['P'],xyz)
    drop=rejects(weights()[model],np.column_stack((X,floors)))
    return set(map(tuple,xyz[~drop].tolist()))


def _predict(image,version,parts,counts,camera,model):
    cells,status=baseline(image,version,parts,counts,camera)
    result=after_placement(image,parts,camera,cells,model)
    return result,dict(status,removed_by_decision=len(cells)-len(result))


def predict_rule(image,version,parts,counts,camera):
    return _predict(image,version,parts,counts,camera,'rule')


def predict_tree(image,version,parts,counts,camera):
    return _predict(image,version,parts,counts,camera,'tree')


def predict_exported_tree(image,version,parts,counts,camera):
    from engine.visible_evidence_filter import filter_cells
    cells,status=baseline(image,version,parts,counts,camera)
    return filter_cells(image,parts,camera,cells),status


def predict_consensus(image,version,parts,counts,camera):
    return _predict(image,version,parts,counts,camera,'consensus')


def predict_family_rule(image,version,parts,counts,camera):
    return _predict(image,version,parts,counts,camera,'family_rule')


def predict_conjunctions(image,version,parts,counts,camera):
    return _predict(image,version,parts,counts,camera,'conjunctions')


def predict_occluded(image,version,parts,counts,camera):
    return _predict(image,version,parts,counts,camera,'occluded')


def fit(rows,kind,cutoff=.22):
    if kind=='conjunctions':
        return fit_conjunctions(rows)
    X,Y=arrays(rows)
    if kind in ('rule','family_rule'):
        groups=None
        if kind=='family_rule':
            group=np.concatenate([np.repeat(r['family'],len(r['y'])) for r in rows])
            groups=[group==g for g in np.unique(group)]
        return dict(kind=kind,rule=fit_rule(X,Y,groups))
    if kind=='tree':
        trees=[fit_tree(X,Y,4,max(35,len(X)//120))]
    else:
        trees=[]
        for g in sorted({r['family'] for r in rows}):
            x,y=arrays([r for r in rows if r['family']!=g])
            trees.append(fit_tree(x,y,4,max(35,len(x)//120)))
    return dict(kind=kind,trees=trees,cutoff=cutoff)


def tally(row,mask):
    Y=row['y']
    return dict(correct=int(Y.sum()),extra=int((~Y).sum()),hidden=int(row['hidden'].sum()),target=row['target'],
                lost=int(np.sum(mask&Y)),removed=int(np.sum(mask&~Y)),hidden_removed=int(np.sum(mask&row['hidden'])))


def totals(items):
    result={k:sum(x[k] for x in items) for k in ('correct','extra','hidden','target','lost','removed','hidden_removed')}
    c=result['correct']-result['lost']
    e=result['extra']-result['removed']
    result.update(remaining_correct=c,remaining_extra=e,precision=c/max(1,c+e),recall=c/max(1,result['target']),
                  f1=2*c/max(1,c+e+result['target']))
    return result


def train():
    rows=feature_rows({'train'})
    groups=sorted({r['family'] for r in rows})
    if len(rows)!=72 or len(groups)!=6:
        raise ValueError('Require all 72 training frames from six disjoint families.')
    cv={}
    models={}
    for kind in ('rule','tree','consensus','family_rule','conjunctions'):
        held=[]
        for g in groups:
            others=[r for r in rows if r['family']!=g]
            model=fit(others,kind)
            for r in (r for r in rows if r['family']==g):
                held.append((r,model))
        # Set threshold on leave-family-out predictions. Require low cost in every family.
        choices=[]
        thresholds=[.22] if kind in ('rule','family_rule','conjunctions') else [.08,.12,.16,.20,.24,.28,.32,.36]
        for t in thresholds:
            by_family={g:[] for g in groups}
            for row,model in held:
                removed=rejects(dict(model,cutoff=t),row['x'])
                by_family[row['family']].append(tally(row,removed))
            agg={g:totals(v) for g,v in by_family.items()}
            total=totals(list(agg.values()))
            safe=all(m['lost']<=.03*m['correct'] and m['removed']>=2*m['lost'] for m in agg.values())
            choices.append(dict(cutoff=t,families=agg,total=total,safe=safe))
        safe=[c for c in choices if c['safe']]
        # When no threshold qualifies, retain the least aggressive as a documented experiment.
        choice=max(safe,key=lambda c:c['total']['removed']-4*c['total']['lost']) if safe else choices[0]
        models[kind]=fit(rows,kind,choice['cutoff'])
        cv[kind]=dict(choices=choices,selected=choice)
    eligible=[kind for kind,v in cv.items() if v['selected']['safe']
              and v['selected']['total']['lost']<=.01*v['selected']['total']['correct']
              and v['selected']['total']['removed']>=5*v['selected']['total']['lost']]
    preferred=max(eligible,key=lambda k:cv[k]['selected']['total']['removed']-5*cv[k]['selected']['total']['lost']) if eligible else None
    write(MODEL,dict(features=FEATURES,models=models,preferred=preferred))
    write(MODEL.parent/'training.json',dict(families=groups,frames=len(rows),
          raw=totals([tally(r,np.zeros(len(r['y']),bool)) for r in rows]),cross_validation=cv))
    print({k:(v['selected']['safe'],v['selected']['cutoff'],v['selected']['total']) for k,v in cv.items()})


def evaluate(args):
    models=read(MODEL)['models']
    models['occluded']=None
    summary={}
    cases=[]
    for split in args.splits:
        rows=feature_rows({split})
        if len(rows)!={'train':72,'validation':12,'test':12,'regression':56}[split]:
            raise ValueError(f'Incomplete {split}')
        for k,m in models.items():
            summaries=[]
            for row in rows:
                if k=='occluded':
                    with np.load(OUT/'rays'/f"{row['id']}.npz") as z:
                        mask=z['drop']
                else:
                    mask=rejects(m,row['x'])
                cells=set(map(tuple,row['xyz'][~mask].tolist()))
                pack=OUT/'packs'/k/row['id']/'reconstruction.zip'
                pack.parent.mkdir(parents=True,exist_ok=True)
                if _write_read(pack,cells)!=cells:
                    raise ValueError('Pack mismatch')
                item=tally(row,mask)
                summaries.append(item)
                cases.append(dict(id=row['id'],family=row['family'],method=k,split=split,rgb=row['rgb'],
                                  inputs=row['input_sha256'],pack=hashlib.sha256(pack.read_bytes()).hexdigest(),
                                  rejected=row['xyz'][mask].tolist(),feature_seconds=row['seconds'],**item))
            summary.setdefault(split,{})[k]=totals(summaries)
    write(MODEL.parent/'evaluation.json',dict(model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),summary=summary))
    write(OUT/'evaluation-cases.json',cases)
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','floor','rays','train','evaluate'])
    p.add_argument('--splits',nargs='+',default=['train'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--follow',action='store_true')
    a=p.parse_args()
    if a.stage=='prepare':
        prepare(a)
    elif a.stage=='floor':
        floor_prepare(a)
    elif a.stage=='rays':
        ray_prepare(a)
    elif a.stage=='train':
        train()
    else:
        evaluate(a)


if __name__=='__main__':
    main()
