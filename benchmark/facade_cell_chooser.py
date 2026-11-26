"""Learn to accept or abstain on individually proposed visible cells from measured placement alternatives."""
import argparse
from pathlib import Path

import numpy as np

from benchmark import facade_placement_eval as replay, framed_depth_eval as framed
from benchmark.facade_placement_chooser import predictions
from benchmark.mask_blocks import decoded_pack
from benchmark.nearby_thin_comparison import EVALUATION
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import checked, save
from benchmark.visible_recovery_report import summarize


def features(cell,prior,votes):
    x,y,z=cell
    neighbors=[(x+1,y,z),(x-1,y,z),(x,y,z+1),(x,y,z-1)]
    horizontal=[p in prior for p in neighbors]
    return [*votes,sum(votes),sum(horizontal),int(any(horizontal[:2]) and any(horizontal[2:])),
            int((x,y-1,z) in prior),int((x,y+1,z) in prior)]


def tree(rows,depth=2):
    # Zero prior on unsupported branches. A false addition costs five recovered positions during fitting.
    value=sum(1 if correct else -5 for _,correct in rows)
    best={'accept':value>0}
    utility=max(0,value)
    if not depth or len(rows)<12:
        return best,utility
    for feature in range(len(rows[0][0])):
        values=sorted({x[feature] for x,_ in rows})
        for threshold in values[:-1]:
            low=[r for r in rows if r[0][feature]<=threshold]
            high=[r for r in rows if r[0][feature]>threshold]
            if min(len(low),len(high))<6:
                continue
            left,a=tree(low,depth-1)
            right,b=tree(high,depth-1)
            if a+b>utility:
                best=dict(feature=feature,threshold=threshold,left=left,right=right)
                utility=a+b
    return best,utility


def choose(model,x):
    while 'accept' not in model:
        model=model['left'] if x[model['feature']]<=model['threshold'] else model['right']
    return model['accept']


def observations(methods):
    loaded,manifests={},{}
    for name,folders in methods:
        loaded[name],manifests[name]=predictions(folders)
    rows=[]
    for split,frame,prior in replay.cases(tuple(framed.SIZES)):
        key=split,frame['id']
        outputs=[]
        for name in loaded:
            r,path=loaded[name][key]
            assert r['rgb_sha256']==prior['rgb_sha256'] and r['input_hashes']==prior['input_hashes']
            outputs.append(set(decoded_pack(checked(path,r['pack_sha256']))))
        baseline,*alts=outputs
        proposals=set.union(*alts)-baseline
        cells=sorted(proposals)
        xx=[features(c,baseline,[int(c in alternative) for alternative in alts]) for c in cells]
        # Labels, historical alignment and registration are used only after features are frozen.
        ref_split='old' if split in ('new','extra') else split
        builds=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
        truth,occupied,T,_=framed.references(ref_split,frame,framed.NATIVE_INPUTS.get(split),framed.TRAINING_DATA,
                                             framed.DEPTH,dict(native=framed.native.BUILDS,old=builds,fresh=builds))
        shift=np.asarray(prior['translation'])
        if cells:
            assert T is not None
            mapped=np.floor((np.asarray(cells)+.5)@T[:3,:3].T+T[:3,3]).astype(int)
        else:
            mapped=np.empty((0,3),int)
        yy=[tuple(c) in truth for c in mapped+shift]
        world=framed.registered(baseline,T)
        assert len({tuple(p+shift) for p in world}&truth)==prior['relative_geometry']['correct']
        rows.append(dict(id=key[1],split=split,family=frame.get('family'),baseline=world,truth=truth,occupied=occupied,
                         xx=xx,yy=yy,mapped=mapped,prior=prior['relative_geometry']))
    return rows,manifests


def fit(rows):
    examples=[(x,y) for r in rows for x,y in zip(r['xx'],r['yy'],strict=True)]
    return tree(examples)[0]


def evaluate(rows,select):
    results=[]
    for r in rows:
        keep=[tuple(c) for c,x,y in zip(r['mapped'],r['xx'],r['yy'],strict=True) if select(r,x,y)]
        prior=r['baseline']
        if keep:
            metrics=placement_scores(prior|set(keep),prior,dict.fromkeys(r['truth']),r['occupied'])['relative_geometry']
        else:
            metrics=dict(r['prior'],gained=0,lost=0,extra_added=0,extra_removed=0,hidden_added=0,hidden_removed=0)
        results.append(dict(relative_geometry=metrics))
    return summarize(results)


def run(methods,out):
    rows,manifests=observations(methods)
    train=[r for r in rows if r['split']=='train']
    assert len(train)==72
    model=fit(train)
    folds={fam:fit([r for r in train if r['family']!=fam]) for fam in sorted({r['family'] for r in train})}
    groups=dict(train=train,fixed=[r for r in rows if r['split'] in EVALUATION],
                extra=[r for r in rows if r['split']=='extra'],new=[r for r in rows if r['split']=='new'])
    stats={}
    for name,rr in groups.items():
        stats[name]=dict(learned=evaluate(rr,lambda r,x,y:choose(model,x)),all=evaluate(rr,lambda r,x,y:True),
                         perfect_additions_at_existing_alignment=evaluate(rr,lambda r,x,y:y))
    stats['train']['family_held_out']=evaluate(train,lambda r,x,y:choose(folds[r['family']],x))
    save(out,dict(input_hashes=manifests,features=[*(name for name,_ in methods[1:]),'votes','horizontal_neighbors',
             'both_horizontal_axes','supported_below','supported_above'],model=model,fold_models=folds,groups=stats))
    print({k:{n:round(v['f1'],5) for n,v in vals.items()} for k,vals in stats.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',nargs='+',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run([(name,list(map(Path,folders))) for name,*folders in a.method],a.out)
