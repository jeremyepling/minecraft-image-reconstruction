"""Select atlas-registered section changes with a cell-visibility classifier."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import json

import numpy as np

from engine.nearby_face_decision import predict_forest
from engine.section_relocation import sections
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _frontmost_cells, _sample_face

WEIGHTS=Path(__file__).with_name('section-decision-weights.json')
STEPS=((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))


def observations(data):
    if data is None or not data[-1]:
        return {}
    pp,owner,P,eye,hits=data
    points=list(map(tuple,pp.tolist()))
    prior=set(points)
    hits=np.array(hits)
    targets=[tuple(p) for p in hits[:,4:7].astype(int).tolist()]
    by_target,oppose,supporting=defaultdict(list),defaultdict(list),defaultdict(list)
    for j,(old,t) in enumerate(zip(hits[:,2].astype(int),targets)):
        by_target[t].append(j)
        if old>=0:
            (supporting if points[old]==t else oppose)[points[old]].append(j)
    own=np.bincount(owner[owner>=0],minlength=len(prior))
    index=dict(zip(points,range(len(points))))
    candidates=((by_target.keys()-prior) | oppose.keys())
    result={}
    for c in sorted(candidates):
        exists=c in prior
        ids=oppose[c] if exists else by_target[c]
        if len(ids)<3:
            continue
        sample=hits[ids]
        cells=[targets[i] for i in ids]
        old=sample[:,2].astype(int)
        axis=np.bincount(sample[:,3].astype(int),minlength=3)
        primary=int(np.argmax(axis))
        adjacent=[tuple(v+step for v,step in zip(c,delta)) for delta in STEPS]
        proposals=Counter(cells)
        partners=[points[i] for i in old if i>=0] if not exists else cells
        displacement=np.abs(np.array(c)-np.array(partners)).max(1) if partners else [0]
        close=sum(n in prior for n in adjacent)
        neighbor=[len(by_target.get(n,[])) for n in adjacent]
        dist=abs(c[primary]-eye[primary])
        sign=1 if eye[primary]>c[primary] else -1
        face=_sample_face(P,c,primary,sign,owner.shape)
        area=face[1].sum() if face is not None else 0
        years=own[index[c]] if exists else 0
        peak=sample[:,7].max()
        margin=sample[:,8].mean()
        bands=np.unique(sample[:,:2].astype(int)//28,axis=0)
        f=[len(ids),len(supporting[c]),len(oppose[c]),len(by_target.get(c,[])),years,area,
           len(ids)/max(1,years),len(ids)/max(1,area),close,sum(n>=3 for n in neighbor),
           sum(n>=10 for n in neighbor),max(neighbor),len(bands),peak,sample[:,7].mean(),margin,
           sample[:,8].max(),sample[:,9].mean(),sum(sample[:,7]>.85),sum(sample[:,8]>.15),
           axis.max()/len(ids),primary==1,c[1],dist,len(set(partners)),np.median(displacement),
           sum(t in prior for t in cells)/len(ids),sum(old<0)/len(ids),
           np.ptp(sample[:,0]),np.ptp(sample[:,1])]
        result[c]=(exists,np.array(f,float))
    return result


@lru_cache(maxsize=2)
def read_model(path, mtime):
    return json.loads(path.read_bytes())


def choose(obs,models,add,remove):
    decisions=[]
    for exists,head,threshold in ((False,'add',add),(True,'remove',remove)):
        keys=[c for c,(s,_) in obs.items() if s==exists]
        if keys:
            prob=predict_forest(models[head],np.stack([obs[c][1] for c in keys]))
            decisions.append({c for c,p in zip(keys,prob) if p>=threshold})
        else:
            decisions.append(set())
    return decisions


def apply(image,prior,evidence,decisions):
    if evidence is None or not evidence[-1]:
        return prior
    add,remove=decisions
    if not add and not remove:
        return prior
    pp,owner,P,eye,rows=evidence
    src=list(map(tuple,pp.tolist()))
    hits=np.array(rows)
    target=[tuple(p) for p in hits[:,4:7].astype(int).tolist()]
    iy,ix=hits[:,0].astype(int)//4,hits[:,1].astype(int)//4
    old=hits[:,2].astype(int)
    # Monotone rejection of unsupported changes also handles newly restored
    # sources that cover proposed blocks. Every retained addition has three
    # photographic witnesses on the asserted face in the final scene.
    for _ in range(5):
        remain=prior-remove
        allowed=_keep_prior_visible(image.size,P,eye,remain,add)-remain if add else set()
        trial=sorted(remain|allowed)
        rendered,axes=_frontmost_cells(trial,P,eye,owner.shape,face_axes=True)
        ids=rendered[iy,ix]
        ok=[i for i,(c,j,a,b) in enumerate(zip(target,ids,axes[iy,ix],hits[:,3])) if j>=0 and trial[j]==c and a==b]
        placed=Counter(target[i] for i in ok)
        retired=Counter(src[old[i]] for i in ok if old[i]>=0 and src[old[i]]!=target[i])
        next_add={c for c in allowed if placed[c]>=3}
        next_remove={c for c in remove if retired[c]>=3}
        if next_add==add and next_remove==remove:
            return set(trial)
        add,remove=next_add,next_remove
    return prior


def recover(image,parts,camera,prior,*,atlas_path=None):
    if not prior:
        return prior
    try:
        model=read_model(WEIGHTS,WEIGHTS.stat().st_mtime_ns)
    except (KeyError,ValueError,OSError):
        return prior
    data=sections(image,parts,camera,prior,atlas_path=atlas_path)
    obs=observations(data)
    return apply(image,prior,data,choose(obs,model,model['add_threshold'],model['remove_threshold']))
