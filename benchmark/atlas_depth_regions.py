"""Check secondary atlas-supported depth fragments within a single merged mask."""
from unittest.mock import patch

import numpy as np

from engine import visible_placement as placer
from engine.atlas_plane_depth import _coefficients,_patch_scores,frozen_weights,options
from engine.camera_atlas_floor import _atlas
from engine.plane_offset_features import ranked_choice


def coordinates(uv,axis,level,eye):
    xyz=np.empty((len(uv),3),dtype=int)
    xyz[:,axis]=level-int(eye[axis]>level)
    xyz[:,[a for a in range(3) if a!=axis]]=uv
    return list(map(tuple,xyz.tolist()))


def adjust(cells,mask,P,eye,rays,axis,primary,rgb,atlas):
    free=[a for a in range(3) if a!=axis]
    coeff=_coefficients(rays[mask],axis,free)
    if len(coeff)<40 or not np.isfinite(coeff).all():
        return cells,0
    def sample(level):
        uv=eye[free]+coeff*(level-eye[axis])
        keys,inv,counts=np.unique(np.floor(uv).astype(int),axis=0,return_inverse=True,return_counts=True)
        scores=_patch_scores(mask,keys,axis,free,level,P,rgb,atlas,by_cell=True)
        return keys,inv.ravel(),counts,dict(scores)
    original,original_idx,total,main_scores=sample(primary)
    if sum(q>.65 for q in main_scores.values())<2:
        return cells,0
    base_scores=np.array([main_scores.get(tuple(k),0.) for k in original])[original_idx]
    best=np.zeros(len(coeff))
    owner=np.full(len(coeff),-1)
    winners=[]
    for level in (primary-2,primary-1,primary+1,primary+2):
        if (level-eye[axis])*(primary-eye[axis])<=0 or (axis==1 and level-int(level<eye[axis])<0):
            continue
        keys,idx,counts,scores=sample(level)
        viable={}
        for n,k in enumerate(map(tuple,keys.tolist())):
            q=scores.get(k,0.)
            if q<.70 or counts[n]<8:
                continue
            rows=idx==n
            if np.mean(base_scores[rows]+.20<q)>=.8:
                viable[k]=(n,q)
        for (a,b),(n,q) in viable.items():
            if not any(x in viable for x in ((a+1,b),(a-1,b),(a,b+1),(a,b-1))):
                continue
            rows=(idx==n)&(q>best)
            owner[rows]=len(winners)
            best[rows]=q
            winners.append((coordinates(np.array([[a,b]]),axis,level,eye)[0],counts[n]))
    live={i for i,(cell,n) in enumerate(winners) if (owner==i).sum()>=.65*n}
    if not live:
        return cells,0
    reallocated=np.isin(owner,list(live))
    covered=np.bincount(original_idx[reallocated],minlength=len(total))
    remove=coordinates(original[covered>=.65*total],axis,primary,eye)
    return cells.difference(remove).union(winners[i][0] for i in live),len(live)


def predict(image,version,parts,counts,camera):
    if camera['P'] is None or not parts:
        return placer.place_with_camera(image,version,parts,counts,camera)
    rgb=np.asarray(image)
    atlas=_atlas(version)
    weights=frozen_weights()
    if atlas is None or weights is None:
        return placer.place_with_camera(image,version,parts,counts,camera)
    original=placer._estimate_plane_cells
    events=[]
    def refine(image,parts,masks,P,eye,ignored=None):
        info={}
        def pick(part,mask,P,eye,rays,axis,baseline,scene):
            opts=options(mask,P,eye,rays,axis,baseline,rgb,atlas,scene)
            level=ranked_choice(opts,baseline,weights)
            info[part['id']]=(rays,axis,level)
            return level
        result=original(image,parts,masks,P,eye,pick)
        for i,(part,mask) in enumerate(zip(parts,masks)):
            if not result[i] or part['id'] not in info:
                continue
            cells,n=adjust(result[i],mask,P,eye,*info[part['id']],rgb,atlas)
            result[i]=cells
            if n:
                events.append(dict(id=part['id'],tiles=n))
        return result
    with patch.object(placer,'_estimate_plane_cells',refine):
        cells,status=placer.place_with_camera(image,version,parts,counts,camera)
    status['secondary_depths']=events
    return cells,status
