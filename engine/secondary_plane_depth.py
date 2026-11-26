"""Use local atlas matches to resolve visible depth changes inside merged masks."""
import numpy as np

from engine.atlas_plane_depth import _coefficients, _patch_scores


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


def wrap_selector(choose_level, rgb, atlas):
    """Refine mask candidates before the common visibility and occlusion checks."""
    info = {}
    def select(part,mask,P,eye,rays,axis,baseline,scene):
        level = choose_level(part,mask,P,eye,rays,axis,baseline,scene)
        info[part['id']] = (rays,axis,level)
        return level
    def refine(parts,masks,P,eye,candidates):
        events = []
        for i,(part,mask) in enumerate(zip(parts,masks,strict=True)):
            if not candidates[i] or part['id'] not in info:
                continue
            candidates[i],n = adjust(candidates[i],mask,P,eye,*info[part['id']],rgb,atlas)
            if n:
                events.append(dict(id=part['id'],tiles=n))
        return events
    return select,refine


