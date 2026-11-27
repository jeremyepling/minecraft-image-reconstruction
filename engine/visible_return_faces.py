"""Test a visible side or top return as evidence for the depth of an adjoining facade face."""
from collections import defaultdict

import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.roof_span_texture import measure
from engine.thin_atlas_faces import StripEvidence
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _sample_face


def adjacent_proposals(ctx):
    # Back-project orientation disagreements. Neighboring rendered depths only nominate planes.
    step=3
    result=set()
    for axis in range(3):
        yy,xx=np.nonzero(ctx.masks[axis,1::step,1::step] & (ctx.axes[1::step,1::step]!=axis))
        if not len(xx):
            continue
        x,y=1+step*xx,1+step*yy
        rays=ctx._rays(x,y)
        for dx,dy in ((0,0),(0,9),(9,0),(0,-9),(-9,0)):
            u=np.clip(x+dx,0,ctx.size[0]-1)
            v=np.clip(y+dy,0,ctx.size[1]-1)
            old=ctx.depth[v,u]
            valid=(old>0)&(abs(rays[:,axis])>1.e-5)
            if not valid.any():
                continue
            old_rays=ctx._rays(u[valid],v[valid])
            seed=np.floor(ctx.eye[axis]+old_rays[:,axis]/old[valid]).astype(int)
            ray=rays[valid]
            for delta in (-2,-1,0,1):
                lower=seed+delta
                plane=lower+(ctx.eye[axis]>lower+1)
                dist=(plane-ctx.eye[axis])/ray[:,axis]
                ok=(dist>1)&(dist<160)&((lower>ctx.eye[axis])|(lower+1<ctx.eye[axis]))
                xyz=np.floor(ctx.eye+dist[ok,None]*ray[ok]).astype(int)
                xyz[:,axis]=lower[ok]
                cells,n=np.unique(xyz,axis=0,return_counts=True)
                result.update(map(tuple,cells[(n>=2)&(cells[:,1]>=0)].tolist()))
    return result-ctx.prior


def witness(ctx,cell,axis,joint=None):
    if cell[axis]<=ctx.eye[axis]<=cell[axis]+1:
        return None
    hit=_sample_face(ctx.native,cell,axis,1 if ctx.eye[axis]>cell[axis]+1 else -1,ctx.shape)
    if hit is None:
        return None
    box,mask,depth,*_=hit
    visible=mask & (depth>ctx.depth[box]+1e-8)
    if joint is not None:
        owner,axes,index=joint
        visible &= (owner[box]==index[cell]) & (axes[box]==axis)
    aligned=visible & ctx.masks[axis][box]
    n=int(aligned.sum())
    if n<8 or n<.75*visible.sum():
        return None
    # At least one of the two faces must explain a change of visible orientation.
    new=int((aligned & (ctx.axes[box]!=axis)).sum())
    return n,new


def corners(ctx,pool,strict,joint=None):
    cells={}
    for cell in pool:
        evidence={a:w for a in range(3) if (w:=witness(ctx,cell,a,joint)) is not None}
        if len(evidence)<2 or not any(n>=8 and n>=.60*total for total,n in evidence.values()):
            continue
        scores={}
        for axis in evidence:
            fit=measure(ctx,cell,axis,joint)
            if fit and fit['oriented']>=.80 and fit['depth']>=.085 and fit['phase']>=.06:
                scores[axis]=fit
        for axis,fit in scores.items():
            if axis==1 or fit['split']<(.75 if strict else .62) or fit['color']>10:
                continue
            others=[w for a,w in scores.items() if a!=axis]
            if strict and not any(w['split']>=.64 and w['depth']>=.12 for w in others):
                continue
            cells[cell]=axis
            break
    return cells


def recover(image,parts,counts,camera,prior,*,strict=True,atlas_path=None):
    stats=dict(proposals=0,paired=0,accepted=0)
    if not prior or not parts or camera.get('P') is None:
        return prior,stats
    atlas=_atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance.get('atlas_sha256')!=ATLAS_SHA256:
        return prior,stats
    ctx=StripEvidence(image,parts,camera,prior,atlas)
    pool=adjacent_proposals(ctx)
    stats['proposals']=len(pool)
    kept=set(corners(ctx,pool,strict))
    stats['paired']=len(kept)
    while kept:
        joint=ctx.joint(kept)
        owner,_,index=joint
        outside=set(np.unique(owner[(ctx.depth==0)&~ctx.union]))
        supported=set(corners(ctx,{c for c in kept if index[c] not in outside},strict,joint))
        supported=_keep_prior_visible(ctx.size,ctx.P,ctx.eye,prior,supported)-prior if supported else set()
        if supported==kept:
            break
        kept=supported
    stats['accepted']=len(kept)
    return prior|kept,stats


def one_texture(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,strict=False)


def return_anchors(ctx):
    """Find exposed faces where the provisional depth and the observed normal agree."""
    out=defaultdict(set)
    for cell in ctx.prior:
        for axis in range(3):
            if cell[axis]<=ctx.eye[axis]<=cell[axis]+1:
                continue
            sign=1 if ctx.eye[axis]>cell[axis]+1 else -1
            neighbor=list(cell)
            neighbor[axis]+=sign
            if tuple(neighbor) in ctx.prior:
                continue
            hit=_sample_face(ctx.native,cell,axis,sign,ctx.shape)
            if hit is None:
                continue
            box,mask,depth,*_=hit
            seen=mask & (ctx.axes[box]==axis) & np.isclose(ctx.depth[box],depth,atol=1.e-8,rtol=0)
            count=np.count_nonzero(seen & ctx.masks[axis][box])
            if count>=12 and count>=.80*seen.sum():
                out[axis].add(cell)
    return out


def recover_strips(image,parts,counts,camera,prior,*,pair=True,atlas_path=None):
    stats=dict(nominated=0,texture_supported=0,accepted=0)
    if not prior or not parts or camera.get('P') is None:
        return prior,stats
    atlas=_atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance.get('atlas_sha256')!=ATLAS_SHA256:
        return prior,stats
    ctx=StripEvidence(image,parts,camera,prior,atlas)
    anchored=return_anchors(ctx)
    candidates=set()
    for face_axis in (0,2):
        for side_axis in (1,2-face_axis):
            anchors=anchored[side_axis]
            for a in anchors:
                # Parallel front candidates share a plane with the measured edge(s).
                # A single edge nominates at most its immediate neighbor.
                if pair:
                    for dist in range(2,7):
                        b=list(a)
                        b[side_axis]+=dist
                        if tuple(b) in anchors:
                            for offset in range(1,dist):
                                c=list(a)
                                c[side_axis]+=offset
                                if tuple(c) not in prior:
                                    candidates.add((tuple(c),face_axis))
                            break
                else:
                    for offset in (-1,1):
                        c=list(a)
                        c[side_axis]+=offset
                        if c[1]>=0 and tuple(c) not in prior:
                            candidates.add((tuple(c),face_axis))
    stats['nominated']=len(candidates)
    def fits(c,a,joint):
        if c[a]<=ctx.eye[a]<=c[a]+1:
            return False
        r=measure(ctx,c,a,joint)
        return bool(r and r['split']>=.76 and r['phase']>=.10 and r['depth']>=.15
                    and r['color']<=8 and r['oriented']>=.85 and r['inside']>=.98)
    candidates={(c,a) for c,a in candidates if fits(c,a,None)}
    cells={c for c,_ in candidates}
    stats['texture_supported']=len(cells)
    while cells:
        joint=ctx.joint(cells)
        owner,_,index=joint
        spill=set(np.unique(owner[(ctx.depth==0)&~ctx.union]))
        supported={c for c,a in candidates if c in cells and index[c] not in spill and fits(c,a,joint)}
        supported=_keep_prior_visible(ctx.size,ctx.P,ctx.eye,prior,supported)-prior if supported else set()
        if cells==supported:
            break
        cells=supported
    stats['accepted']=len(cells)
    return prior|cells,stats


def single_strip(image,parts,counts,camera,prior):
    return recover_strips(image,parts,counts,camera,prior,pair=False)
