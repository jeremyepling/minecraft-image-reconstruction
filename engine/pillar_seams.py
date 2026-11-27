"""Locate distinct photographed pillar faces using registered texture and silhouette seams."""
from collections import Counter

import numpy as np
from PIL import ImageFilter

from engine.plane_consensus import with_mask
from engine.surface_model import intrinsic
from engine.surface_perimeter import _native_alpha


def native_mask(part,size):
    area=np.zeros(size[::-1],bool)
    patch=_native_alpha(part['support'],size) if part.get('support') else None
    if patch:
        mask,(x,y,r,b)=patch
        area[y:b,x:r]=np.asarray(mask,bool)
    return area


def bounds(mask):
    y,x=np.nonzero(mask)
    return int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1


def orientation(camera,size):
    p=np.asarray(camera['P'],float)
    p=p/np.linalg.norm(p[2,:3])
    f=np.linalg.norm(p[0,:3]-.5*size[0]*p[2,:3])
    return p,np.linalg.solve(intrinsic(size,f),p[:,:3])


def canonical_x(x,y,mid,vp):
    # Horizontal intercept of the rectilinear vertical through (x,y).
    return x+(mid-y)*(vp[0]-x*vp[2])/(vp[1]-y*vp[2])


def profile_score(color,mask,x,y,sign):
    left=x-4
    right=x+4
    inside=(left>=3)&(right+3<mask.shape[1])
    y,left,right=y[inside],left[inside],right[inside]
    both=mask[y,left-2]&mask[y,right+2]
    y,left,right=y[both],left[both],right[both]
    if len(y)<28:
        return None
    def band(xx):
        return np.mean([color[y,xx+o] for o in (-2,-1,0,1,2)],axis=0)
    # Compare horizontal mortar/texel rows at the same height. A staggered or
    # differently scaled pattern keeps disagreeing above and below the junction.
    a,b=band(left),band(right)
    alt=band(left-3 if sign>0 else right+3)
    center=a if sign>0 else b
    change=abs(a-b).mean(axis=1)
    intra=abs(alt-center).mean(axis=1)
    score=float(np.mean(np.minimum(change,40)-np.minimum(intra,40)))
    def correlation(u,v):
        u=u.mean(1);v=v.mean(1)
        return float(np.corrcoef(u,v)[0,1]) if u.std()>2 and v.std()>2 else 1.
    return score,correlation(a,b),correlation(alt,center),len(y)


def seams(mask,context,rgb,p):
    x0,y0,x1,y1=bounds(mask)
    if y1-y0<36 or x1-x0<14 or y1-y0<1.25*(x1-x0):
        return []
    mid=(y0+y1)*.5
    vp=p[:,1]
    hist=Counter()
    for side in (1,-1):
        # A visible contour that continues inside a combined mask nominates
        # a cut. Use adjacent roughly parallel patches as additional contours.
        for other in context:
            ox,oy,orr,ob=bounds(other)
            if orr<x0-3 or ox>x1+3 or ob<y0-45 or oy>y1+45:
                continue
            sample=other[oy:ob,ox:orr]
            ys=np.flatnonzero(sample.any(1))
            xx=(sample[ys].argmax(1) if side==-1 else (orr-ox-1)-sample[ys,::-1].argmax(1))+ox
            projected=canonical_x(xx,ys+oy,mid,vp)
            for k,v in zip(*np.unique(np.rint(projected/2).astype(int),return_counts=True)):
                if v>=12 and x0+5<2*k<x1-5:
                    hist[(int(k)*2,side)]+=int(v)
    y=np.arange(y0,y1)
    candidates=[]
    for (cx,sign),anchor in hist.items():
        if anchor<18:continue
        xx=np.rint(canonical_x(cx,mid,y,vp)).astype(int)
        fit=profile_score(rgb,mask,xx,y,sign)
        if fit is None:continue
        score,cross,within,n=fit
        if score>=3.5 and cross<=.35 and within-cross>=.30 and within>=.70:
            candidates.append((score+8*(within-cross),cx,sign,n,anchor,cross,within))
    if not candidates:return []
    _,cx,sign,n,anchor,cross,within=max(candidates)
    xx=np.rint(canonical_x(cx,mid,y,vp)).astype(int)
    left=mask.copy()
    left[y0:y1]&=np.arange(mask.shape[1])[None,:]<xx[:,None]
    right=mask&~left
    if min(left.sum(),right.sum())<48:return []
    return [(left,right,dict(intercept=cx,rows=n,contour_rows=anchor,cross_correlation=round(cross,3),
                             same_side_correlation=round(within,3)))]


def split_pillars(image,parts,camera,mode="pillar"):
    if not parts or camera.get('P') is None:return parts,[]
    size=image.size
    p,rot=orientation(camera,size)
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    rgb=np.asarray(image.convert('RGB'),np.float32)
    masks=[native_mask(part,size) for part in parts]
    boxes=[bounds(m) if m.any() else None for m in masks]
    axes=[int(np.argmax(abs(rot.T @ part['normal_camera'])))
          if box is not None and part.get('normal_camera') is not None else -1
          for part,box in zip(parts,boxes)]
    records=[]
    def make(part,mask,name):
        base={k:v for k,v in part.items() if k not in ('context_crop','source_crop')}
        return dict(with_mask(base,mask,size,gray,np.ones(2)),id=name)
    # A mark across two vertically stacked pieces can be a mortar course.
    # Reconsider it only if a coherent vertical texture break continues
    # through their union and into its own silhouette.
    seen=set()
    out=[]
    for i,part in enumerate(parts):
        if i in seen:continue
        if axes[i] not in (0,2):
            out.append(part);continue
        x,y,r,b=boxes[i]
        group=[i]
        for j in range(i+1,len(parts)):
            if j in seen or axes[j]!=axes[i] or not part.get('parent_id') or part.get('parent_id')!=parts[j].get('parent_id'):continue
            X,Y,R,B=boxes[j]
            if abs(x-X)<=8 and abs(r-R)<=8 and min(b,B)>=max(y,Y)-6:
                group.append(j)
        union=np.logical_or.reduce([masks[k] for k in group])
        cuts=seams(union,[union],rgb,p)
        if cuts:
            a,b,info=cuts[0]
            if mode=='pillar' and (info['rows']<96 or info['cross_correlation']>.2 or info['same_side_correlation']<.85):
                out.append(part)
                continue
            out.extend(make(part,sub,part['id']+'-tex-'+str(j)) for j,sub in enumerate((a,b)))
            records.append(dict(op='split',ids=[parts[k]['id'] for k in group],**info))
            seen.update(group)
        else:out.append(part)
    return out,records
