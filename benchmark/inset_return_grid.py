"""Fit visibly enclosed pockets to a host wall and two perpendicular detected returns."""
from itertools import product

import numpy as np
from PIL import ImageFilter

from engine.pillar_seams import bounds, native_mask, orientation
from engine.plane_consensus import with_mask


def _front(inv, eye, axis, level, x, y):
    rays = np.stack((x+.5,y+.5,np.ones(x.shape)), axis=-1) @ inv.T
    s = (level-eye[axis])/rays[...,axis]
    return eye+s[...,None]*rays


def _labels(front, eye, axis, level, depth, edges):
    """First surface seen along a ray passing the front rim, clipped in both tangent axes."""
    x0,x1,y0,y1 = edges
    tangent = [2-axis,1]
    uv = front[...,tangent]
    inside = (uv[...,0]>=x0)&(uv[...,0]<=x1)&(uv[...,1]>=y0)&(uv[...,1]<=y1)
    delta = (uv-eye[tangent]) * (depth/(level-eye[axis]))
    boundary = np.where(delta>0,[x1,y1],[x0,y0])
    distance = np.divide(boundary-uv,delta,out=np.full_like(uv,np.inf),where=abs(delta)>1.e-8)
    exit_axis = distance.argmin(-1)
    return np.where(inside, np.where(distance.min(-1)<1., 1+exit_axis, 3), 0)


def _iou(a, b):
    return np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b))


def _level(mask, bottom, inv, eye, axis):
    y,x = np.nonzero(mask & ((np.arange(mask.shape[0])[:,None])==bottom))
    if len(x)<16:
        return None
    xyz = _front(inv,eye,1,0,x,y)
    levels = xyz[:,axis]
    usable = np.isfinite(levels)&(abs(levels-eye[axis])<100)
    if usable.sum()<16:
        return None
    active = np.rint(levels[usable]).astype(int)
    values,counts = np.unique(active,return_counts=True)
    best = counts.argmax()
    if counts[best]<max(12,.4*len(active)):
        return None
    return int(values[best])


def _fit(front, eye, axis, level, host, jamb, sill):
    if min(jamb.sum(),sill.sum())<5:
        return None
    uv = front[jamb|sill][:,[2-axis,1]]
    limits = np.array([np.percentile(uv[:,0],.5),np.percentile(uv[:,0],99.5),
                       np.percentile(uv[:,1],.5),np.percentile(uv[:,1],99.5)])
    choices = [sorted(set([int(np.rint(v)),int(np.rint(v-.20)),int(np.rint(v+.20))])) for v in limits]
    options = []
    toward_back = int(np.sign(level-eye[axis]))
    if not toward_back:
        return None
    for edges in product(*choices):
        w,h=edges[1]-edges[0],edges[3]-edges[2]
        if not (1<=w<=12 and 1<=h<=10):
            continue
        inside = ((front[...,2-axis]>edges[0])&(front[...,2-axis]<edges[1])&
                  (front[...,1]>edges[2])&(front[...,1]<edges[3]))
        # Visible wall immediately around the rim rules out a freestanding box.
        u,v=front[...,2-axis],front[...,1]
        sides=[(u>edges[0]-.35)&(u<edges[0])&(v>edges[2])&(v<edges[3]),
               (u>edges[1])&(u<edges[1]+.35)&(v>edges[2])&(v<edges[3]),
               (v>edges[2]-.35)&(v<edges[2])&(u>edges[0])&(u<edges[1]),
               (v>edges[3])&(v<edges[3]+.35)&(u>edges[0])&(u<edges[1])]
        if any(s.sum()<3 or host[s].mean()<.55 for s in sides):
            continue
        for signed in (-3,-2,-1,1,2,3,4):
            labels = _labels(front,eye,axis,level,signed*toward_back,edges)
            back = labels==3
            if back.sum()<18:
                continue
            side_iou = _iou(jamb,labels==1)
            sill_iou = _iou(sill,labels==2)
            host_share = host[back].mean()
            rank = min(side_iou,sill_iou) + .3*(side_iou+sill_iou)
            options.append((rank,signed,edges,side_iou,sill_iou,host_share))
    if not options:
        return None
    ordered = sorted(options, reverse=True)
    rank,signed,edges,side,sill_score,share=ordered[0]
    competitor = max([0.,*[r[0] for r in ordered if r[1]!=signed]])
    if signed<=0 or min(side,sill_score)<.43 or max(side,sill_score)<.64 or share<.83 or rank-competitor<.09:
        return None
    return dict(level=level, depth=signed*toward_back, unsigned_depth=signed, rim=list(edges),
                side_iou=round(side,4), sill_iou=round(sill_score,4), host_fraction=round(float(share),4),
                margin=round(rank-competitor,4))


def _brightness(rgb, front, eye, axis, fit):
    """Compare photo luminance across the back seams and on both sides of the host rim."""
    label = _labels(front,eye,axis,fit['level'],fit['depth'],fit['rim'])
    u,v = front[...,2-axis],front[...,1]
    x0,x1,y0,y1 = fit['rim']
    surround = ((u>x0-.30)&(u<x1+.30)&(v>y0-.30)&(v<y1+.30)&(label==0))
    groups = [rgb[label==k] for k in (1,2,3)]
    if min([len(g) for g in groups]+[int(surround.sum())])<5:
        return False
    side,sill,back=[np.median(g,axis=0) for g in groups]
    exterior=np.median(rgb[surround],axis=0)
    fit['back_return_rgb_step']=[round(float(abs(back-value).mean()),2) for value in (side,sill)]
    fit['back_halo_luma_step']=round(float(np.mean(exterior-back)),2)
    return min(fit['back_return_rgb_step'])>=2.5 and fit['back_halo_luma_step']>=5.


def refine(image, parts, camera):
    if len(parts)<3 or camera.get('P') is None:
        return parts,[]
    P,rot=orientation(camera,image.size)
    inv=np.linalg.inv(P[:,:3])
    eye=-inv@P[:,3]
    masks=[native_mask(p,image.size) for p in parts]
    bb=[bounds(m) if m.any() else None for m in masks]
    axes=[np.argmax(abs(rot.T@p['normal_camera'])) if p.get('normal_camera') is not None else -1 for p in parts]
    horizontal=[i for i,a in enumerate(axes) if a==1 and bb[i] is not None]
    if not horizontal:
        return parts,[]
    bottom=np.where(np.logical_or.reduce(masks),np.arange(image.height)[:,None],-1).max(0)
    rgb=np.asarray(image.convert('RGB').filter(ImageFilter.GaussianBlur(1.0)),np.float32)
    out=[]
    events=[]
    for i,(p,mask,axis) in enumerate(zip(parts,masks,axes)):
        candidates=[]
        if axis in (0,2) and mask.sum()>=3500:
            level=_level(mask,bottom,inv,eye,axis)
            if level is not None:
                hx,hy,hr,hb=bb[i]
                for j,jb in enumerate(bb):
                    if axes[j]!=2-axis or jb is None:
                        continue
                    for k in horizontal:
                        kb=bb[k]
                        x,y,r,b=min(jb[0],kb[0]),min(jb[1],kb[1]),max(jb[2],kb[2]),max(jb[3],kb[3])
                        if not (hx-3<=x<r<=hr+3 and hy-3<=y<b<=hb+3):
                            continue
                        if max(jb[0],kb[0])-min(jb[2],kb[2])>6 or max(jb[1],kb[1])-min(jb[3],kb[3])>6:
                            continue
                        pad=max(12,round(.16*max(r-x,b-y)))
                        x,y,r,b=max(0,x-pad),max(0,y-pad),min(image.width,r+pad),min(image.height,b+pad)
                        roi=np.s_[y:b:3,x:r:3]
                        yy,xx=np.mgrid[y:b:3,x:r:3]
                        front=_front(inv,eye,axis,level,xx,yy)
                        fit=_fit(front,eye,axis,level,mask[roi],masks[j][roi],masks[k][roi])
                        if fit and _brightness(rgb[roi],front,eye,axis,fit):
                            fit.update(host=p['id'], side=parts[j]['id'], sill=parts[k]['id'])
                            candidates.append((fit,(x,y,r,b)))
        if not candidates:
            out.append(p)
            continue
        if not events:
            gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
        clean={key:value for key,value in p.items() if key not in
               ('source_crop','context_crop','corner_junctions','outline','sampling_chart')}
        remainder=mask.copy()
        for fit,(x,y,r,b) in sorted(candidates,key=lambda c:-c[0]['margin']):
            yy,xx=np.mgrid[y:b,x:r]
            front=_front(inv,eye,axis,fit['level'],xx,yy)
            inside=_labels(front,eye,axis,fit['level'],fit['depth'],fit['rim'])==3
            selected=np.zeros_like(mask)
            selected[y:b,x:r]=inside&remainder[y:b,x:r]
            if selected.sum()<95 or selected.sum()<.78*inside.sum():
                continue
            output_id=p['id']+'-inset-'+str(len(events))
            new=with_mask(dict(clean,id=output_id,parent_id=p['id']),selected,image.size,gray,np.ones(2))
            out.append(new)
            remainder&=~selected
            fit.update(output=output_id,reassigned=int(selected.sum()))
            events.append(fit)
        out.append(with_mask(clean,remainder,image.size,gray,np.ones(2)) if np.any(remainder!=mask) else p)
    return out,events
