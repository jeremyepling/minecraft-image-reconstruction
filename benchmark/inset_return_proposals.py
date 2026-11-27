"""Looser RGB-only alternatives to the frozen inset rule for grouped chooser tests.

Each candidate exposes a local native-resolution boolean back mask, its native
half-open ROI, numeric decision features and the corresponding strict decision.
A mask always intersects the original host's observed support.
"""
from itertools import product

import numpy as np
from PIL import ImageFilter

from benchmark.inset_return_grid import _brightness, _fit, _front, _iou, _labels, _level
from engine.pillar_seams import bounds, native_mask, orientation


def _options(front,eye,axis,level,host,jamb,sill):
    uv=front[jamb|sill][:,[2-axis,1]]
    limits=[np.percentile(uv[:,0],.5),np.percentile(uv[:,0],99.5),
            np.percentile(uv[:,1],.5),np.percentile(uv[:,1],99.5)]
    choices=[sorted(set((int(np.rint(v-.25)),int(np.rint(v)),int(np.rint(v+.25))))) for v in limits]
    sign=int(np.sign(level-eye[axis]))
    best={}
    if not sign:
        return best
    u,v=front[...,2-axis],front[...,1]
    for edges in product(*choices):
        x0,x1,y0,y1=edges
        if not (1<=x1-x0<=13 and 1<=y1-y0<=11):
            continue
        sides=[(u>x0-.35)&(u<x0)&(v>y0)&(v<y1),(u>x1)&(u<x1+.35)&(v>y0)&(v<y1),
               (v>y0-.35)&(v<y0)&(u>x0)&(u<x1),(v>y1)&(v<y1+.35)&(u>x0)&(u<x1)]
        rim=[float(host[s].mean()) if s.any() else 0. for s in sides]
        if np.mean(rim)<.23:
            continue
        for signed in (-3,-2,-1,1,2,3,4):
            label=_labels(front,eye,axis,level,signed*sign,edges)
            back=label==3
            if back.sum()<12:
                continue
            side_iou=_iou(label==1,jamb)
            sill_iou=_iou(label==2,sill)
            overlap=float(host[back].mean())
            rank=min(side_iou,sill_iou)+.3*(side_iou+sill_iou)
            if rank>best.get(signed,(-1,))[0]:
                best[signed]=(rank,dict(level=level, depth=signed*sign,unsigned_depth=signed,rim=list(edges),
                    side_iou=float(side_iou),sill_iou=float(sill_iou),host_fraction=overlap,rank=rank,
                    rim_min=min(rim),rim_mean=float(np.mean(rim))))
    return best


def proposals(image,parts,camera):
    """Return the two best positive-depth alternatives per return pair and the strict choice."""
    if len(parts)<3 or camera.get('P') is None:
        return []
    P,rot=orientation(camera,image.size)
    inv=np.linalg.inv(P[:,:3])
    eye=-inv@P[:,3]
    masks=[native_mask(p,image.size) for p in parts]
    boxes=[bounds(m) if m.any() else None for m in masks]
    axes=[int(np.argmax(abs(rot.T@p['normal_camera']))) if p.get('normal_camera') is not None else -1 for p in parts]
    horiz=[k for k,a in enumerate(axes) if a==1 and boxes[k]]
    if not horiz:
        return []
    bottom=np.where(np.logical_or.reduce(masks),np.arange(image.height)[:,None],-1).max(0)
    rgb=np.asarray(image.convert('RGB').filter(ImageFilter.GaussianBlur(1.0)),np.float32)
    result=[]
    for i,(p,host,axis) in enumerate(zip(parts,masks,axes)):
        if axis not in (0,2) or host.sum()<1200:
            continue
        level=_level(host,bottom,inv,eye,axis)
        if level is None:
            continue
        hx,hy,hr,hb=boxes[i]
        for j,jb in enumerate(boxes):
            if axes[j]!=2-axis or jb is None:
                continue
            for k in horiz:
                kb=boxes[k]
                x,y,r,b=min(jb[0],kb[0]),min(jb[1],kb[1]),max(jb[2],kb[2]),max(jb[3],kb[3])
                distance=max(max(jb[0],kb[0])-min(jb[2],kb[2]), max(jb[1],kb[1])-min(jb[3],kb[3]),0)
                if x<hx-12 or y<hy-12 or r>hr+12 or b>hb+12 or distance>14:
                    continue
                strict_pair=x>=hx-3 and y>=hy-3 and r<=hr+3 and b<=hb+3 and distance<=6 and host.sum()>=3500
                pad=max(12,round(.16*max(r-x,b-y)))
                x,y,r,b=max(0,x-pad),max(0,y-pad),min(image.width,r+pad),min(image.height,b+pad)
                yy,xx=np.mgrid[y:b:3,x:r:3]
                take=np.s_[y:b:3,x:r:3]
                front=_front(inv,eye,axis,level,xx,yy)
                h,jamb,sill=host[take],masks[j][take],masks[k][take]
                if min(jamb.sum(),sill.sum())<4:
                    continue
                fits=_options(front,eye,axis,level,h,jamb,sill)
                positive=[val[1] for signed,val in fits.items() if signed>0
                          and min(val[1]['side_iou'],val[1]['sill_iou'])>.08 and val[1]['host_fraction']>.40]
                options=sorted(positive,key=lambda v:-v['rank'])[:2]
                strict=_fit(front,eye,axis,level,h,jamb,sill) if strict_pair else None
                if strict and not _brightness(rgb[take],front,eye,axis,strict):
                    strict=None
                if strict and not any(q['rim']==strict['rim'] and q['depth']==strict['depth'] for q in options):
                    lookup=fits.get(strict['unsigned_depth'])
                    opts=dict(strict,rank=min(strict['side_iou'],strict['sill_iou'])+.3*(strict['side_iou']+strict['sill_iou']),
                              rim_min=lookup[1]['rim_min'] if lookup else .55,rim_mean=lookup[1]['rim_mean'] if lookup else .55)
                    options.append(opts)
                for fit in options:
                    _brightness(rgb[take],front,eye,axis,fit)
                    if 'back_halo_luma_step' not in fit:
                        continue
                    yy0,xx0=np.mgrid[y:b,x:r]
                    native=_front(inv,eye,axis,level,xx0,yy0)
                    back=_labels(native,eye,axis,level,fit['depth'],fit['rim'])==3
                    selected=back&host[y:b,x:r]
                    if selected.sum()<64:
                        continue
                    alternatives=[v[0] for sign,v in fits.items() if sign!=fit['unsigned_depth']]
                    step=fit['back_return_rgb_step']
                    strict_picked=bool(strict and fit['rim']==strict['rim'] and fit['depth']==strict['depth']
                                       and selected.sum()>=95 and selected.sum()>=.78*back.sum())
                    features=dict(side_iou=fit['side_iou'],sill_iou=fit['sill_iou'],host_fraction=fit['host_fraction'],
                        weaker_return_iou=min(fit['side_iou'],fit['sill_iou']),rim_min=fit['rim_min'],rim_mean=fit['rim_mean'],
                        depth_margin=fit['rank']-max([0.,*alternatives]),outward_margin=fit['rank']-max([0.,*[v[0] for s,v in fits.items() if s<0]]),
                        depth=fit['unsigned_depth'],width=fit['rim'][1]-fit['rim'][0],height=fit['rim'][3]-fit['rim'][2],
                        back_side_step=step[0],back_sill_step=step[1],back_halo_luma_step=fit['back_halo_luma_step'],
                        pixels=int(selected.sum()),jamb_pixels=int(jamb.sum())*9,sill_pixels=int(sill.sum())*9)
                    result.append(dict(host=p['id'],side=parts[j]['id'],sill=parts[k]['id'],roi_xyxy=(x,y,r,b),mask=selected,
                                       rim=fit['rim'],level=level,depth=int(fit['depth']),features=features,strict=strict_picked))
    return result
