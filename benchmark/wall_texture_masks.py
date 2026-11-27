"""Image-only wall seams, texture-continuous joins and bottom-edge alignment."""

import numpy as np
from PIL import ImageFilter

from engine.plane_consensus import with_mask
from engine.pillar_seams import bounds,native_mask,orientation,split_pillars


def contact(a,b,color,box):
    x0,y0,x1,y1=box
    if x1<=x0 or y1<=y0:return None
    a=a[y0:y1,x0:x1];b=b[y0:y1,x0:x1]
    ys=[];xs=[];vec=[];other=[];interior=[]
    for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
        # Nearest disjoint owners can have a small unknown gap.
        for distance in (1,3,5):
            h,w=a.shape
            y,x=np.nonzero(a & np.roll(b,(-dy*distance,-dx*distance),(0,1)))
            ok=(x>=8)&(x<w-8)&(y>=8)&(y<h-8)
            x,y=x[ok],y[ok]
            if len(x)==0:continue
            edge=~a[y+dy,x+dx]
            x,y=x[edge],y[edge]
            if len(x)==0:continue
            x=x+x0;y=y+y0
            ys.extend(y);xs.extend(x)
            A=color[y-2*dy,x-2*dx];B=color[y+(distance+2)*dy,x+(distance+2)*dx]
            vec.extend(A);other.extend(B)
            interior.extend(.5*(abs(A-color[y-6*dy,x-6*dx])+abs(B-color[y+(distance+5)*dy,x+(distance+5)*dx])))
    if len(xs)<30:return None
    xy=np.unique(np.column_stack((xs,ys)),axis=0)
    if len(xy)<24:return None
    a,b=np.asarray(vec),np.asarray(other)
    contrast=float(abs(a-b).mean());within=float(np.asarray(interior).mean())
    color_step=float(np.max(abs(np.median(a,axis=0)-np.median(b,axis=0))))
    covariance=np.cov(xy.T)
    values,basis=np.linalg.eigh(covariance)
    return xy.mean(0),basis[:,-1],values,contrast,within,color_step,len(xy)


def rewrite(image,parts,camera,mode):
    if mode in ('split','pillar'):
        return split_pillars(image,parts,camera,mode)
    if mode=='selective':
        separated, cuts = rewrite(image, parts, camera, 'pillar')
        joined, joins = rewrite(image, separated, camera, 'diagonal')
        return joined, cuts + joins
    if not parts or camera.get('P') is None:return parts,[]
    size=image.size
    p,rot=orientation(camera,size)
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    smooth=np.asarray(image.convert('RGB').filter(ImageFilter.GaussianBlur(1.35)),np.float32)
    masks=[native_mask(part,size) for part in parts]
    axes=[int(np.argmax(abs(rot.T @ part['normal_camera']))) if part.get('normal_camera') is not None else -1 for part in parts]
    boxes=[bounds(m) for m in masks]
    records=[]
    def make(part,mask,name):
        base={k:v for k,v in part.items() if k not in ('context_crop','source_crop')}
        return dict(with_mask(base,mask,size,gray,np.ones(2)),id=name)
    if mode=='prune':
        keep=[]
        for i,part in enumerate(parts):
            x,y,r,b=boxes[i]; area=int(masks[i].sum())
            if area<512 or area/max(r-x,b-y)<8:
                records.append(dict(op='prune',ids=[part['id']],area=area))
            else:keep.append(part)
        return keep,records
    if mode in ('join', 'diagonal'):
        pairs=[]
        for i,a in enumerate(parts):
            for j in range(i):
                b=parts[j]
                if axes[i] not in (0,2) or axes[i]!=axes[j] or np.dot(a['normal_camera'],b['normal_camera'])<.985:continue
                x,y,r,bt=boxes[i];X,Y,R,B=boxes[j]
                if min(r,R)<max(x,X)-5 or min(bt,B)<max(y,Y)-5:continue
                box=(max(0,max(x,X)-12),max(0,max(y,Y)-12),min(size[0],min(r,R)+12),min(size[1],min(bt,B)+12))
                obs=contact(masks[i],masks[j],smooth,box)
                if obs is None:continue
                center,direction,vals,c,w,median,n=obs
                angles=[]
                for axis in (1,2-axes[i]):
                    d=p[:2,axis]-center*p[2,axis]
                    angles.append(np.degrees(np.arccos(np.clip(abs(d@direction)/np.linalg.norm(d),0,1))))
                diagonal=min(angles)>=12 and vals[1]>=15*max(1,vals[0])
                small=min(masks[i].sum(),masks[j].sum())<=.08*max(masks[i].sum(),masks[j].sum())
                if mode=='diagonal':
                    # Undo a split only for siblings sharing one long non-projective boundary.
                    siblings = a.get('parent_id') and a.get('parent_id') == b.get('parent_id')
                    if not (siblings and diagonal and n >= 96 and min(angles) >= 18):
                        continue
                if median<=6 and c<=max(4,w*1.5) and (diagonal or small):
                    pairs.append((c-w,i,j,dict(op='join',ids=[a['id'],b['id']],border_pixels=n,
                         contrast=round(c,2),within=round(w,2),angle=round(min(angles),1),median=round(median,2))))
        seen=set();out=[]
        for _,i,j,stat in sorted(pairs,key=lambda v:v[0]):
            if i in seen or j in seen:continue
            seen.update((i,j));records.append(stat)
            base=parts[i] if masks[i].sum()>masks[j].sum() else parts[j]
            out.append(make(base,masks[i]|masks[j],base['id']+'-joined'))
        return [part for i,part in enumerate(parts) if i not in seen]+out,records
    out=[]
    occupied=np.logical_or.reduce(masks)
    for i,part in enumerate(parts):
        mask=masks[i]
        if axes[i] not in (0,2):
            out.append(part);continue
        if mode=='shade':
            updated,info=bottom(mask,occupied,smooth,p,axes[i])
            if info:
                out.append(make(part,updated,part['id']+'-shade'));records.extend(dict(op='shade',ids=[part['id']],**r) for r in info);continue
        out.append(part)
    return out,records


def bottom(mask,occupied,rgb,p,axis):
    """Follow a long photographed horizontal silhouette within a collar of the lower outline."""
    x0,y0,x1,y1=bounds(mask)
    changed=[]
    out=mask.copy()
    vp=p[:,2-axis]
    h=mask.shape[0]
    for xs in np.array_split(np.arange(x0,x1),max(1,(x1-x0)//64)):
        if len(xs)<32:continue
        sub=mask[:,xs]
        if not np.all(sub.any(0)):continue
        last=h-1-sub[::-1].argmax(0)
        if min(last)<8 or max(last)>h-30:continue
        if np.mean(occupied[(last+3),xs])>.18:continue
        mid=float(np.median(xs)); height=float(np.median(last))
        slope=(vp[1]-height*vp[2])/(vp[0]-mid*vp[2])
        if abs(slope)>.45:continue
        nominal=np.rint(height+slope*(xs-mid)).astype(int)
        if np.mean(abs(nominal-last)>10)>.12:continue
        results=[]
        for shift in range(-5,25):
            y=nominal+shift
            if min(y)<5 or max(y)+4>=h:continue
            top=rgb[y-3,xs];below=rgb[y+3,xs]
            edge=abs(top-below).mean(1)
            sides=(abs(top-rgb[y-5,xs]).mean(1)+abs(below-rgb[y+5,xs]).mean(1))/2
            results.append((float(np.median(edge-sides)),shift,float(np.mean(edge>=6))))
        if not results:continue
        score,shift,fraction=max(results)
        if score<7 or fraction<.78 or shift<2:continue
        end=nominal+shift
        support=abs(end-last)<=25
        for xx,start,stop,ok in zip(xs,last,end,support):
            if not ok or stop<=start or occupied[start+1:stop+1,xx].any():continue
            out[start+1:stop+1,xx]=True
        changed.append(dict(x_range=[int(xs[0]),int(xs[-1])],shift=shift,contrast=round(score,2)))
    return out,changed if np.any(out!=mask) else []
