"""Source-only nomination of an upright crease using RGB and an estimated camera."""
import numpy as np
from PIL import ImageFilter

from engine.pillar_seams import native_mask, orientation, canonical_x
from engine.plane_consensus import with_mask


def line_vote(part_mask, gx, gy, p, x0, y0):
    """Vote between world X/Z tangents using gradients inside this subregion."""
    interior = part_mask[2:-2,2:-2].copy()
    for dy,dx in ((0,2),(0,-2),(2,0),(-2,0)):
        h,w = part_mask.shape
        interior &= part_mask[2+dy:h-2+dy,2+dx:w-2+dx]
    y,x = np.nonzero(interior)
    if len(x)<64:
        return None
    x,y=x+2+x0,y+2+y0
    dx,dy=gx[y,x],gy[y,x]
    mag=np.hypot(dx,dy)
    valid=(mag>=8)&(abs(dy)>abs(dx)*.70)
    if valid.sum()<24:
        return None
    dx,dy,x,y,mag=[a[valid] for a in (dx,dy,x,y,mag)]
    votes=[]
    for axis in (2,0):  # horizontal tangent for a face normal to X, then Z
        vp=p[:,axis]
        tx,ty=vp[0]-vp[2]*x,vp[1]-vp[2]*y
        error=abs(dx*tx+dy*ty)/(mag*np.hypot(tx,ty)+1e-6)
        votes.append(float(np.average(np.exp(-(error/.17)**2),weights=np.minimum(35,mag))))
    return votes


def termination_step(mask,line,x0):
    # Along either side of the photographed seam, find a consistent jump in
    # the first or last supported row. Parallel texture stripes alone cannot
    # pass this independent termination check.
    ends=[]
    for side in (-1,1):
        traces=[]
        for offset in (3,5,7):
            x=line+side*offset-x0
            y=np.arange(len(x))
            valid=(x>=0)&(x<mask.shape[1])
            yy=y[valid][mask[y[valid],x[valid]]]
            if len(yy)<12:
                return 0.
            traces.append((yy.min(),yy.max()))
        ends.append(np.asarray(traces))
    differences=ends[0]-ends[1]
    return float(max(np.min(abs(differences[:,i])) if np.all(differences[:,i]>0) or
                     np.all(differences[:,i]<0) else 0 for i in (0,1)))


def nomination(image, parts, camera, family='corner'):
    if not camera or camera.get('P') is None:
        return None
    p,rotation=orientation(camera,image.size)
    blurred=np.asarray(image.convert('RGB').filter(ImageFilter.GaussianBlur(1)),dtype=np.float32)
    gray=blurred.mean(axis=2)
    gy,gx=np.gradient(gray)
    winner=None
    for i,part in enumerate(parts):
        normal=part.get('normal_camera')
        if normal is None or not part.get('support') or part.get('corner_junctions'):
            continue
        old_axis=int(np.argmax(abs(rotation.T@normal)))
        if old_axis==1:
            continue
        mask=native_mask(part,image.size)
        ys,xs=np.nonzero(mask)
        if len(xs)<500:
            continue
        x0,x1,y0,y1=int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
        h,w=y1-y0,x1-x0
        if h<35 or w<18 or w>1.6*h:
            continue
        mid=(y0+y1)/2
        y=np.arange(y0,y1)
        local=mask[y0:y1,x0:x1]
        best=None
        for cx in range(x0+7,x1-6,2):
            line=np.rint(canonical_x(cx,mid,y,p[:,1])).astype(int)
            inside=(line>=x0+5)&(line<x1-5)
            yy,xx=y[inside],line[inside]
            ok=mask[yy,xx-5]&mask[yy,xx]&mask[yy,xx+5]
            yy,xx=yy[ok],xx[ok]
            if len(yy)<max(25,.40*h):
                continue
            delta=blurred[yy,xx+2]-blurred[yy,xx-2]
            side_noise=(abs(blurred[yy,xx+5]-blurred[yy,xx+2])+abs(blurred[yy,xx-5]-blurred[yy,xx-2])).mean(1)*.5
            mean_delta=np.mean(delta,axis=0)
            contrast=abs(mean_delta).mean()
            agreement=np.mean(np.sum(delta*mean_delta,axis=1)>0)
            strength=float(np.median(abs(delta).mean(1)-side_noise))
            if contrast<10 or agreement<.74 or strength<7:
                continue
            step=termination_step(local,line,x0) if family=='parallel' else 0.
            if family=='parallel' and step<max(9,.09*h):
                continue
            rank=strength*np.sqrt(len(yy)/h)
            if best is None or rank>best[0]:
                best=(rank,cx,line,len(yy),float(agreement),step)
        if best is None:
            continue
        rank,cx,line,n,agreement,step=best
        left=local & (np.arange(x0,x1)<line[:,None])
        halves=(left,local&~left)
        areas=[int(a.sum()) for a in halves]
        if min(areas)<max(180,.07*sum(areas)):
            continue
        votes=[line_vote(a,gx,gy,p,x0,y0) for a in halves]
        if any(v is None for v in votes):
            continue
        new_axes=[(0,2)[np.argmax(v)] for v in votes]
        margins=[abs(v[0]-v[1]) for v in votes]
        if min(margins)<.035 or min(map(max,votes))<.17:
            continue
        if family=='corner' and new_axes[0]==new_axes[1]:
            continue
        if family=='parallel' and new_axes!=[old_axis,old_axis]:
            continue
        side=new_axes.index(2-old_axis) if family=='corner' else 0
        axis=new_axes[side]
        sub_y,sub_x=np.nonzero(halves[side])
        ray=np.linalg.solve(p[:,:3], [x0+sub_x.mean(),y0+sub_y.mean(),1])
        sign=1 if ray[axis]<0 else -1
        strength=rank*min(margins)
        if winner is None or strength>winner['strength']:
            winner=dict(family=family,termination_pixels=step,index=i,id=part['id'],x=cx,mid=mid,box=[x0,y0,x1,y1],side=side,
                        old_axis=old_axis,axis=axis,sign=sign,new_normal=(sign*rotation[:,axis]).tolist(),
                        areas=areas,votes=votes,margins=margins,edge_strength=rank,edge_rows=n,agreement=agreement,
                        strength=strength,source=bool(min(margins)>=.10 and rank>=12))
    return winner


def apply(image,parts,camera,event,arm):
    if arm=='baseline':
        return parts
    i=event['index']
    part=parts[i]
    assert event['id']==part['id']
    if arm=='direction':
        return parts[:i]+[dict(part,normal_camera=event['new_normal'])]+parts[i+1:]
    p,_=orientation(camera,image.size)
    old=native_mask(part,image.size)
    line=np.rint(canonical_x(event['x'],event['mid'],np.arange(image.height),p[:,1]))
    left=old & (np.arange(image.width)<line[:,None])
    children=[]
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    for j,mask in enumerate((left,old&~left)):
        seed={k:v for k,v in part.items() if k not in ('context_crop','source_crop')}
        if arm=='both' and j==event['side']:
            seed['normal_camera']=event['new_normal']
        child=with_mask(seed,mask,image.size,gray,np.ones(2))
        child['id']=part['id']+'-crease-'+str(j)
        assert np.array_equal(mask,native_mask(child,image.size))
        children.append(child)
    return parts[:i]+children+parts[i+1:]
