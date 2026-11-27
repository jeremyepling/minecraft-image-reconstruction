"""Fit one integer horizontal plane to an observed roof perimeter and grounded wall tops."""
from engine import floating_grid_policy as floating
from collections import defaultdict

import numpy as np

from engine.surface_model import intrinsic
from engine.visible_horizontal_faces import _surface_masks, _keep_prior_visible
from engine.visible_placement import _frontmost_cells, _sample_face


def rays(x, y, inverse):
    return np.column_stack((x,y,np.ones(len(x)))) @ inverse.T


def pixels(mask, step):
    y,x=np.nonzero(mask)
    return x*step+.5, y*step+.5


def tops(masks, axes, eye, inverse, step):
    """Recover support termination points by intersecting a grounded wall plane."""
    if floating.active():
        return {}
    union=np.logical_or.reduce(masks)
    floor=np.where(union,np.arange(union.shape[0])[:,None],-1).max(axis=0)
    tips=defaultdict(set)
    for mask,axis in zip(masks,axes):
        if axis not in (0,2):
            continue
        x=np.flatnonzero(mask.any(0))
        if len(x)<5:
            continue
        y0=mask[:,x].argmax(0)
        y1=mask.shape[0]-1-mask[::-1,x].argmax(0)
        base=floor[x]-y1<=1
        if base.sum()<max(5,.2*len(x)):
            continue
        directions=rays(x[base]*step+.5,(y1[base]+.5)*step+.5,inverse)
        t=-eye[1]/directions[:,1]
        intercept=eye[axis]+directions[:,axis]*t
        good=(t>1)&(t<145)
        if good.sum()<5:
            continue
        levels,counts=np.unique(np.rint(intercept[good]).astype(int),return_counts=True)
        level=levels[counts.argmax()]
        if (np.abs(intercept[good]-level)<.29).mean()<.55:
            continue
        directions=rays(x*step+.5,(y0-.5)*step+.5,inverse)
        t=(level-eye[axis])/directions[:,axis]
        xyz=eye+t[:,None]*directions
        y=np.rint(xyz[:,1])
        valid=(t>1)&(t<145)&(abs(xyz[:,1]-y)<.25)&(y>=1)&(y<40)
        for cx,cy,cz in xyz[valid]:
            tips[int(round(cy))].add((int(np.floor(cx)),int(np.floor(cz))))
    return tips


def rim_samples(mask, rgb, step, inverse):
    """Snap mask transitions to nearby RGB edges in the native image."""
    samples=[]
    ys,xs=np.nonzero(mask[:,1:]!=mask[:,:-1])
    vs,us=np.nonzero(mask[1:,:]!=mask[:-1,:])
    for x,y,dx,dy in ((xs,ys,1,0),(us,vs,0,1)):
        x=x*step+dx
        y=y*step+dy
        valid=(x>=5)&(x<rgb.shape[1]-5)&(y>=5)&(y<rgb.shape[0]-5)
        x,y=x[valid],y[valid]
        if not len(x):
            continue
        offsets=np.arange(-4,5)
        xx=x[:,None]+(offsets*dx)
        yy=y[:,None]+(offsets*dy)
        strength=np.abs(rgb[yy,xx]-rgb[yy+dy,xx+dx]).sum(axis=2)
        # Prefer a nearby image transition if multiple brick edges have similar contrast.
        strength=strength/(1+.17*abs(offsets))
        top=strength.argmax(1)
        valid=strength.max(1)>=10
        x=x[valid]+offsets[top[valid]]*dx+.5+.5*dx
        y=y[valid]+offsets[top[valid]]*dy+.5+.5*dy
        if len(x):
            samples.extend(zip(x,y))
    if len(samples)<10:
        return None
    xy=np.array(samples)
    if len(xy)>1200:
        xy=xy[np.linspace(0,len(xy)-1,1200,dtype=int)]
    return rays(*xy.T,inverse)


def hypotheses(mask,rim,inverse,eye,tips,prior,step):
    x,y=pixels(mask,step)
    if len(x)>2600:
        take=np.linspace(0,len(x)-1,2600,dtype=int)
        x,y=x[take],y[take]
    dirs=rays(x,y,inverse)
    results=[]
    for height in floating.roof_heights(prior, 39, 4):
        if abs(height-eye[1])<.05:
            continue
        t=(height-eye[1])/dirs[:,1]
        if ((t>1)&(t<130)).mean()<.97:
            continue
        dist=(height-eye[1])/rim[:,1]
        points=eye[[0,2]]+dist[:,None]*rim[:,[0,2]]
        points=points[(dist>1)&(dist<130)]
        if len(points)<10 or np.ptp(points,axis=0).max()>48:
            continue
        # A true horizontal perimeter tracks X- or Z-integer lines along its observed edges.
        error=np.min(abs(points-np.rint(points)),axis=1)
        edge=float(np.mean(np.exp(-.5*(error/.095)**2)))
        uv=np.floor(eye[[0,2]]+t[:,None]*dirs[:,[0,2]]).astype(int)
        bins,n=np.unique(uv,axis=0,return_counts=True)
        footprint={tuple(xz.tolist()) for xz in bins[n>=2]}
        if not footprint:
            continue
        outer={(a+dx,b+dz) for a,b in footprint for dx,dz in ((0,0),(-1,0),(1,0),(0,-1),(0,1))}
        block=height-int(eye[1]>height)
        anchors=outer & (tips.get(block,set()) | tips.get(height,set()))
        # A second estimate comes from stacked prior blocks supporting this footprint.
        columns={xz for xz in footprint if (xz[0],block-1,xz[1]) in prior
                 and (xz[0],block-2,xz[1]) in prior}
        strength=(edge+.15*min(1,len(anchors)/4)+.085*min(1,len(columns)/3))
        results.append((strength,height,block,edge,len(anchors),len(columns)))
    return sorted(results,reverse=True)


def select(image,parts,camera,prior):
    P=np.asarray(camera['P'],float)
    P=P/np.linalg.norm(P[2,:3])
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    inverse=np.linalg.inv(P[:,:3])
    focal=np.linalg.norm(P[0,:3]-image.width/2*P[2,:3])
    rotation=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    step=2
    xs=np.arange(.5,image.width,step)
    ys=np.arange(.5,image.height,step)
    kept=[p for p in parts if p.get('support') and p.get('normal_camera') is not None]
    if not kept:
        return [],[],P,eye,rotation
    masks=[_surface_masks([p],image.size,xs,ys,rotation)[0] for p in kept]
    axes=[int(np.argmax(abs(rotation.T@p['normal_camera']))) for p in kept]
    tips=tops(masks,axes,eye,inverse,step)
    rgb=np.asarray(image.convert('RGB'),dtype=np.int16)
    decisions=[]
    accepted=[]
    for part,axis,mask in zip(kept,axes,masks):
        if axis!=1 or mask.sum()<30:
            continue
        rim=rim_samples(mask,rgb,step,inverse)
        if rim is None:
            continue
        options=hypotheses(mask,rim,inverse,eye,tips,prior,step)
        if not options:
            continue
        score,height,block,edge,anchors,columns=options[0]
        margin=score-options[1][0] if len(options)>1 else score
        take=bool(edge>=.52 and margin>=.08 and (anchors>=2 or columns>=2))
        d=dict(part=part['id'],plane_y=height,block_y=block,accepted=take,rim=edge,
               margin=float(margin),anchors=anchors,columns=columns,pixels=int(mask.sum()*step**2))
        decisions.append(d)
        if take:
            accepted.append((part,d))
    return decisions,accepted,P,eye,rotation


def materialize(image,parts,accepted,P,eye,rotation,prior):
    """Require original pixels of this patch to witness each placed horizontal face."""
    inverse=np.linalg.inv(P[:,:3])
    native=P.copy()
    native[:2]*=4
    xs=np.arange(image.width)+.5
    ys=np.arange(image.height)+.5
    union,axis_masks=_surface_masks(parts,image.size,xs,ys,rotation)
    proposals={}
    for part,d in accepted:
        mask=_surface_masks([part],image.size,xs,ys,rotation)[0]
        v,u=np.nonzero(mask[::2,::2])
        dirs=rays(u*2+.5,v*2+.5,inverse)
        t=(d['plane_y']-eye[1])/dirs[:,1]
        valid=(t>1)&(t<130)
        uv=np.floor(eye[[0,2]]+dirs[valid][:,[0,2]]*t[valid,None]).astype(int)
        positions,counts=np.unique(uv,axis=0,return_counts=True)
        for a,b in positions[counts>=3].tolist():
            c=(a,d['block_y'],b)
            if c in prior:
                continue
            hit=_sample_face(native,c,1,1 if eye[1]>d['plane_y'] else -1,mask.shape)
            if not hit:
                continue
            box,visible,*_=hit
            n=np.count_nonzero(visible&mask[box])
            if n>=12 and n>=.69*visible.sum():
                proposals[c]=mask
    if not proposals:
        return prior
    scene=sorted(prior|proposals.keys())
    front,axes=_frontmost_cells(scene,native,eye,union.shape,face_axes=True)
    witnessed=front>=0
    total=np.bincount(front[witnessed],minlength=len(scene))
    ins=np.bincount(front[witnessed & union],minlength=len(scene))
    match=np.take_along_axis(axis_masks,np.maximum(0,axes)[None],axis=0)[0]
    aligned=np.bincount(front[witnessed & match],minlength=len(scene))
    add=set()
    for i,c in enumerate(scene):
        if c not in proposals or total[i]<12 or ins[i]<.985*total[i] or aligned[i]<.85*total[i]:
            continue
        hit=_sample_face(native,c,1,1 if eye[1]>c[1]+1 else -1,union.shape)
        box=hit[0]
        observed=(front[box]==i)&(axes[box]==1)&proposals[c][box]
        if observed.sum()>=max(12,.15*total[i]):
            add.add(c)
    return _keep_prior_visible(image.size,P,eye,prior,add) if add else prior


def recover(image,parts,counts,camera,prior):
    if not prior or not parts or camera.get('P') is None:
        return prior,dict(height_decisions=[],primary_height=None)
    decisions,accepted,P,eye,rotation=select(image,parts,camera,prior)
    changed=materialize(image,parts,accepted,P,eye,rotation,prior) if accepted else prior
    primary=max((d for _,d in accepted),key=lambda d:d['pixels']*d['rim'],default=None)
    return changed,dict(height_decisions=decisions,primary_height=primary)
