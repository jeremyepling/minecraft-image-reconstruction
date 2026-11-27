"""Rank local face-direction and integer-depth alternatives on photographed surface rays."""
from engine import floating_grid_policy as floating
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json

import numpy as np

from engine.surface_model import intrinsic
from engine.visible_horizontal_faces import _surface_masks, _keep_prior_visible
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE

MODEL = Path(__file__).with_name('nearby-face-model.json')
MODEL_SHA = '5b0950f8cfcc7f4355c947df8a542ec7b8f2df9deb2e6847ed1d58170731776b'
DIRECTIONS = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))


def _project(P, xyz):
    p = xyz @ P[:,:3].T + P[:,3]
    return p[...,:2] / np.maximum(.01,p[...,2:]),p[...,2]


def _support_features(points, axis, rgb, P, eye, owner, axes, masks, invdepth):
    # A regular sample on each proposed face measures mask orientation, native
    # texture repeat and disagreement with the current frontmost rendered face.
    grid = (np.arange(6)+.5)/6
    uv = np.stack(np.meshgrid(grid, grid),-1).reshape(-1,2)
    free = [d for d in range(3) if d!=axis]
    xyz = np.repeat(points[:,None,:],len(uv),axis=1).astype(float)
    xyz[:,:,axis] += (eye[axis]>points[:,axis]+1)[:,None]
    xyz[:,:,free] += uv
    xy,z = _project(P,xyz)
    h,w=owner.shape
    def indices(xy):
        x,y = np.floor(xy/4).astype(int).transpose(2,0,1)
        inside = (x>=0)&(x<w)&(y>=0)&(y<h)
        return x.clip(0,w-1),y.clip(0,h-1),inside
    x,y,inside = indices(xy)
    inside &= (z>1)&((points[:,axis]>eye[axis])|(points[:,axis]+1<eye[axis]))[:,None]
    mask = inside & masks[axis,y,x]
    count = np.maximum(1,inside.sum(1))
    owner_depth = invdepth[y,x]
    # Inverse depth tolerance only covers roundoff and coarse-ray sampling here.
    front = ((1/np.maximum(z,.01))>=owner_depth-.0003)
    hit = owner[y,x]>=0
    feat = [mask.sum(1)/count,(mask&front).sum(1)/count,
            (mask&(~hit|(axes[y,x]!=axis))).sum(1)/count,
            (mask&hit&(axes[y,x]==axis)).sum(1)/count,
            inside.sum(1)/len(uv)]
    xp,yp = np.floor(xy).astype(int).transpose(2,0,1)
    def colors(px,py):
        return rgb[py.clip(0,rgb.shape[0]-1),px.clip(0,rgb.shape[1]-1)]
    color = colors(xp,yp)
    flat = color.mean(-1)
    nmask = np.maximum(1,mask.sum(1))
    mean = (flat*mask).sum(1)/nmask
    deviation = np.sqrt((((flat-mean[:,None])**2)*mask).sum(1)/nmask)
    feat.append(np.log1p(deviation))
    similarities = []
    phase_gains = []
    for dim in free:
        for step in (-1,1):
            adjacent = xyz.copy()
            adjacent[:,:,dim] += step
            adjxy,az = _project(P,adjacent)
            xx,yy,valid=indices(adjxy)
            ok=mask & valid & (az>1) & masks[axis,yy,xx]
            n=ok.sum(1)
            ax,ay=np.floor(adjxy).astype(int).transpose(2,0,1)
            same = abs(colors(ax,ay)-color).mean(-1)
            adjacent[:,:,dim] += np.where((xyz[:,:,dim]%1)<.75,.25,-.75)
            other,_=_project(P,adjacent)
            bx,by=np.floor(other).astype(int).transpose(2,0,1)
            rival=abs(colors(bx,by)-color).mean(-1)
            err=(same*ok).sum(1)/np.maximum(1,n)
            gain=((rival-same)*ok).sum(1)/np.maximum(1,n)
            similarities.append(np.where(n>=8,np.exp(-err/12),0))
            phase_gains.append(np.where(n>=8,np.clip(gain/16,-1,2),0))
    feat.extend((np.max(similarities,axis=0),np.mean(similarities,axis=0),
                 np.max(phase_gains,axis=0),np.mean(phase_gains,axis=0)))
    return np.column_stack(feat)


def observations(image, parts, camera, prior):
    if camera['P'] is None or not prior or not parts:
        return None
    P=np.asarray(camera['P'],float)
    P=P/np.linalg.norm(P[2,:3])
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rot=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    xs,ys=np.arange(2,image.width,4),np.arange(2,image.height,4)
    _,masks=_surface_masks(parts,image.size,xs,ys,rot)
    cells=sorted(prior)
    points=np.asarray(cells)
    owner,axes=_frontmost_cells(cells,P,eye,(len(ys),len(xs)),face_axes=True)
    y,x=np.nonzero(owner>=0)
    if not len(y):
        return None
    rays=np.column_stack((xs[x],ys[y],np.ones(len(x))))@np.linalg.inv(P[:,:3]).T
    which=owner[y,x]
    current_axis=axes[y,x]
    samples=np.bincount(which,minlength=len(cells))
    coord=points[which,current_axis]
    boundary=coord+(eye[current_axis]>coord+1)
    invdepth=np.zeros_like(owner,dtype=float)
    invdepth[y,x]=rays[np.arange(len(rays)),current_axis]/(boundary-eye[current_axis])
    raw=[]
    for axis in range(3):
        active=masks[axis,y,x] & (abs(rays[:,axis])>1e-5)
        source=which[active]
        ray=rays[active]
        prior_axis=current_axis[active]
        if not len(source):
            continue
        base=points[source]
        for delta in (-2,-1,0,1,2):
            c=base[:,axis]+delta
            face=c+(eye[axis]>c+1)
            d=(face-eye[axis])/ray[:,axis]
            target=np.floor(eye+ray*d[:,None]).astype(int)
            target[:,axis]=c
            valid=(d>1)&((c>eye[axis])|(c+1<eye[axis]))&(abs(target-base).max(axis=1)<=4)
            valid &= (floating.active() | (target[:,1]>=0))&(target!=base).any(axis=1)
            if not np.any(valid):
                continue
            # Aggregate observed support by source and reprojected coordinate.
            keys=np.column_stack((source[valid],target[valid])).astype(np.int32)
            unique,inv,n=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
            flipped=np.bincount(inv,weights=(prior_axis[valid]!=axis),minlength=len(n))
            for (s,tx,ty,tz),count,changed in zip(unique,n,flipped,strict=True):
                t=(int(tx),int(ty),int(tz))
                if count>=3 and count>=.06*samples[s] and t not in prior:
                    raw.append((int(s),t,axis,delta,int(count),float(changed/count)))
    if not raw:
        return None
    targets=sorted({t for s,t,a,d,n,g in raw})
    pos=cells+targets
    index={c:i for i,c in enumerate(pos)}
    src,dst,axis,delta,n,changed=np.array([(s,index[t],a,d,n,g) for s,t,a,d,n,g in raw]).T
    src,dst,axis,delta,n=(v.astype(int) for v in (src,dst,axis,delta,n))
    ij=np.column_stack((src,dst,axis))
    points=np.array(pos)
    rgb=np.asarray(image.convert('RGB')).astype(np.float32)
    face=np.stack([np.concatenate([_support_features(points[i:i+128],a,rgb,P,eye,owner,axes,masks,invdepth)
                                   for i in range(0,len(points),128)]) for a in range(3)],axis=1)
    relative=(points[:,None,:]+np.asarray(DIRECTIONS)[None]).reshape(-1,3)
    near=np.array([tuple(p) in prior for p in relative],float).reshape(-1,3,2).sum(2)
    # Work in a face-relative frame. Both vertical world axes share features.
    horizontal=axis==1
    adjacent=np.sort(near[dst],axis=1)
    displaced=points[dst]-points[src]
    cols=[np.abs(delta),np.linalg.norm(displaced,axis=1),n/np.maximum(1,samples[src]),np.log1p(n),
          changed,horizontal,np.sign(delta)*np.sign(points[src,axis]-eye[axis]),
          near[src,axis],near[dst,axis],near[src].sum(1)-near[src,axis],
          near[dst].sum(1)-near[dst,axis],*adjacent.T,
          np.log1p(np.bincount(dst,minlength=len(pos))[dst]),np.log1p(samples[src])]
    # Compare both source and destination orientations, plus the best observed
    # orientation at the prior cell, to decide a keep, turn, or depth move.
    feats=np.column_stack([*cols,face[dst,axis],face[src,axis],face[src].max(1),
                           face[dst,axis]-face[src,axis]]).astype(np.float32)
    return dict(points=points,edges=ij,features=feats,base_count=len(cells))


def predict_forest(model, features):
    pred=np.full(len(features),model['bias'])
    for tree in model['trees']:
        nodes=np.zeros(len(features),dtype=int)
        for splits in tree['levels']:
            f,t=np.asarray(splits).T
            nodes=2*nodes+(features[np.arange(len(features)),f[nodes].astype(int)]>t[nodes])
        pred+=np.asarray(tree['leaves'])[nodes]
    return 1/(1+np.exp(-np.clip(pred,-25,25)))


def choose(data, model):
    edges=data['edges']
    source,target,axis=edges.T
    new=predict_forest(model['destination'],data['features'])
    old=predict_forest(model['source'],data['features'])
    # Keep the best evidence for each new cube. A source can only be removed
    # once, even where its texture projects onto several alternative cells.
    accept=new>=model['add']
    remove=accept & (old<=model['drop']) & (new-old>=model['margin'])
    return set(zip(source[remove],target[remove])),set(target[accept]),set(zip(target[accept],axis[accept]))


def apply(image, parts, camera, prior, data, decisions):
    if data is None:
        return prior
    removal_pairs,added,witness=decisions
    if not added:
        return prior
    points=[tuple(p) for p in data['points'].tolist()]
    candidate={points[i] for i in added}
    retired={points[i] for i,_ in removal_pairs}
    remain=prior-retired
    P=np.asarray(camera['P'],float)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    ordered=sorted(remain|candidate)
    owner,axes=_frontmost_cells(ordered,P,eye,tuple(len(np.arange(2,n,4)) for n in image.size[::-1]),face_axes=True)
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])/np.linalg.norm(P[2,:3])
    rot=np.linalg.solve(intrinsic(image.size,focal),P[:,:3]/np.linalg.norm(P[2,:3]))
    _,masks=_surface_masks(parts,image.size,np.arange(2,image.width,4),np.arange(2,image.height,4),rot)
    accepted=set()
    ids={p:i for i,p in enumerate(ordered)}
    for a in range(3):
        hit=(owner>=0)&(axes==a)&masks[a]
        hits=np.bincount(owner[hit],minlength=len(ordered))
        accepted.update(points[t] for t,axis in witness if a==axis and hits[ids[points[t]]]>=3)
    if not accepted:
        return prior
    # Restore the original source if its replacement did not keep a visible
    # witness after joint rasterization.
    replaced={points[s] for s,t in removal_pairs if points[t] in accepted}
    remain=prior-replaced
    result=_keep_prior_visible(image.size,P,eye,remain,accepted)
    final=result-remain
    if final!=accepted or remain!=prior-retired:
        # Restored sources can cover new faces. Retest the final scene.
        ordered=sorted(result)
        owner,axes=_frontmost_cells(ordered,P,eye,tuple(len(np.arange(2,n,4)) for n in image.size[::-1]),face_axes=True)
        ids={p:i for i,p in enumerate(ordered)}
        visible=set()
        for a in range(3):
            hit=(owner>=0)&(axes==a)&masks[a]
            counts=np.bincount(owner[hit],minlength=len(ordered))
            visible.update(points[t] for t,axis in witness if a==axis and points[t] in final and counts[ids[points[t]]]>=3)
        if visible!=accepted:
            return prior
    return result


@lru_cache(maxsize=2)
def _weights(raw):
    if sha256(raw.replace(b'\r\n',b'\n')).hexdigest()!=MODEL_SHA:
        return None
    return json.loads(raw)


def recover(image, parts, camera, prior, *, atlas_path=None):
    """Apply the frozen chooser or retain the supplied placement when its inputs are unavailable."""
    if not prior or not parts or camera.get('P') is None:
        return prior
    try:
        model=_weights(MODEL.read_bytes())
    except OSError:
        model=None
    if model is None:
        return prior
    data=observations(image,parts,camera,prior)
    if data is not None and model.get('atlas'):
        from engine.nearby_face_art import enrich
        data=enrich(image,parts,camera,data,atlas_path=atlas_path)
    return apply(image,parts,camera,prior,data,choose(data,model)) if data is not None else prior

