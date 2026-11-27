"""Test coherent depth changes for image-supported, connected wall patches."""
from collections import defaultdict

import numpy as np

from benchmark.wall_source_projection import FaceRenderer, Scene, STEP
from engine.visible_horizontal_faces import _keep_prior_visible


def panel_groups(scene, parts, size):
    """Assign exposed front cells to a detected mask and split on depth or local appearance."""
    h,w=scene.shape
    labels=np.full((h,w),-1,int)
    for n,part in enumerate(parts):
        sup=part.get('support')
        if not sup:
            continue
        width,height=sup['size']
        mask=np.zeros((height,width),bool)
        for y,l,r in sup['runs']:
            mask[y,l:r]=True
        iy=((np.arange(h)*STEP+.5)*height/size[1]).astype(int)
        ix=((np.arange(w)*STEP+.5)*width/size[0]).astype(int)
        labels[mask[np.ix_(iy,ix)]]=n
    labels=labels.ravel()
    rgb=scene.rgb.reshape(-1,3)
    planes=defaultdict(dict)
    for n,idx in scene.owned.items():
        cell=scene.cells[n]
        for axis in (0,2):
            face=idx[(scene.axes[idx]==axis)&scene.masks[axis,idx]&(labels[idx]>=0)]
            if len(face)<8:
                continue
            keys,counts=np.unique(labels[face],return_counts=True)
            j=counts.argmax()
            if counts[j]<.65*len(face):
                continue
            patch=face[labels[face]==keys[j]]
            color=np.median(rgb[patch],0)
            returns=int(np.count_nonzero((scene.axes[idx]!=axis)&scene.masks[scene.axes[idx],idx]))
            planes[axis,cell[axis],int(keys[j])][cell]=(color,returns)
    proposals={}
    for (axis,level,region),cells in planes.items():
        for compare_color in (False,True):
            pending=set(cells)
            while pending:
                seed=min(pending)
                pending.remove(seed)
                group={seed}
                todo=[seed]
                while todo:
                    c=todo.pop()
                    for a in (1,2-axis):
                        for step in (-1,1):
                            v=list(c)
                            v[a]+=step
                            v=tuple(v)
                            if v in pending and (not compare_color or
                                    abs(cells[v][0]-cells[seed][0]).max()<=10):
                                pending.remove(v)
                                group.add(v)
                                todo.append(v)
                if 2<=len(group)<=12:
                    returns=sum(cells[c][1] for c in group)
                    proposals[axis,tuple(sorted(group))]=returns
    return sorted(proposals,key=lambda k:(proposals[k]==0,-min(proposals[k],128),len(k[1]),k))[:20]


class Stack:
    """Exact next camera intersection on rays whose first visible block is removed."""
    def __init__(self,scene):
        faces=[scene.faces(c) for c in scene.cells]
        i=np.concatenate([v[0] for v in faces])
        a=np.concatenate([v[1] for v in faces])
        z=np.concatenate([v[2] for v in faces])
        ids=np.repeat(np.arange(len(faces)),[len(v[0]) for v in faces])
        order=np.lexsort((ids,-z,i))
        self.ray=np.append(i[order],-1)
        self.owner=np.append(ids[order],-1)
        self.axis=np.append(a[order],-1)
        self.depth=np.append(z[order],0.)

    def uncover(self,scene,idx,removed):
        ids=scene.owner[idx].copy()
        axes=scene.axes[idx].copy()
        z=scene.dep[idx].copy()
        loc=np.flatnonzero(np.isin(ids,removed))
        if len(loc):
            target=idx[loc]
            pos=np.searchsorted(self.ray[:-1],target)
            while True:
                same=self.ray[pos]==target
                drop=same & np.isin(self.owner[pos],removed)
                if not drop.any():
                    break
                pos[drop]+=1
            ids[loc]=np.where(same,self.owner[pos],-1)
            axes[loc]=np.where(same,self.axis[pos],-1)
            z[loc]=np.where(same,self.depth[pos],0.)
        return ids,axes,z


def shifts(scene,group,axis):
    center=np.mean(group,axis=0)+.5
    for depth in (-1,1):
        shift=np.zeros(3,int)
        shift[axis]=depth
        yield shift
        # Preserve the observed center of the patch after moving the depth plane.
        if abs(center[axis]-scene.eye[axis])>.01:
            recentered=np.rint((center-scene.eye)*depth/(center[axis]-scene.eye[axis])).astype(int)
            if np.any(recentered!=shift) and abs(recentered).max()<=2:
                yield recentered


def render(scene,stack,group,ids,dest,axis):
    fragments=[scene.faces(c) for c in dest]
    union=np.unique(np.concatenate([scene.owned[n] for n in ids]+[v[0] for v in fragments]))
    owners,axes,z=stack.uncover(scene,union,ids)
    original=scene.owner[union]
    for j,(ray,a,depth) in enumerate(fragments):
        ix=np.searchsorted(union,ray)
        keep=depth>z[ix]+1.e-10
        loc=ix[keep]
        owners[loc],axes[loc],z[loc]=len(scene.cells)+j,a[keep],depth[keep]
    changed=(original!=owners)|(scene.axes[union]!=axes)
    idx=union[changed]
    owners,axes=owners[changed],axes[changed]
    if len(idx)<16:
        return None
    new=owners>=len(scene.cells)
    seen=np.bincount(owners[new]-len(scene.cells),minlength=len(dest))
    inside=scene.union[idx]
    support=np.bincount(owners[new&inside]-len(scene.cells),minlength=len(dest))
    if (support<4).any() or (support<.80*seen).any():
        return None
    # Do not erase the last source ray of any surrounding block.
    old=scene.owner[idx]
    lost=old[(old>=0)&~np.isin(old,ids)]
    lost_id,count=np.unique(lost,return_counts=True)
    if any(k==len(scene.owned[n]) and not (owners==n).any() for n,k in zip(lost_id,count)):
        return None
    orig_axis=scene.axes[idx]
    def mismatches(own,a):
        fg=own>=0
        sil=np.count_nonzero(fg!=inside)
        bad=fg&inside&scene.masks[:,idx].any(0)&~scene.masks[a,idx]
        return sil,int(bad.sum())
    old_sil,old_norm=mismatches(old,orig_axis)
    sil,norm=mismatches(owners,axes)
    return idx,owners,axes,dict(pixels=len(idx),sil=int(old_sil-sil),normal=old_norm-norm,
                                returns=int((new&inside&(axes!=axis)&scene.masks[axes,idx]).sum()))


def texture(scene,renderer,idx,owners,axes,dest):
    """Compare only changed first-hit pixels independently fitted in both alternatives."""
    use=scene.union[idx] & (scene.owner[idx]>=0) & (owners>=0) & renderer.test[idx] & renderer.fitted[idx]
    improvement=0.
    tested=0
    for n in np.unique(owners[use]):
        for a in np.unique(axes[(owners==n)&use]):
            rows=(owners==n)&(axes==a)
            test=rows&use
            if n>=len(scene.cells):
                cell=dest[n-len(scene.cells)]
                mdl=renderer.fit(idx[rows],cell,a)
            else:
                cell=scene.cells[n]
                mdl=renderer.face_models.get((n,a))
            pred,fitted=renderer.draw(idx[test],cell,a,mdl)
            if fitted:
                ray=idx[test]
                source=renderer.source[ray]
                improvement+=float((abs(renderer.image[ray]-source).mean(1)-abs(pred-source).mean(1)).sum())
                tested+=len(ray)
    return improvement/max(1,tested),tested


def trials(image,parts,camera,prior):
    if not prior or not parts or camera.get('P') is None:
        return None,[]
    scene=Scene(image,parts,camera,prior)
    groups=panel_groups(scene,parts,image.size)
    index={c:i for i,c in enumerate(scene.cells)}
    stack=Stack(scene)
    renderer=None
    options=[]
    for axis,group in groups:
        ids=[index[c] for c in group]
        for shift in shifts(scene,group,axis):
            dest=[tuple(map(int,np.array(c)+shift)) for c in group]
            if min(c[1] for c in dest)<0 or not set(dest).isdisjoint(prior):
                continue
            result=render(scene,stack,group,ids,dest,axis)
            if result is None:
                continue
            idx,owners,axes,score=result
            score.update(before=group,after=dest,axis=axis,texture_gain=None,texture_pixels=0)
            if score['sil']>=0 and score['normal']>=0 and score['returns']>=6 and score['sil']+.65*score['normal']>=3:
                if renderer is None:
                    renderer=FaceRenderer(scene)
                score['texture_gain'],score['texture_pixels']=texture(scene,renderer,idx,owners,axes,dest)
            options.append(score)
    return scene,options


def choose(options):
    good=[o for o in options if o['texture_pixels']>=24 and o['texture_gain']>=.50 and
          (o['sil']+.65*o['normal'])/o['pixels']>=.01]
    return max(good,key=lambda o:(o['sil']+.65*o['normal'])/o['pixels']+min(5.,o['texture_gain'])/20) if good else None


def materialize(image,scene,prior,o):
    if o is None:
        return prior
    rest=prior-set(o['before'])
    wanted=rest|set(o['after'])
    actual=_keep_prior_visible(image.size,scene.P,scene.eye,rest,set(o['after']))
    return wanted if actual==wanted else prior


def recover(image,parts,counts,camera,prior):
    scene,options=trials(image,parts,camera,prior)
    picked=choose(options)
    result=materialize(image,scene,prior,picked)
    return result,dict(options=len(options),eligible=picked is not None,changed=result!=prior)
