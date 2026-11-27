"""Compete local wall, top and underside interpretations on identical photographed strips."""
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFilter

from benchmark.framed_geometry_order import ATLAS
from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.projected_appearance import Scene, STEP
from engine.thin_atlas_faces import _corr, _mips
from engine.visible_horizontal_faces import _keep_prior_visible


def patches(scene):
    """Group unclaimed mask samples next to the projected rim of an existing visible cell."""
    h,w=scene.shape
    occupied=(scene.owner>=0).reshape(h,w)
    nearby=np.array(Image.fromarray(occupied.astype('uint8')).filter(ImageFilter.MaxFilter(11)),bool)
    need=scene.union.reshape(h,w)&~occupied&nearby
    yy,xx=np.nonzero(need)
    # Split intersecting branches by the original mask orientation and a small
    # spatial window. Source texture and shape can then override this label.
    ori=np.where(scene.masks[1],1,np.where(scene.masks[0],0,np.where(scene.masks[2],2,3)))
    ww=(w+19)//20
    keys=((yy//20)*ww+xx//20)*4+ori[yy*w+xx]
    order=np.argsort(keys,kind='stable')
    for group in np.split(order,np.flatnonzero(np.diff(keys[order]))+1):
        if len(group)<9:
            continue
        ys,xs=yy[group],xx[group]
        remaining=set(map(tuple,np.column_stack((ys,xs)).tolist()))
        while remaining:
            start=remaining.pop()
            comp=[start]
            for y,x in comp:
                for other in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if other in remaining:
                        remaining.remove(other)
                        comp.append(other)
            if len(comp)<9:
                continue
            y,x=np.array(comp).T
            cov=np.linalg.eigvalsh(np.cov(x,y))
            if cov[1]<4 or cov[1]<2.5*max(.5,cov[0]):
                continue
            yield y*w+x,int(ori[y[0]*w+x[0]])


def choices(scene,idx,atlas,maps):
    h,w=scene.shape
    y,x=np.divmod(idx,w)
    y0,y1=max(0,y.min()-6),min(h,y.max()+7)
    x0,x1=max(0,x.min()-6),min(w,x.max()+7)
    near=np.unique(scene.owner.reshape(h,w)[y0:y1,x0:x1])
    anchors=[scene.cells[i] for i in near if i>=0]
    if not anchors:
        return []
    palette=scene.rgb.reshape(-1,3)[idx].astype(float)
    gray=palette.mean(1)
    if gray.std()<1.2:
        return []
    mean=palette.mean(0)
    gain=np.clip((atlas.means@mean)/np.maximum(1,(atlas.means**2).sum(1)),.28,1.18)
    costs=abs(gain[:,None]*atlas.means-mean).mean(1)+2*abs(np.log(gain/.8))
    short=np.argpartition(costs,8)[:8]
    templates=maps[short]
    rays=np.column_stack((x*STEP+.5,y*STEP+.5,np.ones(len(x))))@scene.inv.T
    boundary=[]
    neighboring_color=[]
    image=scene.rgb.reshape(-1,3)
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        adj=np.clip(y+dy,0,h-1)*w+np.clip(x+dx,0,w-1)
        use=(scene.owner[adj]>=0)|~scene.union[adj]
        # A geometric first-hit boundary alone is insufficient. Demand a
        # change of brightness/color at the same photographed rim.
        use &= abs(palette-image[adj]).mean(1)>=3
        boundary.extend(np.flatnonzero(use).tolist())
        neighboring_color.extend((dy,dx) for _ in range(int(use.sum())))
    if len(boundary)<2:
        return []
    boundary=np.array(boundary)
    dy,dx=np.array(neighboring_color).T
    rimrays=np.column_stack(((x[boundary]+.5*dx)*STEP+.5,
                             (y[boundary]+.5*dy)*STEP+.5,np.ones(len(dx))))@scene.inv.T
    folds=((x//2+y//2)%2).astype(bool)
    if min(folds.sum(),(~folds).sum())<4:
        return []
    result=[]
    for axis in (0,1,2):
        if np.min(abs(rays[:,axis]))<1.e-6:
            continue
        free=[i for i in range(3) if i!=axis]
        levels=sorted({a[axis]+d for a in anchors for d in (-2,-1,0,1,2)})
        for level in levels:
            if level<0 and axis==1 or level<=scene.eye[axis]<=level+1:
                continue
            sign=1 if scene.eye[axis]>level+1 else -1
            plane=level+(sign>0)
            distance=(plane-scene.eye[axis])/rays[:,axis]
            if distance.min()<1 or distance.max()>130:
                continue
            world=scene.eye+distance[:,None]*rays
            xyz=np.floor(world).astype(int);xyz[:,axis]=level
            cells,count=np.unique(xyz,axis=0,return_counts=True)
            supported=set()
            for c in cells[count>=3]:
                ct=tuple(map(int,c))
                if ct[1]<0 or ct in scene.prior:
                    continue
                if any(sum(abs(a[i]-ct[i]) for i in range(3))<=2 for a in anchors):
                    supported.add(ct)
            if not supported:
                continue
            # Score the projected atlas on these exact pixels at every
            # feasible axis/level. Fit the sprite on one checkerboard of
            # patches, then confirm phase independently on the other.
            tex=np.floor(world[:,free]*16).astype(int)%16
            key=tex[:,0]+16*tex[:,1]
            vals,grp,n=np.unique(key[~folds],return_inverse=True,return_counts=True)
            if len(vals)<6 or len(np.unique(key[folds]))<5:
                continue
            obs=(np.bincount(grp,weights=gray[~folds])/n).astype('float32')
            fit=_corr(templates[...,vals],obs)-costs[short,None,None]/60
            ti,mip,rot=np.unravel_index(fit.argmax(),fit.shape)
            template=templates[ti,mip,rot]
            held=float(_corr(template[key[folds]],gray[folds]))
            whole=float(_corr(template[key],gray))
            other=scene.eye+(plane-scene.eye[axis])*rimrays/rimrays[:,[axis]]
            residual=np.min(abs(other[:,free]-np.rint(other[:,free])),axis=1)
            edge=float(np.mean(residual<.10))
            score=min(whole,held)+.12*edge-costs[short[ti]]/75
            result.append(dict(axis=axis,sign=sign,level=level,cells=supported,rim=edge,
                               score=float(score),held=held,corr=whole,texels=len(np.unique(key))))
    return result


def interpret(option):
    return ('top' if option['sign']>0 else 'underside') if option['axis']==1 else 'wall'


def analyze(image,parts,camera,prior):
    atlas=_atlas('26.3',ATLAS)
    if not prior or not parts or camera.get('P') is None or atlas is None or atlas.provenance.get('atlas_sha256')!=ATLAS_SHA256:
        return None,[],Counter()
    scene=Scene(image.convert('RGB'),parts,camera,prior)
    scene.prior=prior
    maps=_mips(atlas)
    proposals=[]
    stats=Counter()
    for idx,upstream in patches(scene):
        stats['source_patches']+=1
        options=choices(scene,idx,atlas,maps)
        if not options:
            continue
        options.sort(key=lambda row:row['score'],reverse=True)
        best=options[0]
        different=[r['score'] for r in options if r['axis']!=best['axis']]
        other=[r['score'] for r in options[1:]]
        kind=interpret(best)
        classes={interpret(r) for r in options}
        for c in classes: stats['candidate_'+c]+=1
        if 'wall' in classes and len(classes)>1: stats['wall_horizontal_competitions']+=1
        stats['winner_'+kind]+=1
        best=dict(best,axis_margin=best['score']-max(different,default=best['score']),
                  level_margin=best['score']-max(other,default=best['score']))
        proposals.append((idx,upstream,options,best))
    return scene,proposals,stats


def materialize(scene,nominated):
    """Recheck every candidate against jointly rendered first-visible rays and the opening."""
    if not nominated:
        return set()
    ids,axes,deps=scene.owner.copy(),scene.axes.copy(),scene.dep.copy()
    ordered=sorted(nominated)
    for i,c in enumerate(ordered):
        pix,axis,d=scene.faces(c)
        win=d>deps[pix]+1.e-8
        target=pix[win]
        ids[target]=len(scene.cells)+i
        axes[target]=axis[win];deps[target]=d[win]
    accepted=set()
    for i,c in enumerate(ordered):
        pix,_,_=scene.faces(c)
        pix=pix[ids[pix]==len(scene.cells)+i]
        if not len(pix) or np.any(~scene.union[pix]&(scene.owner[pix]<0)):
            continue
        votes=sum(np.count_nonzero(np.isin(pix[axes[pix]==a],list(idx))) for a,idx in nominated[c].items())
        if votes>=5 and votes>=.26*len(pix):
            accepted.add(c)
    if accepted==set(nominated):
        return _keep_prior_visible(scene.size,scene.P,scene.eye,scene.prior,accepted)-scene.prior
    return materialize(scene,{c:nominated[c] for c in accepted})


def donor_match(scene,c,axis,idx):
    """A neighbor must expose the same texture phase on its own first-hit face."""
    h,w=scene.shape
    pix,axes,dep=scene.faces(c)
    use=(axes==axis)&(dep>scene.dep[pix]+1.e-8)&np.isin(pix,list(idx))
    y,x=np.divmod(pix[use],w)
    if len(y)<6:
        return False
    rays=np.column_stack((x*STEP+.5,y*STEP+.5,np.ones(len(x))))@scene.inv.T
    plane=c[axis]+int(scene.eye[axis]>c[axis]+1)
    xyz=scene.eye+(plane-scene.eye[axis])*rays/rays[:,[axis]]
    for tangent in [v for v in range(3) if v!=axis]:
        for d in (-2,-1,1,2):
            donor=list(c);donor[tangent]+=d
            if tuple(donor) not in scene.prior:
                continue
            points=xyz.copy()
            points[:,tangent]+=d
            proj=points@scene.P[:,:3].T+scene.P[:,3]
            projected=np.rint((proj[:,:2]/proj[:,[2]]-.5)/STEP).astype(int)
            u,v=projected.T
            ok=(u>=0)&(u<w)&(v>=0)&(v<h)
            a=(v[ok]*w+u[ok])
            visible=(scene.owner[a]==scene.cells.index(tuple(donor)))&(scene.axes[a]==axis)
            if visible.sum()<max(6,.55*len(x)):
                continue
            original=scene.rgb[y[ok][visible],x[ok][visible]].astype(float)
            expected=scene.rgb[v[ok][visible],u[ok][visible]].astype(float)
            g1,g2=original.mean(1),expected.mean(1)
            if min(g1.std(),g2.std())>1.3 and _corr(g1,g2)>.70 and np.mean(abs(original-expected))<8:
                return True
    return False


def select(image,parts,counts,camera,prior,strict=True,horizontal=False,donor=False):
    scene,patches,stats=analyze(image,parts,camera,prior)
    votes=defaultdict(lambda:defaultdict(set))
    for idx,upstream,options,r in patches:
        if horizontal and r['axis']!=1:
            continue
        if r['held']<(.72 if strict else .55) or r['corr']<.65 or r['rim']<.40:
            continue
        if r['axis_margin']<(.14 if strict else .06) or r['level_margin']<(.055 if strict else .015):
            continue
        for c in r['cells']:
            if donor and not donor_match(scene,c,r['axis'],idx):
                continue
            votes[c][r['axis']].update(idx.tolist())
        stats['selected_'+interpret(r)]+=1
    if scene is None:
        return prior,dict(stats)
    scene.size=image.size
    kept=materialize(scene,votes)
    stats['nominated_cells']=len(votes)
    for c in kept: stats['kept_'+('wall' if 1 not in votes[c] else 'top' if scene.eye[1]>c[1]+1 else 'underside')]+=1
    return prior|kept,dict(stats)


def conservative(image,parts,counts,camera,prior):
    return select(image,parts,counts,camera,prior)


def exploratory(image,parts,counts,camera,prior):
    return select(image,parts,counts,camera,prior,False)


def horizontal(image,parts,counts,camera,prior):
    return select(image,parts,counts,camera,prior,horizontal=True)


def corroborated(image,parts,counts,camera,prior):
    return select(image,parts,counts,camera,prior,strict=False,horizontal=True,donor=True)

