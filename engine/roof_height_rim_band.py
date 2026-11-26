"""Solve a visible, one-course roof fascia by fitting both long edges and the lateral ends."""
import numpy as np

from engine import roof_height_rim as rim
from engine.visible_horizontal_faces import _keep_prior_visible


def fit(mask,axis,rgb,P,eye,inverse,tips,prior):
    step=2
    edge=rim.rim_samples(mask,rgb,step,inverse)
    if edge is None:
        return None
    v,u=np.nonzero(mask)
    sample=np.linspace(0,len(v)-1,min(len(v),1400),dtype=int)
    rays=rim.rays(u[sample]*step+.5,v[sample]*step+.5,inverse)
    across=2-axis
    starts=[c[axis] for c in prior]
    best=[]
    for fixed in range(min(starts)-2,max(starts)+4):
        t=(fixed-eye[axis])/rays[:,axis]
        if ((t>1)&(t<130)).mean()<.99:
            continue
        yy=eye[1]+rays[:,1]*t
        if not .67<=np.ptp(yy)<=1.48:
            continue
        level=int(np.rint(np.median(yy)-.5))
        if level<1:
            continue
        xx=eye[across]+rays[:,across]*t
        lim=np.quantile(xx,[.003,.997])
        l,r=np.rint(lim).astype(int)
        if not 3<=r-l<=35 or max(abs(lim-[l,r]))>.32:
            continue
        # Both measured horizontal boundaries must intersect almost the same
        # integer-Y lines over the long central part of this same photographed face.
        et=(fixed-eye[axis])/edge[:,axis]
        ex=eye[across]+edge[:,across]*et
        ey=eye[1]+edge[:,1]*et
        mid=(ex>=l+.35)&(ex<=r-.35)&(et>1)&(et<130)
        if mid.sum()<15:
            continue
        edge_y=ey[mid]
        lower=abs(edge_y-level)
        upper=abs(edge_y-(level+1))
        if min((lower<.13).sum(),(upper<.13).sum()) < max(5,.18*len(edge_y)):
            continue
        # Check completeness of the strip, avoiding a thin partial crop of a taller wall.
        bins=np.floor((xx-l)/(r-l)*8).astype(int).clip(0,7)
        rows=np.floor((yy-level)*3).astype(int).clip(0,2)
        if np.count_nonzero(np.bincount(rows*8+bins,minlength=24))<22:
            continue
        signal=float(np.mean(np.exp(-.5*(np.minimum(lower,upper)/.09)**2)))
        wall=fixed-int(eye[axis]>fixed)
        footprint={(wall,i) if axis==0 else (i,wall) for i in range(l,r)}
        near={(a+dx,b+dz) for a,b in footprint for dx,dz in ((0,0),(-1,0),(1,0),(0,-1),(0,1))}
        anchors=len(near&tips.get(level,set()))
        columns=sum((a,level-1,b) in prior and (a,level-2,b) in prior for a,b in near)
        penalty=float(np.mean(abs(lim-[l,r])))
        score=signal-.45*penalty+.12*min(1,anchors/3)+.09*min(1,columns/3)
        best.append((score,signal,anchors,columns,fixed,level,wall,l,r))
    if not best:
        return None
    best.sort(reverse=True)
    score,signal,anchors,columns,fixed,y,wall,l,r=best[0]
    margin=score-best[1][0] if len(best)>1 else 1.
    ok=signal>=.55 and margin>=.08 and (anchors>=2 or columns>=2)
    cells={(wall,y,i) if axis==0 else (i,y,wall) for i in range(l,r)}
    return dict(plane_y=y+int(eye[1]>y+1),block_y=y,accepted=bool(ok),rim=signal,
                anchors=anchors,columns=columns,margin=float(margin),kind='one_course_rim',
                pixels=int(mask.sum()*4),vertical_axis=axis,vertical_plane=fixed), cells


def adjoining(mask,other):
    """Keep other surfaces from a disconnected or distant roof out of this vote."""
    a=np.pad(mask,2)
    h,w=mask.shape
    edge=np.logical_or.reduce([a[2+dy:2+dy+h,2+dx:2+dx+w]
                               for dx,dy in ((-2,0),(2,0),(0,-2),(0,2))])
    return int((edge&other).sum())


def recover(image,parts,counts,camera,prior):
    empty=dict(height_decisions=[],primary_height=None)
    if not prior or not parts or camera.get('P') is None:
        return prior,empty
    P=np.asarray(camera['P'],float)
    P=P/np.linalg.norm(P[2,:3])
    inverse=np.linalg.inv(P[:,:3])
    eye=-inverse@P[:,3]
    focal=np.linalg.norm(P[0,:3]-image.width/2*P[2,:3])
    rot=np.linalg.solve(rim.intrinsic(image.size,focal),P[:,:3])
    xs,ys=np.arange(.5,image.width,2),np.arange(.5,image.height,2)
    kept=[p for p in parts if p.get('support') and p.get('normal_camera') is not None]
    if not kept:
        return prior,empty
    masks=[rim._surface_masks([p],image.size,xs,ys,rot)[0] for p in kept]
    axes=[int(np.argmax(abs(rot.T@p['normal_camera']))) for p in kept]
    tips=rim.tops(masks,axes,eye,inverse,2)
    rgb=np.asarray(image.convert('RGB'),dtype=np.int16)
    decisions=[]
    bands=[]
    for part,mask,axis in zip(kept,masks,axes):
        if axis==1 or mask.sum()<35:
            continue
        result=fit(mask,axis,rgb,P,eye,inverse,tips,prior)
        if result is None:
            continue
        d,cells=result
        d['part']=part['id']
        decisions.append(d)
        if d['accepted']:
            bands.append((mask,d,part,cells))
    horizontal=[]
    for part,mask,axis in zip(kept,masks,axes):
        if axis!=1 or mask.sum()<25:
            continue
        links=[(adjoining(bmask,mask),d) for bmask,d,_,_ in bands]
        links=[(length,d) for length,d in links if length>=6]
        if not links:
            continue
        links.sort(key=lambda pair:pair[0],reverse=True)
        length,near=links[0]
        if any(d['block_y']!=near['block_y'] and k>=.5*length for k,d in links[1:]):
            continue
        if near['block_y']<=eye[1]<=near['block_y']+1:
            continue
        # The same height must put their near joint at the very same 3D edge.
        axis=near['vertical_axis']
        v,u=np.nonzero(mask)
        rays=rim.rays(u*2+.5,v*2+.5,inverse)
        t=(near['plane_y']-eye[1])/rays[:,1]
        x=eye[axis]+t*rays[:,axis]
        if ((t>1)&(t<130)).mean()<.98 or not np.any(abs(x-near['vertical_plane'])<.19):
            continue
        d=dict(near,kind='horizontal_from_rim',part=part['id'],rim_part=near['part'],pixels=int(mask.sum()*4))
        horizontal.append((part,d))
        decisions.append(d)
    candidates=rim.materialize(image,parts,horizontal,P,eye,rot,prior) if horizontal else prior
    if bands:
        xs,ys=np.arange(image.width)+.5,np.arange(image.height)+.5
        union,axes=rim._surface_masks(parts,image.size,xs,ys,rot)
        native=P.copy()
        native[:2]*=4
        added=set.union(*(cells for *_,cells in bands))-candidates
        if added:
            scene=sorted(candidates|added)
            owner,front=rim._frontmost_cells(scene,native,eye,union.shape,face_axes=True)
            ids={p:i for i,p in enumerate(scene)}
            visible=owner>=0
            total=np.bincount(owner[visible],minlength=len(scene))
            inside=np.bincount(owner[visible&union],minlength=len(scene))
            agree=np.take_along_axis(axes,np.maximum(0,front)[None],axis=0)[0]
            aligned=np.bincount(owner[visible&agree],minlength=len(scene))
            passed=set()
            for _,d,part,cells in bands:
                mask=rim._surface_masks([part],image.size,xs,ys,rot)[0]
                voted=np.bincount(owner[mask & (front==d['vertical_axis'])],minlength=len(scene))
                for c in cells&added:
                    i=ids[c]
                    if voted[i]>=max(12,.25*total[i]) and inside[i]>=.98*total[i] and aligned[i]>=.91*total[i]:
                        passed.add(c)
            if passed:
                candidates=_keep_prior_visible(image.size,P,eye,candidates,passed)
    active=[d for d in decisions if d['accepted']]
    primary=max(active,key=lambda d:d['pixels']*d['rim'],default=None)
    return candidates,dict(height_decisions=decisions,primary_height=primary)
