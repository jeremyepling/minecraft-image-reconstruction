"""Replace a competing plane only where the photographed rim and all native rays agree."""
import numpy as np

from engine import roof_height_rim as rim
from engine.visible_horizontal_faces import _surface_masks
from engine.visible_placement import _frontmost_cells


def substitute(prior, part, decision, P, eye, rotate, image, parts):
    xs,ys=np.arange(image.width)+.5,np.arange(image.height)+.5
    mask=_surface_masks([part],image.size,xs,ys,rotate)[0]
    union,axis_masks=_surface_masks(parts,image.size,xs,ys,rotate)
    y=decision['block_y']
    native=P.copy()
    native[:2]*=4
    pts=sorted(prior)
    old,old_axis=_frontmost_cells(pts,native,eye,mask.shape,face_axes=True)
    seen=old>=0
    sizes=np.bincount(old[seen],minlength=len(pts))
    overlap=np.bincount(old[seen & mask],minlength=len(pts))
    horizontal=np.bincount(old[seen & (old_axis==1)],minlength=len(pts))
    # Protect photographed supporting columns. Surfaces known to explain any independent
    # patch outside the roof can never vanish to make the new plane visible.
    bad={i for i,p in enumerate(pts) if p[1]!=y and sizes[i]>=4 and overlap[i]==sizes[i]
         and horizontal[i] < .15*sizes[i] and (p[0],p[1]-1,p[2]) not in prior}
    if not bad:
        return prior
    bad_pixels=np.isin(old,list(bad))
    v,u=np.nonzero(bad_pixels)
    directions=rim.rays(u+.5,v+.5,np.linalg.inv(P[:,:3]))
    t=(decision['plane_y']-eye[1])/directions[:,1]
    valid=(t>1)&(t<130)
    xz=np.floor(eye[[0,2]]+t[valid,None]*directions[valid][:,[0,2]]).astype(int)
    choices,counts=np.unique(xz,axis=0,return_counts=True)
    added={(a,y,b) for a,b in choices[counts>=8].tolist()}-prior
    remove={pts[i] for i in bad}
    # If a hypothesis fails, restore original blocks and rerender until every survivor
    # has a witness and every removed cell's original rays reach this actual roof patch.
    while added and remove:
        scene=sorted((prior-remove)|added)
        front,axes=_frontmost_cells(scene,native,eye,mask.shape,face_axes=True)
        visible=front>=0
        sizes_now=np.bincount(front[visible],minlength=len(scene))
        covered=np.bincount(front[visible&union],minlength=len(scene))
        matching=np.take_along_axis(axis_masks,np.maximum(axes,0)[None],axis=0)[0]
        oriented=np.bincount(front[visible&matching],minlength=len(scene))
        hits=np.bincount(front[visible&(axes==1)&mask],minlength=len(scene))
        ids={p:i for i,p in enumerate(scene)}
        good={c for c in added if covered[ids[c]]==sizes_now[ids[c]]
              and oriented[ids[c]]>=.97*sizes_now[ids[c]] and hits[ids[c]]>=max(12,.65*sizes_now[ids[c]])}
        winners=np.isin(front,[ids[c] for c in good])&(axes==1)&mask
        explained=np.bincount(old[seen&winners],minlength=len(pts))
        remove_now={pts[i] for i in bad if pts[i] in remove and explained[i]==sizes[i]}
        # Every other visible old cell must retain a pixel. Reject offending incoming cubes.
        lost=[i for i,p in enumerate(pts) if sizes[i] and p not in remove and not sizes_now[ids[p]]]
        if lost:
            blockers=np.unique(front[np.isin(old,lost)])
            good-={scene[i] for i in blockers if i>=0}
        if good==added and remove_now==remove:
            return set(scene)
        added,remove=good,remove_now
    return prior


def recover(image,parts,counts,camera,prior):
    if not prior or not parts or camera.get('P') is None:
        return prior,dict(height_decisions=[],primary_height=None)
    decisions,accepted,P,eye,rotation=rim.select(image,parts,camera,prior)
    for _,d in accepted:
        d['accepted']=bool(d['block_y']>=1 and d['anchors']>=2 and d['columns']>=2)
    accepted=[(p,d) for p,d in accepted if d['accepted']]
    kept=rim.materialize(image,parts,accepted,P,eye,rotation,prior) if accepted else prior
    replacement_added=replacement_removed=0
    for part,d in accepted:
        result=substitute(kept,part,d,P,eye,rotation,image,parts)
        replacement_added+=len(result-kept)
        replacement_removed+=len(kept-result)
        kept=result
    primary=max((d for _,d in accepted),key=lambda d:d['pixels']*d['rim'],default=None)
    return kept,dict(height_decisions=decisions,primary_height=primary,
                     replacement_added=replacement_added,replacement_removed=replacement_removed)
