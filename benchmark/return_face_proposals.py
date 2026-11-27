"""Propose an incident wall level from another, independently anchored visible face."""
import numpy as np
from PIL import Image, ImageFilter
from engine.atlas_plane_depth import _patch_scores, _sample
from engine.camera_atlas_floor import _atlas
from engine.surface_model import intrinsic
from engine.visible_raster import observed_support
from benchmark.harbor_partition_probe import bounded_rgb_regions
from engine.plane_consensus import row_runs


def edges(region, other):
    """Midpoints of four-neighbor transitions, without jumping gaps or overlapping regions."""
    other = other & ~region
    coords = []
    for dy, dx in ((1,0), (0,1), (-1,0), (0,-1)):
        y,x = np.nonzero(region & np.roll(other, (-dy,-dx), axis=(0,1)))
        keep = (y+dy>=0)&(x+dx>=0)&(y+dy<region.shape[0])&(x+dx<region.shape[1])
        coords.extend(np.column_stack(((x[keep]+.5+dx/2)*4,(y[keep]+.5+dy/2)*4)).tolist())
    return np.array(coords)


def anchors(mask, axis, rgb, projection, eye, rays, union, atlas):
    """Vertical floor contact or a unique multi-tile atlas peak on a perpendicular plane."""
    levels = []
    if axis != 1:
        yy=np.arange(mask.shape[0])[:,None]
        foot=mask & (yy==np.where(union,yy,-1).max(0)) & (yy<mask.shape[0]-1) & (rays[:,:,1]<-.001)
        if foot.sum()>=max(5,.15*mask.any(0).sum()):
            ray=rays[foot]
            coord=eye[axis]-eye[1]*ray[:,axis]/ray[:,1]
            level=int(np.rint(np.median(coord)))
            if np.quantile(abs(coord-level),.8)<=.32:
                return [dict(level=level,method='ground',strength=1.)]
        # Enumerate depths that the observed floor silhouettes would allow for a vertical plane.
        foot=union & (yy==np.where(union,yy,-1).max(0)) & (rays[:,:,1]<-.001)
        if not foot.any():
            return []
        ray=rays[foot]
        coord=eye[axis]-eye[1]*ray[:,axis]/ray[:,1]
        low,high=np.quantile(coord,[.05,.95])
        levels=range(max(-150,int(np.floor(low))-3),min(150,int(np.ceil(high))+4))
    else:
        levels=range(1,25)
    free=[j for j in range(3) if j!=axis]
    sampled=_sample(rays[mask],180)
    good=abs(sampled[:,axis])>1e-6
    sampled=sampled[good]
    rows=[]
    for level in levels:
        depths=(level-eye[axis])/sampled[:,axis]
        valid=(depths>1.5)&(depths<150)
        if not valid.any():
            continue
        origins=eye[free]+sampled[valid][:,free]*depths[valid,None]
        scores=_patch_scores(mask,origins,axis,free,level,projection,rgb,atlas)
        if len(scores)>=2:
            rows.append((np.mean(scores[:2]),scores[1],level))
    rows.sort(reverse=True)
    if rows and rows[0][1]>=.55 and rows[0][0]>=.68 and (len(rows)==1 or rows[0][0]-rows[1][0]>=.08):
        return [dict(level=rows[0][2],method='atlas',strength=float(rows[0][0]))]
    return []


def propose(image, parts, P, planarity=None):
    h,w=image.height//4,image.width//4
    shape=h,w
    rgb=np.asarray(image.convert('RGB'))
    inv=np.linalg.inv(P[:,:3])
    eye=-inv@P[:,3]
    focal=np.linalg.norm(P[0,:3]-image.width/2*P[2,:3])
    rotation=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    def axis_of(normal):
        return int(np.argmax(abs(rotation.T@normal)))
    items=[(p['id'],axis_of(p['normal_camera']),observed_support(p['support'],shape,image.size))
           for p in parts if p.get('normal_camera') is not None]
    union=np.logical_or.reduce([m for _,_,m in items])
    tag,axis,mask=max((t for t in items if t[1]!=1), key=lambda t:t[2].sum())
    regions=bounded_rgb_regions(image,mask)
    yy,xx=np.indices(shape)
    rays=np.stack(((xx+.5)*4,(yy+.5)*4,np.ones(shape)),axis=-1)@inv.T
    near=np.array(Image.fromarray(mask.astype('uint8')).filter(ImageFilter.MaxFilter(3)),bool)
    if planarity is not None:
        labels,normals=planarity
        rim=set(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
        for k in np.unique(labels[near]):
            if k and k not in rim and np.linalg.norm(normals[k])>.9:
                items.append((f'relaxed-{k}',axis_of(normals[k]),labels==k))
    atlas=_atlas('26.3',None)
    if atlas is None:
        raise ValueError('Missing Java 26.3 atlas')
    offers=[]
    scanned=[]
    for key,other,face in items:
        if axis==other or not (near & face).any() or face.sum()<12:
            continue
        # An independently segmented return may overlap the parent carrier.
        # Deduct each RGB component separately so an embedded return can still supply a seam.
        witnesses=anchors(face,other,rgb,P,eye,rays,union,atlas)
        scanned.append(dict(id=key,axis=other,anchors=witnesses))
        if not witnesses:
            continue
        halo=np.array(Image.fromarray(face.astype('uint8')).filter(ImageFilter.MaxFilter(3)),bool)
        for group in np.unique(regions[halo]):
            if not group:
                continue
            child=regions==group
            if (face&child).sum()>.25*child.sum():
                continue
            xy=edges(child,face)
            if len(xy)<3:
                continue
            ray=np.column_stack((xy,np.ones(len(xy))))@inv.T
            for anchor in witnesses:
                dep=np.divide(anchor['level']-eye[other],ray[:,other],out=np.zeros(len(ray)),where=abs(ray[:,other])>1e-8)
                coord=eye[axis]+dep*ray[:,axis]
                finite=(dep>1.5)&(dep<150)
                for level in np.unique(np.rint(coord[finite]).astype(int)):
                    support=finite & (abs(coord-level)<=.35)
                    points=xy[support]
                    if not len(points):
                        continue
                    extent=float(np.linalg.norm(np.ptp(points,axis=0)))
                    fraction=float(support.sum()/len(xy))
                    # Each endpoint must independently locate the same integer plane.
                    order=np.argsort(xy[:,np.argmax(np.ptp(xy,axis=0))])
                    halves=np.array_split(order,2)
                    consistent=all(len(a)>=3 and np.median(abs(coord[a]-level))<=.22 for a in halves)
                    passed=bool(consistent and len(points)>=7 and extent>=20 and fraction>=.72)
                    offers.append(dict(group=int(group),target_level=int(level),anchor=key,anchor_axis=other,
                        anchor_level=anchor['level'],method=anchor['method'],strength=anchor['strength'],
                        sampled=len(xy),support=len(points),extent_px=extent,fraction=fraction,accepted=passed,
                        endpoints=points[[0,-1]].tolist()))
    return dict(parent=tag,axis=axis,scanned=scanned,options=offers,components=int(regions.max())),regions
