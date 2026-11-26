"""Test misplaced cells against the photograph's floor contacts, surface masks and tile scale."""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np

from engine.camera_atlas_floor import _atlas
from engine.nearby_face_art import scores as texture_scores
from engine.nearby_face_decision import predict_forest
from engine.surface_model import intrinsic
from engine.visible_horizontal_faces import _surface_masks
from engine.visible_placement import _frontmost_cells


WEIGHTS = Path(__file__).with_name('grounded-depth-model.json')
FEATURES = ('outside_contacts','distance_contact','depth_outside','depth_fraction','near_depth_fraction',
            'outside_dense','depth_neighbor','local_neighbors','same_height_neighbors','column_height',
            'connected_size','connected_contacts','component_outside','column_ground','ground_distance',
            'visible_samples','axis_agreement','surface_coverage','primary_axis_fraction','atlas_fit','atlas_phase',
            'component_atlas','component_density','contact_width','anchor_axes','vertical_neighbors')


def contact_points(rgb, size, union, axes, P, eye):
    """Accept contiguous, straight silhouette runs where a wall meets floor-colored pixels."""
    width,height=size
    h,w=union.shape
    columns=np.flatnonzero(union.any(0))
    if len(columns)<8:
        return np.empty((0,3)),0
    row=h-1-np.argmax(union[::-1,columns],axis=0)
    x=4*columns+2
    # Silhouette lies between the last occupied and first free sample.
    y=4*row+4
    rays=np.column_stack((x,y,np.ones(len(x)))) @ np.linalg.inv(P[:,:3]).T
    d=np.divide(-eye[1],rays[:,1],out=np.zeros(len(x)),where=rays[:,1]<-1e-6)
    xyz=eye+d[:,None]*rays
    floor=rgb[-max(1,height//18)::5,::5].reshape(-1,3)
    tone=np.median(floor,axis=0)
    tolerance=max(18.,np.quantile(np.abs(floor-tone).mean(1),.90)*1.6)
    below=rgb[(y+8).clip(0,height-1),x.clip(0,width-1)]
    ok=(d>1)&(d<150)&(y+8<height)&(abs(below-tone).mean(1)<=tolerance)
    selected=[]
    directions=set()
    for axis in (0,2):
        levels=np.rint(xyz[:,axis])
        valid=ok & axes[axis,row,columns] & (abs(xyz[:,axis]-levels)<.32)
        idx=np.flatnonzero(valid)
        if len(idx)<4:
            continue
        groups=np.split(idx,1+np.flatnonzero((np.diff(columns[idx])>2)|(np.diff(levels[idx])!=0)))
        for run in groups:
            if len(run)>=4 and np.ptp(xyz[run,2-axis])>=.30:
                selected.extend(run)
                directions.add(axis)
    return xyz[sorted(set(selected))],len(directions)


def measure(image, parts, camera, prior, *, atlas_path=None):
    if not prior or not parts or camera.get('P') is None:
        return None
    P=np.asarray(camera['P'],float)
    P=P/np.linalg.norm(P[2,:3])
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rotation=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    xs,ys=np.arange(2,image.width,4),np.arange(2,image.height,4)
    union,axes=_surface_masks(parts,image.size,xs,ys,rotation)
    rgb=np.asarray(image.convert('RGB'),dtype='float32')
    anchors,num_axes=contact_points(rgb,image.size,union,axes,P,eye)
    if len(anchors)<4:
        return None
    atlas=_atlas('26.3',atlas_path)
    if atlas is None:
        return None
    cells=sorted(prior)
    pts=np.asarray(cells)
    n=len(pts)
    center=pts+.5
    ground=center[:,[0,2]]
    low,high=np.quantile(anchors[:,[0,2]],[.02,.98],axis=0)
    extent=np.maximum(0,np.maximum(low-ground,ground-high)).max(1)
    nearest=np.min(np.max(abs(ground[:,None,:]-anchors[:,[0,2]][None,:,:]),axis=2),axis=1)
    z=center@P[2,:3]+P[2,3]
    ground_z=anchors@P[2,:3]+P[2,3]
    zlo,zhi=np.quantile(ground_z,[.02,.98])
    dz=np.maximum(0,np.maximum(zlo-z,z-zhi))
    normal_z=(z-np.median(ground_z))/max(2,zhi-zlo)
    # Local occupancy is sparse; do not allocate a square matrix for a large build.
    index = {cell: i for i, cell in enumerate(cells)}
    columns = Counter((x,z0) for x,y,z0 in cells)
    ncol = np.asarray([columns[x,z0] for x,y,z0 in cells])
    column_ground = np.asarray([(x,0,z0) in prior for x,y,z0 in cells])
    neighbors, horizontal, vertical = np.zeros((3,n), int)
    adjacent = [[] for _ in cells]
    steps = [s for s in product((-1,0,1),repeat=3) if any(s)]
    for i,(x,y,z0) in enumerate(cells):
        for dx,dy,dz0 in steps:
            j=index.get((x+dx,y+dy,z0+dz0))
            if j is None:
                continue
            neighbors[i]+=1
            horizontal[i]+=int(dy==0)
            vertical[i]+=int(dx==0 and dz0==0)
            if abs(dx)+abs(dy)+abs(dz0)==1:
                adjacent[i].append(j)
    def distances(points, references, empty):
        if not len(references):
            return np.full(len(points),empty)
        return np.concatenate([abs(points[i:i+32,None,:]-references[None,:,:]).max(2).min(1)
                               for i in range(0,len(points),32)])
    distance_dense=distances(pts,pts[neighbors>=5],0)
    ground_dist=distances(pts[:,[0,2]],pts[pts[:,1]==0][:,[0,2]],40)
    nearest_depth=np.quantile(abs(z-np.median(z)),.8)
    owner,face=_frontmost_cells(cells,P,eye,union.shape,face_axes=True)
    mask=owner>=0
    visible=np.bincount(owner[mask],minlength=n)
    seen=np.maximum(1,visible)
    covered=np.bincount(owner[mask & union],minlength=n)/seen
    matched=[]; arts=[]
    for a in range(3):
        active=mask & (face==a) & axes[a]
        matched.append(np.bincount(owner[active],minlength=n)/seen)
        arts.append(np.concatenate([texture_scores(pts[i:i+48],a,rgb,P,eye,axes[a],atlas) for i in range(0,n,48)]))
    matched=np.stack(matched,axis=1)
    arts=np.stack(arts,axis=1)
    art=arts.max(1)
    # Face-connected groups often expose a distant copy of a photographed surface.
    group=np.full(n,-1)
    for i in range(n):
        if group[i]>=0:continue
        group[i]=i
        stack=[i]
        while stack:
            j=stack.pop()
            take=[k for k in adjacent[j] if group[k]<0]
            group[take]=i
            stack.extend(take)
    comp=np.zeros((n,5))
    for g in set(group):
        idx=group==g
        comp[idx]=[np.sum(idx),np.min(nearest[idx]),np.median(extent[idx]),np.median(art[idx,0]),np.mean(neighbors[idx])]
    span=float(max(high-low))
    X=np.column_stack([extent,nearest,dz,normal_z, z/np.median(ground_z),distance_dense,
        abs(z-np.median(z))/max(2,nearest_depth),neighbors,horizontal,ncol,
        comp[:,0],comp[:,1],comp[:,2],column_ground,ground_dist,np.log1p(visible),
        matched.sum(1),covered,matched.max(1),art[:,0],art[:,1],comp[:,3],comp[:,4],
        np.full(n,span),np.full(n,num_axes),vertical])
    return pts,X.astype('float32')


@lru_cache(maxsize=2)
def _read_weights(path,mtime):
    raw=path.read_bytes().replace(b'\r\n',b'\n')
    if sha256(raw).hexdigest() != '8f44bab7ede07fb3910eada089fa53ca1d3017fbfde0ca92cedf5d755d5d704a':
        raise ValueError('Grounded depth weights do not match the frozen model')
    obj=json.loads(raw)
    if obj['features']!=list(FEATURES):
        raise ValueError('Incompatible grounded depth model')
    return obj


def recover(image,parts,camera,prior,*,atlas_path=None,threshold=None):
    """Keep the previous build if photograph, contacts or calibrated model are insufficient."""
    try:
        obj=_read_weights(WEIGHTS,WEIGHTS.stat().st_mtime_ns)
    except (OSError,ValueError,KeyError):
        return prior,dict(removed=0,measured=False)
    obs=measure(image,parts,camera,prior,atlas_path=atlas_path)
    if obs is None:
        return prior,dict(removed=0,measured=False)
    pts,X=obs
    drop=predict_forest(obj['forest'],X)>=(obj['threshold'] if threshold is None else threshold)
    if obj['local']:
        drop &= (X[:,0]>.5)|(X[:,11]>2)
    return prior-{tuple(p) for p in pts[drop]},dict(removed=int(drop.sum()),measured=True)
