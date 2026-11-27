"""Recover only back-wall faces with photographed pixel support in a measured recess."""
import numpy as np

from engine.pillar_seams import native_mask, orientation
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def _back_cells(mask, inverse, eye, axis, level, rim):
    """Require four interior native pixels after a small exclusion at the detected cell edge."""
    y,x = np.nonzero(mask)
    rays = np.stack([x+.5,y+.5,np.ones_like(x)],axis=1) @ inverse.T
    depths = np.divide(level-eye[axis],rays[:,axis],out=np.zeros(len(x)),where=abs(rays[:,axis])>1.e-7)
    points=eye+rays*depths[:,None]
    tangent=2-axis
    xx,yy=points[:,tangent],points[:,1]
    good=(depths>1.5)&(depths<150)&(xx>=rim[0])&(xx<rim[1])&(yy>=rim[2])&(yy<rim[3])
    frac=points[:,[tangent,1]]%1
    good&=((frac>=.015)&(frac<=.985)).all(1)
    cells=np.floor(points[good]).astype(int)
    cells[:,axis]=level-int(eye[axis]>level)
    cells,hits=np.unique(cells,axis=0,return_counts=True)
    return set(map(tuple,cells[(hits>=4)&(cells[:,1]>=0)].tolist()))


def _clear_front(cells, owner, covered, axis, front, depth, rim, eye):
    tang=2-axis
    face=front-int(eye[axis]>front)
    sign=int(np.sign(depth))
    total=np.bincount(owner[owner>=0],minlength=len(cells))
    inside=np.bincount(owner[covered&(owner>=0)],minlength=len(cells))
    return {c for j,c in enumerate(cells)
            if 0<=(c[axis]-face)*sign<abs(depth) and rim[0]<=c[tang]<rim[1] and rim[2]<=c[1]<rim[3]
            and total[j]>=4 and 100*inside[j]>=90*total[j]}


def inset_edits(size, camera, parts, events, original):
    """Return signed cell edits from estimated planes; local coordinates match the ordinary export.

    Each event has the host's front grid level, signed setback, two tangent-axis rim bounds,
    and observed mask IDs for the side, sill and back. The ordinary cell set supplies only
    localized removal candidates and the first-hit geometry for each of their visible rays.
    """
    original=set(original)
    added,removed=set(),set()
    if not events or camera.get('P') is None:
        return added,removed
    p,rot=orientation(camera,size)
    inv=np.linalg.inv(p[:,:3])
    eye=-inv@p[:,3]
    by_id={part['id']:part for part in parts}
    cells=sorted(original)
    shape=tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in size[::-1])
    owner=_frontmost_cells(cells,p,eye,shape)
    for event in events:
        if any(event.get(key) not in by_id for key in ('output','side','sill')):
            continue
        back=by_id[event['output']]
        if back.get('normal_camera') is None:
            continue
        axis=int(np.argmax(abs(rot.T@back['normal_camera'])))
        level,depth,rim=event['level'],event['depth'],event['rim']
        if axis not in (0,2) or depth*(level-eye[axis])<=0:
            continue
        mask=native_mask(back,size)
        observed=_back_cells(mask,inv,eye,axis,level+depth,rim)
        if not observed:
            continue
        added.update(observed)
        supported=mask|native_mask(by_id[event['side']],size)|native_mask(by_id[event['sill']],size)
        iy=np.minimum(size[1]-1,np.arange(shape[0])*SAMPLE_STRIDE+SAMPLE_STRIDE//2)
        ix=np.minimum(size[0]-1,np.arange(shape[1])*SAMPLE_STRIDE+SAMPLE_STRIDE//2)
        covered=supported[np.ix_(iy,ix)]
        removed.update(_clear_front(cells,owner,covered,axis,level,depth,rim,eye))
    return added-original,(removed & original)-added

