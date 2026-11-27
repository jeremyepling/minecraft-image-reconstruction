"""Join touching, jagged fragments only when both predict the same repeated texels."""

from collections import Counter
from itertools import combinations

import numpy as np
from PIL import ImageFilter

from engine.diagonal_texture_join import correlation, texture_fit
from engine.guarded_wall_outlines import _on_box, _remap_junctions
from engine.pillar_seams import native_mask, orientation
from engine.plane_consensus import with_mask
from engine.surface_perimeter import _native_alpha


METHOD = 'jagged-seam-texture-join'


def _bent_contact(left, right, size):
    # Work only where the two observed supports could meet. Adjacent interior
    # samples on both sides reject one-pixel spurs, holes and overlap artifacts.
    a, b = left[1], right[1]
    box = (max(0, max(a[0], b[0])-12), max(0, max(a[1], b[1])-12),
           min(size[0], min(a[2], b[2])+12), min(size[1], min(a[3], b[3])+12))
    if box[2]-box[0] < 16 or box[3]-box[1] < 16:
        return False
    first, second = _on_box(left,box), _on_box(right,box)
    if np.count_nonzero(first & second) > .10 * min(left[0].sum(), right[0].sum()):
        return False
    a, b = first & ~second, second & ~first
    h, w = a.shape
    points = []
    for dy, dx in ((1,0), (-1,0), (0,1), (0,-1)):
        for gap in range(1,5):
            y, x = np.nonzero(a & np.roll(b, (-gap*dy, -gap*dx), axis=(0,1)))
            keep = (x>=9)&(x<w-9)&(y>=9)&(y<h-9)
            y, x = y[keep], x[keep]
            inside = ~a[y+dy,x+dx] & a[y-5*dy,x-5*dx] & b[y+(gap+5)*dy,x+(gap+5)*dx]
            if gap > 1:
                inside &= ~b[y+(gap-1)*dy,x+(gap-1)*dx]
            if np.any(inside):
                points.append(np.column_stack((x[inside],y[inside])))
    if not points:
        return False
    xy = np.unique(np.concatenate(points), axis=0)
    if len(xy)<18 or max(np.ptp(xy,axis=0))<16:
        return False
    centered = xy - xy.mean(0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    return np.quantile(abs(centered @ basis[1]),.9) >= 3.5


def _matches(first, second):
    if not first or not second or first['level'] != second['level']:
        return False
    if min(first['margin'],second['margin'])<.08 or min(first['corr'],second['corr'])<.65:
        return False
    common = first['valid'] & second['valid']
    if common.sum()<28:
        return False
    a, b = first['template'][common], second['template'][common]
    return correlation(a,b)>=.72 and abs(a.mean()-b.mean())<=8


def refine(image, parts, version, *, camera=None, atlas_path=None):
    """Return the original list on ambiguous seams and reuse any estimated camera."""
    patches = {}
    for i,p in enumerate(parts):
        normal = np.asarray(p.get('normal_camera'), dtype=float)
        if p.get('junction_split') or normal.shape!=(3,) or not np.isfinite(normal).all() or not p.get('support'):
            continue
        patch = _native_alpha(p['support'],image.size)
        if patch:
            mask, box = np.asarray(patch[0],bool), patch[1]
            if mask.sum()>=64:
                patches[i] = mask, box
    pairs = [(i,j) for i,j in combinations(patches,2)
             if np.dot(parts[i]['normal_camera'],parts[j]['normal_camera'])>=.965
             and _bent_contact(patches[i],patches[j],image.size)]
    if not pairs:
        return parts, [], camera
    if camera is None:
        from engine.visible_placement import seam_camera
        camera = seam_camera(image, version, atlas_path=atlas_path)
    projection = np.asarray(camera.get('P') if camera else None, dtype=float)
    if (projection.shape!=(3,4) or not np.isfinite(projection).all()
            or np.linalg.matrix_rank(projection[:,:3])!=3 or np.linalg.norm(projection[2,:3])<1.e-8):
        return parts, [], camera
    p, rot = orientation(camera,image.size)
    inv = np.linalg.inv(p[:,:3])
    eye = -inv @ p[:,3]
    floor_ray = inv @ [image.width/2,.72*image.height,1.]
    if abs(floor_ray[1])<1.e-6 or floor_ray[1]*eye[1]>=0:
        return parts, [], camera
    center = eye - floor_ray*(eye[1]/floor_ray[1])
    rgb = np.asarray(image)
    fits = {}
    accepted = []
    for i,j in pairs:
        axis = int(np.argmax(abs(rot.T @ parts[i]['normal_camera'])))
        if axis != int(np.argmax(abs(rot.T @ parts[j]['normal_camera']))):
            continue
        origin = round(center[axis])
        for k in (i,j):
            if (k,axis) not in fits:
                fits[k,axis] = texture_fit(rgb,*patches[k],axis,eye,inv,range(origin-24,origin+25))
        if _matches(fits[i,axis],fits[j,axis]):
            accepted.append((i,j))
    involved = Counter(k for pair in accepted for k in pair)
    exclusive = [(i,j) for i,j in accepted if involved[i]==involved[j]==1]
    if not exclusive:
        return parts, [], camera
    gray = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    events, merged, retired = [], [], {}
    for i,j in exclusive:
        largest = max((i,j), key=lambda k:patches[k][0].sum())
        source = {k:v for k,v in parts[largest].items() if k not in ('context_crop','source_crop','corner_junctions')}
        mask = native_mask(parts[i],image.size) | native_mask(parts[j],image.size)
        item = with_mask(source,mask,image.size,gray,np.ones(2))
        item['id'] += '-seam-union'
        inputs = [parts[i]['id'],parts[j]['id']]
        retired.update(dict.fromkeys(inputs,item['id']))
        merged.append(item)
        events.append(dict(inputs=inputs,output=item['id']))
    output = [p for p in parts if p['id'] not in retired] + merged
    if any(p.get('corner_junctions') for p in parts):
        output = _remap_junctions(parts,output,retired)
    return output, events, camera
