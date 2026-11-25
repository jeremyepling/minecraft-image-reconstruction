"""Refine shared mask boundaries with image-native line segments and numerical geometry."""
import numpy as np

from engine.camera_lines import _runs
from engine.mask_first import _image_depth
from engine.normal_regions import _field


SETTINGS = dict(min_length=9., probe_offset=4., pair_agreement=.65, direction_change_deg=20.,
                min_inverse_depth_step=.035, geometry_agreement=.5, max_move=3.,
                junction_radius=2.5, endpoint_inset=2., min_region_fraction=.82,
                units='mask grid samples; source lines use original image xy pixels')


def _indices(points, shape):
    xy = np.floor(points).astype(int)
    return np.clip(xy[:,1], 0, shape[0]-1), np.clip(xy[:,0], 0, shape[1]-1)


def _evidence(a, b, labels, vectors, depth, junctions):
    length = np.linalg.norm(b-a)
    if length < SETTINGS['min_length']:
        return None
    along = (b-a)/length
    cross = np.array([-along[1], along[0]])
    # Endpoints near a junction may vote right up to the corner.
    inset = [0. if len(junctions) and np.linalg.norm(junctions-end,axis=1).min()<=SETTINGS['junction_radius']
             else SETTINGS['endpoint_inset'] for end in (a,b)]
    position = np.linspace(inset[0],length-inset[1],int(length/2)+1)
    center = a + position[:,None]*along
    sides = [_indices(center + sign*SETTINGS['probe_offset']*cross, labels.shape) for sign in (-1,1)]
    left, right = sides
    pairs = np.column_stack((labels[left],labels[right]))
    different = (pairs[:,0]!=pairs[:,1]) & np.all(pairs>0,axis=1)
    if not different.any():
        return None
    types, counts = np.unique(pairs[different],axis=0,return_counts=True)
    pair = types[counts.argmax()]
    agree = np.all(pairs==pair,axis=1)
    if agree.mean()<SETTINGS['pair_agreement']:
        return None
    turns = np.sum(vectors[left]*vectors[right],axis=1)<np.cos(np.deg2rad(SETTINGS['direction_change_deg']))
    jumps = abs(depth[left]-depth[right])>=SETTINGS['min_inverse_depth_step']
    if (turns|jumps)[agree].mean()<SETTINGS['geometry_agreement']:
        return None
    return along, cross, length, inset, pair


def _topology(mask):
    # With a surrounding background ring its component count includes all enclosed holes.
    return len(_runs(mask)), len(_runs(~np.pad(mask,1)))


def refine(image, regions, normals, depth, evidence):
    """Transfer narrow strips between two existing surfaces. Unassigned openings stay open."""
    if not regions or not evidence['segments']:
        return regions
    size = tuple(regions[0]['support']['size'])
    labels = np.zeros(size[::-1],dtype=np.int32)
    for i,r in enumerate(regions,1):
        if tuple(r['support']['size'])!=size:
            raise ValueError('Mismatched mask resolutions')
        for y,x0,x1 in r['support']['runs']:
            labels[y,x0:x1] = i
    scale = np.asarray(size)/image.size
    lines = np.asarray(evidence['segments'],float).reshape(-1,2,2)*scale
    junctions = np.asarray(evidence.get('junctions',[]),float).reshape(-1,2)*scale
    if not np.all(np.isfinite(lines)) or not np.all(np.isfinite(junctions)):
        raise ValueError('Nonfinite image coordinates')
    vectors = _field(normals,size)
    relative = _image_depth(depth,size)
    proposed = labels.copy()
    best_distance = np.full(labels.shape,np.inf)
    radius = SETTINGS['max_move']
    for a,b in lines:
        supported = _evidence(a,b,labels,vectors,relative,junctions)
        if supported is None:
            continue
        along,cross,length,inset,pair = supported
        lo = np.maximum(0,np.floor(np.minimum(a,b)-radius).astype(int))
        hi = np.minimum(size,np.ceil(np.maximum(a,b)+radius).astype(int))
        if np.any(hi<=lo):
            continue
        cut = np.s_[lo[1]:hi[1],lo[0]:hi[0]]
        y,x = np.ogrid[lo[1]:hi[1],lo[0]:hi[0]]
        dx,dy = x+.5-a[0],y+.5-a[1]
        t, signed = dx*along[0]+dy*along[1], dx*cross[0]+dy*cross[1]
        dist = abs(signed)
        # No claim on gaps, no novel surface, and no modification outside the old pair.
        valid = ((t>=inset[0]) & (t<=length-inset[1]) & (dist<=radius)
                 & (dist<best_distance[cut]) & np.isin(labels[cut],pair))
        proposed[cut][valid] = np.where(signed[valid]<0,pair[0],pair[1])
        best_distance[cut][valid] = dist[valid]

    # A short line cannot erase a region, disconnect its thin bridge or plug its hole.
    # Cancel both ends of a failed transfer, then recheck its neighbors.
    while np.any(labels!=proposed):
        moved = labels!=proposed
        touched = np.unique(np.r_[labels[moved],proposed[moved]])
        rejected = []
        for i in touched:
            before,after = labels==i,proposed==i
            if min(before.sum(),after.sum()) < SETTINGS['min_region_fraction']*max(before.sum(),after.sum()):
                rejected.append(i)
                continue
            ys,xs = np.nonzero(before|after)
            box = np.s_[ys.min():ys.max()+1,xs.min():xs.max()+1]
            if _topology(before[box])!=_topology(after[box]):
                rejected.append(i)
        if not rejected:
            break
        undo = moved & (np.isin(labels,rejected)|np.isin(proposed,rejected))
        proposed[undo] = labels[undo]

    result = []
    for i,region in enumerate(regions,1):
        after = proposed==i
        changed = int(np.count_nonzero(after!=(labels==i)))
        if changed:
            runs = [list(row) for component in _runs(after) for row in component]
            region = dict(region,support=dict(size=list(size),runs=runs),support_area=int(after.sum()),
                          line_changed_pixels=changed)
        result.append(region)
    return result
