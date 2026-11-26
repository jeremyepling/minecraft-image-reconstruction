"""Reject repeated duplicate side-face placements with an exposed-floor witness."""
from collections import defaultdict

import numpy as np


def _clear_rays(eye, points, blockers):
    """Test segments ending on the floor against the other predicted unit cubes."""
    if not len(blockers):
        return np.ones(len(points), dtype=bool)
    low = np.minimum(eye, points.min(axis=0))
    high = np.maximum(eye, points.max(axis=0))
    blockers = blockers[((blockers + 1 >= low) & (blockers <= high)).all(axis=1)]
    delta = points - eye
    # In a stationary coordinate the ray's interval is either unbounded or empty.
    moving = abs(delta) > 1.e-10
    denominator = np.where(moving, delta, 1.)
    enter = np.zeros((len(blockers), len(points)))
    leave = np.full(enter.shape, 1.-1.e-7)
    for axis in range(3):
        a = (blockers[:, None, axis] - eye[axis]) / denominator[:, axis]
        b = a + 1. / denominator[:, axis]
        inside = (blockers[:, None, axis] <= eye[axis]) & (eye[axis] < blockers[:, None, axis]+1)
        enter = np.maximum(enter, np.where(moving[:,axis], np.minimum(a,b), np.where(inside,-np.inf,np.inf)))
        leave = np.minimum(leave, np.where(moving[:,axis], np.maximum(a,b), np.inf))
    return ~(enter < leave).any(axis=0)


def _image_samples(rgb, projection, points):
    q = points @ projection[:, :3].T + projection[:, 3]
    uv = q[:, :2] / np.maximum(q[:,2,None], 1.e-8)
    xx,yy = np.floor(uv).astype(int).T
    height,width = rgb.shape[:2]
    valid = (q[:,2]>0) & (xx>=0) & (xx<width) & (yy>=0) & (yy<height)
    return rgb[yy.clip(0,height-1),xx.clip(0,width-1)],valid


def _floor_match(rgb, projection, eye, cells, column, front, sign):
    phase = (np.arange(10)+.5)/10
    x,z = np.meshgrid(phase,phase)
    points = np.column_stack((column[0]+x.ravel(),np.zeros(x.size),column[1]+z.ravel()))
    reference = points.copy()
    reference[:,front] += sign
    proposed = (cells[:,0]==column[0]) & (cells[:,2]==column[1])
    clear = _clear_rays(eye,points,cells[~proposed]) & _clear_rays(eye,reference,cells)
    color, in_frame = _image_samples(rgb,projection,points)
    floor, ref_in_frame = _image_samples(rgb,projection,reference)
    measured = clear & in_frame & ref_in_frame
    if measured.sum() < 12:
        return False
    observed = color[measured]
    expected = floor[measured]
    if np.mean(abs(observed-expected).max(axis=1)<=14) < .60:
        return False
    # Similar average colors cannot distinguish a smooth gray wall from a floor.
    # Require spatial variation that recurs at corresponding sub-tile locations.
    textures = np.column_stack((observed.mean(axis=1),expected.mean(axis=1)))
    low,high = np.quantile(textures,[.05,.95],axis=0)
    if (high-low).min()<12:
        return False
    pattern = np.clip(textures,low,high)
    return np.corrcoef(pattern.T)[0,1] >= .20


def _side_explains_front(cell, adjacent, front, side, projection, eye, shape, support, sample_face):
    def samples(pos, axis):
        return sample_face(projection,pos,axis,1 if eye[axis]>pos[axis]+1 else -1,shape)
    front_face = samples(cell,front)
    side_face = samples(adjacent,side)
    if front_face is None or side_face is None:
        return False
    fbox, fp = front_face[:2]
    sbox, sp = side_face[:2]
    y0,y1 = max(fbox[0].start,sbox[0].start),min(fbox[0].stop,sbox[0].stop)
    x0,x1 = max(fbox[1].start,sbox[1].start),min(fbox[1].stop,sbox[1].stop)
    if y0>=y1 or x0>=x1:
        return False
    shared = (fp[y0-fbox[0].start:y1-fbox[0].start,x0-fbox[1].start:x1-fbox[1].start] &
              sp[y0-sbox[0].start:y1-sbox[0].start,x0-sbox[1].start:x1-sbox[1].start] &
              support[y0:y1,x0:x1])
    return shared.sum()>=max(4,.40*fp.sum())


def grounded_side_owners(image, cells, strict, masks, projection, eye, sample_face, render, seeds):
    if not seeds:
        return {}
    seed_columns = {(p[0],p[2],q[0],q[2]) for p,qs in seeds.items() if len(qs)==1 for q in qs}
    rgb = np.asarray(image,dtype=np.int16)
    support = np.logical_or.reduce(masks)
    columns = defaultdict(list)
    for cell in sorted(cells):
        columns[cell[0],cell[2]].append(cell)
    coordinates = np.array(sorted(cells))
    owners = defaultdict(set)
    silhouette = None
    for column,stack in columns.items():
        # Floor evidence cannot decide ownership of an elevated beam or overhang.
        removable = []
        for level,p in enumerate(stack):
            if p[1]!=level or p in strict:
                break
            removable.append(p)
        if len(removable)<3:
            continue
        for side,front in ((0,2),(2,0)):
            # The candidate is on the camera-facing side of a retained column.
            if cell_between(eye,stack[0],side) or cell_between(eye,stack[0],front):
                continue
            toward = 1 if eye[side]>stack[0][side]+1 else -1
            front_sign = 1 if eye[front]>stack[0][front]+1 else -1
            adjacent_floor = list(stack[0]); adjacent_floor[side] -= toward
            donor_column = (adjacent_floor[0],adjacent_floor[2])
            if column + donor_column not in seed_columns:
                continue
            linked = []
            for p in removable:
                adjacent = list(p); adjacent[side] -= toward
                if tuple(adjacent) in cells and _side_explains_front(
                        p,adjacent,front,side,projection,eye,support.shape,support,sample_face):
                    linked.append((p, tuple(adjacent)))
            if len(linked)<3 or linked[0][0][1]!=0:
                continue
            if not any(seeds.get(p)=={q} for p,q in linked):
                continue
            if not _floor_match(rgb,projection,eye,coordinates,column,front,front_sign):
                continue
            if silhouette is None:
                silhouette = render(sorted(cells),projection,eye,support.shape)>=0
            proposed = {cell for cell,_ in linked}
            remaining = render(sorted(cells-proposed),projection,eye,support.shape)>=0
            if np.count_nonzero(silhouette & support & ~remaining)>len(proposed):
                continue
            for cell, adjacent in linked:
                owners[cell].add(adjacent)
    return owners


def cell_between(eye,cell,axis):
    return cell[axis]<=eye[axis]<=cell[axis]+1
