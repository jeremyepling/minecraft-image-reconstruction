"""Fit photographed rectangular roof bands to two physically separated visible supports."""
from collections import defaultdict

import numpy as np

from engine.surface_model import intrinsic
from engine.visible_horizontal_faces import _surface_masks, _keep_prior_visible
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import observed_support, SAMPLE_STRIDE


def support_courses(points, owner, axes, oriented):
    """Require photographed vertical faces on two consecutive courses of a candidate support."""
    good = (owner >= 0) & (axes != 1)
    good &= np.take_along_axis(oriented, np.maximum(0, axes)[None], axis=0)[0]
    pixels = np.bincount(owner[good], minlength=len(points))
    visible = {p for p, n in zip(points, pixels) if n >= 3}
    courses = defaultdict(list)
    for x, y, z in visible:
        if (x, y-1, z) in visible:
            courses[y+1].append((x, z))
    return {y: np.array(positions) for y, positions in courses.items() if len(positions) >= 2}


def rectified_bounds(uv):
    # Percentile edges reject occasional background or a shared antialiased mask edge.
    bounds = np.quantile(uv, [.004, .996], axis=0)
    low, high = np.rint(bounds).astype(int)
    lengths = high - low
    if lengths.min() < 1 or lengths.max() > 32 or lengths.max() < 3:
        return None
    if (np.abs(bounds - [low, high]) > .31).any():
        return None
    # Broken silhouettes and diagonally clipped surfaces should not define a plane.
    bins = np.clip(((uv-low)/lengths*5).astype(int), 0, 4)
    fill = np.bincount(bins[:, 0]*5+bins[:, 1], minlength=25)
    if np.count_nonzero(fill) < 22:
        return None
    residual = np.abs(bounds - [low, high]).mean()
    return low, high, float(residual)


def rectangle(mask, axis, eye, inverse, courses):
    v, u = np.nonzero(mask)
    if len(v) < 50:
        return None
    take = np.linspace(0, len(u)-1, min(850, len(u)), dtype=int)
    rays = np.column_stack((u[take]*4+2, v[take]*4+2, np.ones(len(take)))) @ inverse.T
    if np.min(abs(rays[:,axis])) < .001:
        return None
    free = [i for i in range(3) if i != axis]
    if axis == 1:
        levels = {y + int(eye[1]>y+1) for y in courses if not y <= eye[1] <= y+1}
    else:
        levels = {int(p[axis//2])+offset for pillars in courses.values() for p in pillars
                  for offset in range(-2, 4)}
    proposals = []
    for level in levels:
        distances = (level-eye[axis])/rays[:,axis]
        if np.min(distances) < 2 or np.max(distances) > 130:
            continue
        uv = eye[free]+rays[:,free]*distances[:,None]
        bounds = rectified_bounds(uv)
        if bounds is None:
            continue
        low, high, residual = bounds
        if axis == 1:
            y = level - int(eye[1] > level)
            if y not in courses:
                continue
            anchors = courses[y]
            inside = np.all((anchors >= low) & (anchors < high), axis=1)
            anchors = anchors[inside]
        else:
            yi = free.index(1)
            y = int(low[yi])
            if high[yi]-y != 1 or y not in courses:
                continue
            lateral = 2-axis
            i = free.index(lateral)
            plane_cell = level - int(eye[axis]>level)
            anchors = courses[y]
            inside = (anchors[:,lateral//2] >= low[i]) & (anchors[:,lateral//2] < high[i])
            anchors = anchors[inside & (np.abs(anchors[:,axis//2]-plane_cell) <= 2)]
        if len(anchors)<2 or np.ptp(anchors, axis=0).max() < 2:
            continue
        # Compete with other integer planes using the same photographed perimeter.
        cells = set()
        for i in range(low[0],high[0]):
            for j in range(low[1],high[1]):
                cell = [0,0,0]
                cell[axis] = int(level) - int(eye[axis]>level)
                cell[free[0]], cell[free[1]] = int(i), int(j)
                cells.add(tuple(cell))
        proposals.append((residual, level, cells))
    if not proposals:
        return None
    proposals.sort(key=lambda row: row[0])
    residual, level, cells = proposals[0]
    margin = proposals[1][0]-residual if len(proposals)>1 else 1.
    return cells, residual, margin


def recover(image, parts, counts, camera, prior, *, mode='guarded'):
    if not prior or camera.get('P') is None or not parts:
        return prior
    P = np.asarray(camera['P'],float)
    P = P/np.linalg.norm(P[2,:3])
    inverse = np.linalg.inv(P[:,:3])
    eye = -inverse@P[:,3]
    focal = np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rot = np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    xs,ys = np.arange(2,image.width,4),np.arange(2,image.height,4)
    union,orient = _surface_masks(parts,image.size,xs,ys,rot)
    points = sorted(prior)
    owner, axes = _frontmost_cells(points,P,eye,union.shape,face_axes=True)
    courses = support_courses(points,owner,axes,orient)
    if not courses:
        return prior
    native = P.copy()
    native[:2] *= SAMPLE_STRIDE
    proposals = []
    for part in parts:
        if part.get('normal_camera') is None or not part.get('support'):
            continue
        axis = int(np.argmax(abs(rot.T@part['normal_camera'])))
        mask = observed_support(part['support'],union.shape,image.size)
        result = rectangle(mask,axis,eye,inverse,courses)
        if result:
            cells, error, margin = result
            if error <= (.19 if mode == 'broad' else .14) and margin >= (.0 if mode == 'broad' else .05):
                new = cells-prior
                if new:
                    proposals.append((axis,part,new))
    if not proposals:
        return prior
    full_union, native_axes = _surface_masks(parts,image.size,np.arange(image.width)+.5,np.arange(image.height)+.5,rot)
    added = set.union(*(row[2] for row in proposals))
    # Re-render to a fixed point. Removing a front layer can expose an unsupported side.
    while added:
        scene = sorted(prior|added)
        ids = {c:i for i,c in enumerate(scene)}
        front, axes = _frontmost_cells(scene,native,eye,full_union.shape,face_axes=True)
        valid = front>=0
        seen = np.bincount(front[valid],minlength=len(scene))
        inside = np.bincount(front[valid & full_union],minlength=len(scene))
        matched = np.take_along_axis(native_axes,np.maximum(axes,0)[None],axis=0)[0]
        aligned = np.bincount(front[valid & matched],minlength=len(scene))
        keep = set()
        for axis,part,candidates in proposals:
            mask = _surface_masks([part],image.size,np.arange(image.width)+.5,
                                  np.arange(image.height)+.5,rot)[0]
            support = np.bincount(front[(axes==axis) & mask],minlength=len(scene))
            for cell in candidates & added:
                i = ids[cell]
                if (support[i]>=12 and support[i]>=.12*seen[i] and inside[i]>=.96*seen[i]
                        and aligned[i]>=.80*seen[i]):
                    keep.add(cell)
        keep = _keep_prior_visible(image.size,P,eye,prior,keep)-prior if keep else set()
        if keep==added:
            break
        added=keep
    return prior|added


def broad(image, parts, counts, camera, prior):
    return recover(image,parts,counts,camera,prior,mode='broad')
