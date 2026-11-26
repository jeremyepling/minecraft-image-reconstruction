"""Image-only dense placement pass for unclaimed horizontal faces adjoining observed blocks."""
from math import ceil

import numpy as np

from engine.surface_model import intrinsic
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE
from engine.visible_ray_filter import fully_occluded_cells


def residual(image, parts, camera, baseline, stride, minimum_area, *, depth_aware=False, interior=False,
             confirm_axes=False):
    """Project a neighboring block only when its visible horizontal face has matching mask evidence."""
    if not baseline or camera['P'] is None:
        return baseline
    P = np.asarray(camera['P'], float)
    P = P/np.linalg.norm(P[2,:3])
    eye = np.linalg.solve(P[:,:3], -P[:,3])
    focal = np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    R = np.linalg.solve(intrinsic(image.size, focal), P[:,:3])
    # At stride four retain the production centers. The denser pass samples
    # native pixel indices 0,2,4,... at their actual centers 0.5,2.5,4.5,...
    first = stride/2 if stride==SAMPLE_STRIDE else .5
    xs = np.arange(first, image.width, stride)
    ys = np.arange(first, image.height, stride)
    shape = len(ys),len(xs)
    scaled = P.copy()
    scaled[:2] = (P[:2] + (stride/2-first)*P[2])*SAMPLE_STRIDE/stride
    all_masks = np.zeros(shape,bool)
    horizontal = np.zeros(shape,bool)
    axis_masks = np.zeros((3,*shape),bool) if confirm_axes else None
    for part in parts:
        if not part.get('support'):
            continue
        support = part['support']
        sw,sh = support['size']
        raw = np.zeros((sh,sw),bool)
        for y,x0,x1 in support['runs']:
            raw[y,x0:x1] = True
        mask = raw[np.ix_((ys*sh/image.height).astype(int), (xs*sw/image.width).astype(int))]
        all_masks |= mask
        if part.get('normal_camera') is not None:
            axis = np.argmax(abs(R.T @ part['normal_camera']))
            if axis==1:
                horizontal |= mask
            if confirm_axes:
                axis_masks[axis] |= mask
    if not horizontal.any():
        return baseline
    points = np.asarray(sorted(baseline))
    if depth_aware:
        owner, base_axes = _frontmost_cells(sorted(baseline),scaled,eye,shape,face_axes=True)
        base_z = np.zeros(shape)
        v,u = np.nonzero(owner>=0)
        normals = base_axes[v,u]
        rays = np.column_stack((xs[u],ys[v],np.ones(len(u)))) @ np.linalg.inv(P[:,:3]).T
        coordinate = points[owner[v,u],normals]
        plane = coordinate + (eye[normals]>coordinate+1)
        base_z[v,u] = rays[np.arange(len(v)),normals]/(plane-eye[normals])
    else:
        owner = _frontmost_cells(sorted(baseline),scaled,eye,shape)
    unclaimed = owner < 0
    evidence = unclaimed & horizontal
    minimum = ceil(minimum_area/stride**2)
    candidates = {(x+dx,y,z+dz) for x,y,z in baseline for dx,dz in ((1,0),(-1,0),(0,1),(0,-1))}
    proposals = set()
    for point in candidates-baseline:
        y = point[1]
        if interior:
            x,_,z = point
            if sum((x+dx,y,z+dz) in baseline for dx,dz in ((-1,0),(1,0),(0,-1),(0,1)))<2:
                continue
        if y <= eye[1] <= y+1:
            continue
        hit = _sample_face(scaled,point,1,1 if eye[1] > y+1 else -1,shape)
        if hit is None:
            continue
        box, pixels, inverse_depth, *_ = hit
        if depth_aware:
            visible = pixels & (inverse_depth>base_z[box]+1e-8)
            hits = np.count_nonzero(visible & horizontal[box] & (base_axes[box]!=1))
            good = hits>=minimum and hits>=.85*visible.sum()
        else:
            hits = np.count_nonzero(pixels & evidence[box])
            good = hits>=minimum and 2*hits>=pixels.sum()
        if good:
            proposals.add(point)
    if not proposals:
        return baseline
    cells = sorted(baseline | proposals)
    owners, axes = _frontmost_cells(cells,scaled,eye,shape,face_axes=True)
    newly_visible = (owners>=0) if depth_aware else unclaimed & (owners>=0)
    extent = np.bincount(owners[newly_visible], minlength=len(cells))
    supported = np.bincount(owners[newly_visible & all_masks], minlength=len(cells))
    eligible = newly_visible & horizontal & (axes==1)
    if depth_aware:
        eligible &= base_axes!=1
    face = np.bincount(owners[eligible], minlength=len(cells))
    selected = {cells[i] for i in np.flatnonzero((face>=minimum) & (supported>=.9*extent))
                if cells[i] in proposals}
    if confirm_axes:
        # A cube inserted for its top may also expose its sides. Check their
        # visible orientations against the screenshot before emitting the cube.
        matching = np.take_along_axis(axis_masks, np.maximum(0,axes)[None], axis=0)[0]
        confirmed = np.bincount(owners[newly_visible & matching], minlength=len(cells))
        selected &= {cells[i] for i in np.flatnonzero(confirmed >= .85*extent)}
    if not selected:
        return baseline
    combined = baseline | selected
    return combined - fully_occluded_cells(image.size, P, combined)
