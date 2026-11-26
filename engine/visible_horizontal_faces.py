"""Recover visible horizontal faces using their orientations at native pixel centers."""
from collections import Counter

import numpy as np

from engine.surface_model import intrinsic
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE


def _surface_masks(parts, size, xs, ys, rotation):
    shape = len(ys), len(xs)
    union = np.zeros(shape, bool)
    axes = np.zeros((3, *shape), bool)
    for part in parts:
        support = part.get('support')
        if not support:
            continue
        width, height = support['size']
        raw = np.zeros((height, width), bool)
        for y, left, right in support['runs']:
            raw[y, left:right] = True
        mask = raw[np.ix_((ys * height / size[1]).astype(int), (xs * width / size[0]).astype(int))]
        union |= mask
        if part.get('normal_camera') is not None:
            axis = np.argmax(abs(rotation.T @ part['normal_camera']))
            axes[axis] |= mask
    return union, axes


def _keep_prior_visible(size, projection, eye, prior, added):
    """Reject additions that remove the final visible native pixel of any prior block."""
    native = projection.copy()
    native[:2] *= SAMPLE_STRIDE
    before = sorted(prior)
    old_owner = _frontmost_cells(before, native, eye, size[::-1])
    original_ids = np.unique(old_owner[old_owner >= 0])
    protected = {before[i] for i in original_ids}
    while added:
        cells = sorted(prior | added)
        owner = _frontmost_cells(cells, native, eye, size[::-1])
        seen_ids = np.unique(owner[owner >= 0])
        lost = protected.difference(cells[i] for i in seen_ids)
        if not lost:
            break
        old_pixels = np.isin(old_owner, [i for i in original_ids if before[i] in lost])
        front = np.unique(owner[old_pixels])
        blockers = {cells[i] for i in front if i >= 0} & added
        if not blockers:
            # Preserve the input build if rounding at a shared cube edge is ambiguous.
            return prior
        added = added - blockers
    return prior | added


def recover(image, parts, camera, prior):
    """Return the supported build, including additions confirmed by the RGB-derived masks."""
    if not prior or camera['P'] is None:
        return prior
    projection = np.asarray(camera['P'], float)
    projection = projection / np.linalg.norm(projection[2, :3])
    eye = np.linalg.solve(projection[:, :3], -projection[:, 3])
    focal = np.linalg.norm(projection[0, :3] - .5 * image.width * projection[2, :3])
    rotation = np.linalg.solve(intrinsic(image.size, focal), projection[:, :3])
    # Use original pixel centers (0.5, 2.5, ...) so every sampled ray sees actual evidence.
    xs = np.arange(.5, image.width, 2)
    ys = np.arange(.5, image.height, 2)
    shape = len(ys), len(xs)
    scaled = projection.copy()
    scaled[:2] = (projection[:2] + .5 * projection[2]) * (SAMPLE_STRIDE / 2)
    union, axis_masks = _surface_masks(parts, image.size, xs, ys, rotation)
    horizontal = axis_masks[1]
    if not horizontal.any():
        return prior
    points = sorted(prior)
    owner, base_axes = _frontmost_cells(points, scaled, eye, shape, face_axes=True)
    base_depth = np.zeros(shape)
    v, u = np.nonzero(owner >= 0)
    normals = base_axes[v, u]
    rays = np.column_stack((xs[u], ys[v], np.ones(len(u)))) @ np.linalg.inv(projection[:, :3]).T
    coordinate = np.asarray(points)[owner[v, u], normals]
    plane = coordinate + (eye[normals] > coordinate + 1)
    base_depth[v, u] = rays[np.arange(len(v)), normals] / (plane - eye[normals])

    # Neighboring cells suggest where to test. Only mask-supported, visible faces can pass.
    neighbors = Counter((x+dx, y, z+dz) for x, y, z in points for dx, dz in ((-1,0), (1,0), (0,-1), (0,1)))
    proposals = set()
    for cell, anchors in neighbors.items():
        if anchors < 2 or cell in prior or cell[1] <= eye[1] <= cell[1] + 1:
            continue
        hit = _sample_face(scaled, cell, 1, 1 if eye[1] > cell[1]+1 else -1, shape)
        if hit is None:
            continue
        box, pixels, inverse_depth, *_ = hit
        visible = pixels & (inverse_depth > base_depth[box] + 1e-8)
        matches = np.count_nonzero(visible & horizontal[box] & (base_axes[box] != 1))
        if matches >= 4 and matches >= .85 * visible.sum():
            proposals.add(cell)
    if not proposals:
        return prior
    cells = sorted(prior | proposals)
    owner, axes = _frontmost_cells(cells, scaled, eye, shape, face_axes=True)
    visible = owner >= 0
    extent = np.bincount(owner[visible], minlength=len(cells))
    supported = np.bincount(owner[visible & union], minlength=len(cells))
    matches = np.take_along_axis(axis_masks, np.maximum(0, axes)[None], axis=0)[0]
    confirmed = np.bincount(owner[visible & matches], minlength=len(cells))
    changed = horizontal & (axes == 1) & (base_axes != 1)
    horizontal_hits = np.bincount(owner[visible & changed], minlength=len(cells))
    passes = (horizontal_hits >= 4) & (supported >= .90 * extent) & (confirmed >= .85 * extent)
    added = {cells[i] for i in np.flatnonzero(passes)} & proposals
    if not added:
        return prior
    # An original native-pixel witness remains valid after dropping rejected proposals.
    if prior <= {cells[i] for i in np.flatnonzero(extent)}:
        return prior | added
    return _keep_prior_visible(image.size, projection, eye, prior, added)
