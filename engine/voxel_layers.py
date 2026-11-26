"""Estimate stacked axis-aligned footprints using ground contacts and silhouette runs."""

import numpy as np


def _footprint(scene):
    """Vote for floor edges whose projected midpoints follow the lower contour.

    Multiple well-supported edges can come from separate pillars. Their union
    bounds the horizontal support without assuming a single front contact.
    """
    hull = scene["hull"]
    nx, _, nz = hull.shape
    mask = scene["mask"]
    matrix = scene["P"]
    bottom = mask.shape[0] - mask[::-1].argmax(axis=0)
    bounds = [[0, nx], [0, nz]]
    contacts = []
    for axis, length, perpendicular, width in ((0, nx, 2, nz), (2, nz, 0, nx)):
        fixed, varying = np.indices((length+1, width))
        points = np.zeros((*fixed.shape, 3))
        points[..., axis] = fixed
        points[..., perpendicular] = varying + .5
        uvz = points @ matrix[:, :3].T + matrix[:, 3]
        uv = uvz[..., :2] / uvz[..., 2, None]
        px = np.rint(uv[..., 0]).astype(int).clip(0, mask.shape[1]-1)
        above = uvz + matrix[:, 1]
        step = np.abs(above[..., 1]/above[..., 2] - uv[..., 1])
        residual = abs(uv[..., 1] - bottom[px])
        weight = np.maximum(0, 1 - residual / np.maximum(1.5, .4*step))
        weight *= mask.any(axis=0)[px] & (uvz[..., 2] > 0)
        score = weight.sum(axis=1)
        local_max = score >= np.maximum(np.r_[0,score[:-1]], np.r_[score[1:],0])
        supported = local_max & (score >= max(1.45, .35*score.max()))
        contact = points[supported][weight[supported] > .35]
        if len(contact):
            contacts.extend(contact[:, [0, 2]])
    if len(contacts) >= 4:
        extents = np.quantile(contacts, [.015,.985], axis=0)
        for i, size in enumerate((nx, nz)):
            if extents[1,i] - extents[0,i] >= 2:
                bounds[i] = [max(0, int(np.floor(extents[0,i]))-1),
                             min(size, int(np.ceil(extents[1,i]))+1)]
    return bounds


def _layer(mask):
    """Decompose a horizontal slice into rectangles with common side planes.

    Each X row votes for depth endpoints on neighboring X rows. A sub-block
    staircase from perspective is snapped onto a common plane when supported.
    Separate runs preserve disconnected columns and openings.
    """
    starts = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)))
    ends = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)))
    result = np.zeros_like(mask)
    for x, row in enumerate(mask):
        if not row.any():
            continue
        nearby = slice(max(0, x-3), x+4)
        left_score = starts[nearby].sum(axis=0)
        right_score = ends[nearby].sum(axis=0)
        for a, b in zip(np.flatnonzero(starts[x]), np.flatnonzero(ends[x])):
            if b-a >= 2:
                mid = (a+b)//2
                left = np.arange(a, min(mid, a+2)+1)
                right = np.arange(max(mid, b-2), b+1)
                a = left[np.argmax(left_score[left] - .35*(left-a))]
                b = right[np.argmax(right_score[right] - .35*(b-right))]
            result[x, a:b+1] = True
    return result


def predict(scene):
    hull = np.asarray(scene["hull"], dtype=bool)
    result = np.zeros_like(hull)
    if not hull.any():
        return result
    (x0, x1), (z0, z1) = _footprint(scene)
    # Permit upper setbacks and eaves to move beyond the contact footprint.
    # Two extra cells allow cantilevers while keeping depth bounded by ground evidence.
    for y in range(hull.shape[1]):
        margin = 0 if y == 0 else 2
        xs = slice(max(0, x0-margin), x1+margin)
        zs = slice(max(0, z0-margin), z1+margin)
        result[xs, y, zs] = _layer(hull[xs, y, zs])
    return result
