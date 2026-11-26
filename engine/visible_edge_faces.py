"""Recover visible edge faces from photographed pixels left unclaimed by the first pass."""
from itertools import product

import numpy as np
from PIL import Image

from engine.partial_atlas_face import verify
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE


CUBE_CORNERS = np.array(list(product((0, 1), repeat=3)))


def _support_masks(support, size, shape):
    """Decode the runs once. Coarse samples and native pixels have different centers."""
    width, height = support['size']
    raw = np.zeros((height, width), bool)
    for y, left, right in support['runs']:
        raw[y, left:right] = True
    if size == (width, height):
        center = SAMPLE_STRIDE // 2
        return raw[center::SAMPLE_STRIDE, center::SAMPLE_STRIDE], raw
    x = ((np.arange(shape[1]) + .5) * SAMPLE_STRIDE * width / size[0]).astype(int)
    y = ((np.arange(shape[0]) + .5) * SAMPLE_STRIDE * height / size[1]).astype(int)
    native = np.asarray(Image.fromarray(raw).resize(size, Image.Resampling.NEAREST))
    return raw[np.ix_(y, x)], native


def _neighboring(cell, axis, cells):
    for dim in range(3):
        if dim != axis:
            for offset in (-1, 1):
                point = list(cell)
                point[dim] += offset
                if tuple(point) in cells:
                    return True
    return False


def _near_unclaimed(neighbors, projection, available):
    """Skip cubes whose projected bounding box contains fewer than four available rays."""
    if not neighbors or np.count_nonzero(available) < 4:
        return []
    points = np.array(sorted(neighbors))
    corners = points[:, None, :] + CUBE_CORNERS
    screen = corners @ projection[:, :3].T + projection[:, 3]
    # A cube crossing the camera plane may have unbounded screen extent. Let the
    # exact per-face rasterizer handle those cubes without a bounding-box test.
    bounded = np.all(screen[:, :, 2] > 1.e-5, axis=1)
    keep = ~bounded
    homogeneous = screen[bounded]
    xy = homogeneous[:, :, :2] / (SAMPLE_STRIDE * homogeneous[:, :, 2:])
    height, width = available.shape
    limit = [width, height]
    low = np.floor(xy.min(axis=1).clip(0, limit)).astype(int)
    high = np.ceil(xy.max(axis=1).clip(0, limit)).astype(int)
    summed = np.pad(available, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    x0, y0 = low.T
    x1, y1 = high.T
    count = summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0]
    keep[bounded] = count >= 4
    return list(map(tuple, points[keep].tolist()))


def _verify_neighbors(cells, rgb, projection, eye, available, native, atlas, proposals):
    neighbors = set()
    for cell in cells:
        for dim in range(3):
            for offset in (-1, 1):
                point = list(cell)
                point[dim] += offset
                if point[1] >= 0:
                    neighbors.add(tuple(point))
    for cell in _near_unclaimed(neighbors - cells, projection, available):
        for axis in range(3):
            if cell[axis] <= eye[axis] <= cell[axis] + 1 or not _neighboring(cell, axis, cells):
                continue
            sign = 1 if eye[axis] > cell[axis] + 1 else -1
            hit = _sample_face(projection, cell, axis, sign, available.shape)
            if hit is None:
                continue
            box, pixels, *_ = hit
            if np.count_nonzero(pixels & available[box]) < max(4, .20 * pixels.sum()):
                continue
            if verify(rgb, atlas, available, projection, cell, axis, sign,
                      native_mask=native, threshold=.75, minimum=56):
                proposals.setdefault(cell, []).append((axis, available))


def _visible_proposals(cells, proposals, projection, eye, available):
    if not proposals:
        return set()
    all_cells = sorted(cells | proposals.keys())
    front, axes = _frontmost_cells(all_cells, projection, eye, available.shape, face_axes=True)
    # Group only the available rays owned by a proposed cell. The extra False
    # entry accounts for the background owner (-1) when indexing the lookup.
    proposed = np.array([p in proposals for p in all_cells] + [False])
    pixels = np.flatnonzero(available & proposed[front])
    if not len(pixels):
        return set()
    owners = front.ravel()[pixels]
    order = np.argsort(owners, kind='stable')
    groups = np.split(pixels[order], np.flatnonzero(np.diff(owners[order])) + 1)
    accepted = set()
    for group in groups:
        cell = all_cells[front.flat[group[0]]]
        visible_axes = axes.ravel()[group]
        if any(np.count_nonzero((visible_axes == axis) & mask.ravel()[group]) >= 2
               for axis, mask in proposals[cell]):
            accepted.add(cell)
    return accepted


def extend(image, parts, P, cells, offsets, atlas):
    if not cells or not offsets or atlas is None:
        return set()
    stride = SAMPLE_STRIDE
    shape = tuple((n + (stride-1)//2)//stride for n in image.size[::-1])
    projection = np.asarray(P, float)
    eye = np.linalg.solve(projection[:, :3], -projection[:, 3])
    owner = _frontmost_cells(sorted(cells), projection, eye, shape)
    unclaimed = owner < 0
    available = np.zeros(shape, bool)
    native_union = np.zeros(image.size[::-1], bool)
    by_id = {row['id']: row for row in offsets}
    rgb = np.asarray(image.convert('RGB'))
    inverse = np.linalg.inv(projection[:, :3])
    proposals = {}
    for part in parts:
        if not part.get('support'):
            continue
        mask, native = _support_masks(part['support'], image.size, shape)
        mask = mask & unclaimed
        available |= mask
        native_union |= native
        row = by_id.get(part['id'])
        if row is None:
            continue
        axis, level = row['axis'], row['level']
        y, x = np.nonzero(mask)
        if len(y) < 2:
            continue
        rays = np.column_stack(((x+.5)*stride, (y+.5)*stride, np.ones(len(x)))) @ inverse.T
        for plane in range(level-2, level+3):
            scale = np.divide(plane-eye[axis], rays[:, axis], out=np.zeros(len(x)), where=abs(rays[:, axis])>1.e-7)
            good = (scale > 1.5) & (scale < 150)
            xyz = np.floor(eye + scale[good,None] * rays[good]).astype(int)
            sign = 1 if eye[axis] > plane else -1
            xyz[:, axis] = plane - int(sign > 0)
            candidates, counts = np.unique(xyz, axis=0, return_counts=True)
            matches = set()
            for point in candidates[counts >= 2]:
                cell = tuple(point.tolist())
                if cell[1] >= 0 and cell not in cells and verify(
                        rgb, atlas, mask, projection, cell, axis, sign, native_mask=native,
                        threshold=.60 if plane == level else .72, minimum=42):
                    matches.add(cell)
            for cell in matches:
                if plane == level or _neighboring(cell, axis, matches) or _neighboring(cell, axis, cells):
                    proposals.setdefault(cell, []).append((axis, mask))
    _verify_neighbors(cells, rgb, projection, eye, available, native_union, atlas, proposals)
    return _visible_proposals(cells, proposals, projection, eye, available)
