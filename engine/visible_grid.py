"""Place fully supported cells on planes lifted from measured surface charts."""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from engine.visible_raster import SAMPLE_STRIDE
from engine.surface_model import assemble, intrinsic, pitch

SETTINGS = dict(minimum_carrier_iou=.5, maximum_plane_corner_residual_blocks=.5, minimum_face_samples=4)


def _plane_from_chart(row, extrinsics):
    if row.get('units') != 'blocks':
        return None, 'unresolved_scale'
    rotation = extrinsics[:, :3]
    corners = np.linalg.solve(rotation, (np.asarray(row['corners_camera']) - extrinsics[:, 3]).T).T
    normal = rotation.T @ np.asarray(row['normal_camera'])
    axis = int(np.argmax(abs(normal)))
    level = int(np.rint(np.median(corners[:, axis])))
    if abs(corners[:, axis]-level).max() > SETTINGS['maximum_plane_corner_residual_blocks']:
        return None, 'unresolved_depth'
    return (axis, level, int(np.sign(normal[axis]))), 'supported'


def _carriers(image, parts, masks, P):
    if P is None:
        return []
    scale = np.linalg.norm(P[2, :3])
    focal = np.linalg.norm(P[0, :3]-image.width/2*P[2, :3]) / scale
    K = intrinsic(image.size, focal)
    extrinsics = np.linalg.solve(K, P/scale)
    charts, pitches, supports = [], [], []
    for part, mask in zip(parts, masks, strict=True):
        if part.get('quad') is None or not mask.any():
            continue
        charts.append(dict(id=part['id'], quad=part['quad']))
        pitches.append(part.get('pitch') or pitch(image, part['quad'], '26.3', 'repeat'))
        supports.append(mask)
    lifted = assemble(charts, pitches, image.size, K)['planes']
    return [(mask, *_plane_from_chart(row, extrinsics)) for mask, row in zip(supports, lifted, strict=True)]


def _carrier(mask, carriers):
    best, selected, reason = 0., None, 'unresolved_chart'
    for prior, plane, status in carriers:
        intersection = np.count_nonzero(mask & prior)
        if not intersection:
            continue
        iou = intersection / np.count_nonzero(mask | prior)
        if iou >= SETTINGS['minimum_carrier_iou'] and iou > best:
            best, selected, reason = iou, plane, status
    return selected, reason


@dataclass
class CellMap:
    axis: int
    level: int
    front: int
    valid_pixels: np.ndarray
    cells: np.ndarray
    inverse: np.ndarray
    counts: np.ndarray

    def occupied(self, mask):
        covered = np.bincount(self.inverse, weights=mask.ravel()[self.valid_pixels], minlength=len(self.cells))
        complete = self.cells[(covered == self.counts) & (self.counts >= SETTINGS['minimum_face_samples'])]
        xyz = np.empty((len(complete), 3), dtype=int)
        xyz[:, self.axis] = self.level - int(self.front > 0)
        xyz[:, [i for i in range(3) if i != self.axis]] = complete
        return set(map(tuple, xyz.tolist()))


def _cell_map(P, shape, plane):
    """Assign every visible pixel center a unit square. A hole vetoes the whole block.

    Grouping all viewport pixels also handles cells clipped by the image boundary.
    Pixels past a plane's horizon cannot vote for cells behind the observer.
    """
    axis, level, front = plane
    free = [i for i in range(3) if i != axis]
    H = np.column_stack((P[:, free], P[:, 3] + level*P[:, axis]))
    y, x = np.indices(shape, dtype=float)
    rays = np.stack(((x.ravel()+.5)*SAMPLE_STRIDE, (y.ravel()+.5)*SAMPLE_STRIDE, np.ones(x.size)))
    uvw = np.linalg.solve(H, rays)
    valid = np.flatnonzero(uvw[2] > 1e-8)
    uv = np.floor((uvw[:2, valid]/uvw[2, valid]).T).astype(np.int64)
    unique, inverse, counts = np.unique(uv, axis=0, return_inverse=True, return_counts=True)
    return CellMap(axis, level, front, valid, unique, inverse.ravel(), counts)


def _blocks(masks, carriers, P, cache):
    blocks, resolved = set(), Counter()
    details = []
    for index, mask in enumerate(masks):
        plane, status = _carrier(mask, carriers) if P is not None else (None, 'unresolved_camera')
        cells = set()
        if plane is not None:
            if plane not in cache:
                try:
                    cache[plane] = _cell_map(P, mask.shape, plane)
                except np.linalg.LinAlgError:
                    cache[plane] = None
            if cache[plane] is not None:
                cells = cache[plane].occupied(mask)
            status = 'emitted' if cells else 'no_fully_supported_cell'
        blocks.update(cells)
        resolved[status] += 1
        details.append(dict(index=index, status=status, plane=plane, cells=len(cells)))
    return blocks, dict(resolved), details
