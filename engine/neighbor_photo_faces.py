"""Recover visible block faces using repeated texture phase in adjacent image regions."""
from collections import defaultdict

import numpy as np

from engine.visible_edge_faces import _support_masks
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE


def _offsets(axis):
    for dim in range(3):
        if axis != dim:
            for step in (-1, 1):
                delta = np.zeros(3, int)
                delta[dim] = step
                yield delta


def _sample_reference(rgb, owner, axes, P, xyz, index, axis):
    h, w = axes.shape
    projected = xyz @ P[:, :3].T + P[:, 3]
    xy = np.floor(projected[:, :2] / projected[:, 2:]).astype(int)
    x, y = xy.T
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (projected[:, 2] > 0)
    x, y = x.clip(0, w-1), y.clip(0, h-1)
    return rgb[y, x], inside & (owner[y, x] == index) & (axes[y, x] == axis)


def _repeat(rgb, owner, axes, P, xyz, color, anchor, delta, axis, free):
    sample, valid = _sample_reference(rgb, owner, axes, P, xyz+delta, anchor, axis)
    if valid.sum() < 12:
        return None
    # Photographic phase control: the neighboring block should match more closely
    # at the same sub-block location than at two shifted locations on the same face.
    errors = []
    normal = abs(sample-color).mean(1)
    for dim in free:
        shifted = xyz.copy()
        frac = shifted[:, dim] % 1
        shifted[:, dim] += np.where(frac < .75, .25, -.75)
        rival, inside = _sample_reference(rgb, owner, axes, P, shifted+delta, anchor, axis)
        use = valid & inside
        if use.sum() >= 12:
            errors.append(float(np.mean(abs(rival[use]-color[use])) - normal[use].mean()))
    if not errors:
        return None
    return [int(valid.sum()), round(float(normal[valid].mean()), 3),
            round(max(errors), 3), round(min(errors), 3)]


def rows(image, parts, P, original, competing=None):
    if not original or P is None:
        return []
    P = np.asarray(P, float)
    cells = sorted(original)
    indices = {p:i for i,p in enumerate(cells)}
    eye = np.linalg.solve(P[:, :3], -P[:, 3])
    inverse = np.linalg.inv(P[:, :3])
    native = P.copy()
    native[:2] *= SAMPLE_STRIDE
    shape = image.size[::-1]
    owner, axes = _frontmost_cells(cells, native, eye, shape, face_axes=True)
    source_owner, base_axes = owner, axes
    if competing:
        joint = cells + sorted(competing)
        lookup = {p:i for i,p in enumerate(joint)}
        source_owner, axes = _frontmost_cells(joint, native, eye, shape, face_axes=True)
    support = np.zeros(shape, bool)
    coarse = tuple((n+(SAMPLE_STRIDE-1)//2)//SAMPLE_STRIDE for n in shape)
    for part in parts:
        if part.get('support'):
            support |= _support_masks(part['support'], image.size, coarse)[1]
    # Find extensions only of a face with measured visibility in the base.
    foreground = owner >= 0
    observed_faces = owner[foreground][::3] * 3 + base_axes[foreground][::3]
    exposed = [divmod(int(value), 3) for value in np.unique(observed_faces)]
    candidates = defaultdict(list)
    for idx, axis in exposed:
        src = np.array(cells[idx])
        for delta in _offsets(axis):
            target = tuple((src-delta).tolist())
            if target in indices or target[1] < 0 or (competing is not None and target not in competing):
                continue
            candidates[target, axis].append((idx, delta))
    rgb = np.asarray(image.convert('RGB')).astype(np.float32)
    result = []
    for (cell, axis), anchors in sorted(candidates.items()):
        sign = 1 if eye[axis] >= cell[axis]+1 else -1
        if cell[axis] <= eye[axis] <= cell[axis]+1:
            continue
        hit = _sample_face(native, cell, axis, sign, shape)
        if hit is None:
            continue
        box, mask, *_ = hit
        valid = mask & (owner[box] < 0) & support[box]
        if competing:
            valid &= (source_owner[box] == lookup[cell]) & (axes[box] == axis)
        y, x = np.nonzero(valid)
        if len(x) < 12:
            continue
        observed = len(x)
        step = max(1, len(x)//160)
        y, x = y[::step]+box[0].start, x[::step]+box[1].start
        rays = np.column_stack([x+.5, y+.5, np.ones(len(x))]) @ inverse.T
        xyz = eye + rays * ((cell[axis]+int(sign>0)-eye[axis])/rays[:,axis])[:,None]
        color = rgb[y,x]
        matches = []
        for idx, delta in anchors:
            match = _repeat(rgb, source_owner, axes, P, xyz, color, idx, delta, axis,
                            [dim for dim in range(3) if dim != axis])
            if match:
                matches.append(match)
        if matches:
            result.append([list(cell), axis, observed, int(mask.sum()), matches])
    return result


def select_joint(rows, *, seed_fraction=.35, grow=True):
    seeds = set()
    possible = set()
    for cell, axis, seen, area, matches in rows:
        cell = tuple(cell)
        if seen >= seed_fraction * area and any(n >= 24 and err <= 4 and gain >= 2.5 and low >= .3
                                      for n, err, gain, low in matches):
            seeds.add((cell, axis))
        if seen >= .30 * area and any(n >= 20 and err <= 7 and gain >= 2 and low >= 0
                                      for n, err, gain, low in matches):
            possible.add((cell, axis))
    accepted = {cell for cell, axis in seeds}
    if grow:
        for cell, axis in possible:
            if any((tuple(cell+d), axis) in seeds for d in _offsets(axis)):
                accepted.add(cell)
    return accepted


def recover(image, parts, projection, base_cells):
    """Return additions with native phase matches that survive joint visibility."""
    if projection is None or not base_cells:
        return set()
    extra = select_joint(rows(image, parts, projection, base_cells))
    if extra:
        extra &= select_joint(rows(image, parts, projection, base_cells, competing=extra))
    if extra:
        from engine.visible_ray_filter import fully_occluded_cells
        extra -= fully_occluded_cells(image.size, projection, base_cells | extra)
    return extra
