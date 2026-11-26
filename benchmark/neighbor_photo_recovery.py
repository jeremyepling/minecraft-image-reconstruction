"""Test whether fractional-cell texture phase repeats across photographed neighboring blocks."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from engine.visible_edge_faces import _support_masks
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE
from benchmark.visible_depth_score import integrated


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
    exposed = set(zip(owner[owner >= 0][::3].tolist(), base_axes[owner >= 0][::3].tolist()))
    candidates = defaultdict(list)
    for idx, axis in sorted(exposed):
        src = np.array(cells[idx])
        for delta in _offsets(axis):
            target = tuple((src-delta).tolist())
            if target not in indices and target[1] >= 0 and (competing is None or target in competing):
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


def select(rows, variant):
    accepted = set()
    if variant.startswith('joint'):
        seeds = set()
        possible = set()
        for cell, axis, seen, area, matches in rows:
            cell = tuple(cell)
            fraction = .08 if variant == 'joint_partial' else .35
            if seen >= fraction * area and any(n >= 24 and err <= 4 and gain >= 2.5 and low >= .3
                                          for n, err, gain, low in matches):
                seeds.add((cell, axis))
            if seen >= .30 * area and any(n >= 20 and err <= 7 and gain >= 2 and low >= 0
                                          for n, err, gain, low in matches):
                possible.add((cell, axis))
        accepted.update(cell for cell, axis in seeds)
        if variant == 'joint_growth':
            for cell, axis in possible:
                if any((tuple(cell+d), axis) in seeds for d in _offsets(axis)):
                    accepted.add(cell)
        return accepted
    for cell, axis, seen, area, matches in rows:
        if variant == 'oracle_pool':
            yes = bool(matches)
        elif variant == 'permissive':
            yes = any(n >= 16 and err <= 10 and gain >= 1.2 for n, err, gain, _ in matches)
        elif variant == 'phase':
            yes = any(n >= 20 and err <= 7 and gain >= 2.5 and low >= -.3
                      for n, err, gain, low in matches)
        elif variant == 'strict':
            yes = any(n >= 24 and err <= 4 and gain >= 2.5 and low >= .3
                      for n, err, gain, low in matches)
        else:
            raise ValueError(variant)
        if yes:
            accepted.add(tuple(cell))
    return accepted


def predict(image, version, parts, counts, camera):
    # Private benchmark cache contains image-only intermediate rows and placement.
    cache = os.environ.get('MCIR_NEIGHBOR_PHOTO_CACHE')
    key = hashlib.sha256(image.convert('RGB').tobytes())
    key.update(json.dumps([version, image.size, parts, counts, camera], sort_keys=True).encode())
    path = Path(cache)/f'{key.hexdigest()}.json' if cache else None
    if path and path.exists():
        data = json.loads(path.read_text())
        base = set(map(tuple, data['base']))
        features = data['rows']
        status = {}
    else:
        base, status = integrated(image, version, parts, counts, camera)
        features = rows(image, parts, camera['P'], base)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(base=sorted(base), rows=features)))
    variant = os.environ.get('MCIR_NEIGHBOR_PHOTO_VARIANT','phase')
    extra = select(features, variant)
    if extra and variant.startswith('joint'):
        # Recheck every source and target pixel used in a match after resolving
        # visibility of all selected candidates together.
        extra &= select(rows(image, parts, camera['P'], base, competing=extra), variant)
    if extra:
        from engine.visible_ray_filter import fully_occluded_cells
        extra -= fully_occluded_cells(image.size, camera['P'], base | extra)
    return base | extra, status
