"""Resolve a pair of counted wall charts against their common corner on the ground."""
from itertools import combinations, product

import numpy as np

from engine.pillar_seams import orientation
from engine.visible_grid import _blocks, _carriers, _cell_map
from engine.visible_raster import SAMPLE_STRIDE, observed_support, rasterize


def face_iou(projection, mask, axis, level, start, width, height):
    free = 2 - axis
    quad = np.zeros((4, 3))
    quad[:, axis] = level
    quad[:, free] = [start, start + width, start + width, start]
    quad[:, 1] = [0, 0, height, height]
    screen = quad @ projection[:, :3].T + projection[:, 3]
    if screen[:, 2].min() <= 1.:
        return 0.
    screen[:, :2] /= SAMPLE_STRIDE
    samples = rasterize(screen, mask.shape[1], mask.shape[0])
    if samples is None:
        return 0.
    box, pixels, *_ = samples
    common = np.count_nonzero(pixels & mask[box])
    return common / (int(mask.sum()) + int(pixels.sum()) - common)


def grounded_pairs(image, charts, parts, masks, P):
    """Return the two signed planes only for a unique, fully measured convex corner."""
    walls = []
    for i, (chart, part) in enumerate(zip(charts, parts, strict=True)):
        spans = chart['pitch']['counts']
        if spans is None or chart['quad'] is None or part.get('normal_camera') is None:
            continue
        spans = np.asarray(spans, float)
        quad = np.asarray(chart['quad'], float)
        if spans.shape != (2,) or quad.shape != (4, 2) or not np.isfinite(spans).all() or not np.isfinite(quad).all():
            continue
        size = np.rint(spans)
        if min(size) < 2 or max(abs(size - spans)) > .2:
            continue
        walls.append((i, size, quad))
    if len(walls) < 2:
        return {}
    if P.shape != (3, 4) or not np.isfinite(P).all() or np.linalg.matrix_rank(P[:, :3]) != 3:
        return {}
    projection, rotation = orientation(dict(P=P), image.size)
    inverse = np.linalg.inv(projection[:, :3])
    eye = -inverse @ projection[:, 3]
    match = {}
    for (i, dims, first), (j, other, second) in combinations(walls, 2):
        if dims[1] != other[1]:
            continue
        normals = [rotation.T @ parts[k]['normal_camera'] for k in (i, j)]
        axes = [int(np.argmax(abs(n))) for n in normals]
        if set(axes) != {0, 2}:
            continue
        signs = [int(np.sign(n[a])) for a, n in zip(axes, normals)]
        # Check both endpoints so a projection that merely crosses a second wall
        # does not pose as a shared vertical edge.
        edges = [(a, b) for a, b in product((0, 1), repeat=2)
                 if max(np.linalg.norm(first[a] - second[b]),
                        np.linalg.norm(first[3-a] - second[3-b])) <= 6.]
        if len(edges) != 1:
            continue
        a, b = edges[0]
        bottom = .5 * (first[3-a] + second[3-b])
        ray = inverse @ [*bottom, 1.]
        if ray[1] >= -1.e-7:
            continue
        contact = eye - ray * (eye[1] / ray[1])
        nearest = np.rint(contact[axes])
        fits = []
        for delta in product((-1, 0, 1), repeat=2):
            left, right = (int(v) for v in nearest + delta)
            if signs[0] * (eye[axes[0]] - left) <= 0 or signs[1] * (eye[axes[1]] - right) <= 0:
                continue
            left_iou = face_iou(projection, masks[i], axes[0], left, right, -signs[1] * dims[0], dims[1])
            right_iou = face_iou(projection, masks[j], axes[1], right, left, -signs[0] * other[0], other[1])
            if min(left_iou, right_iou) >= .90:
                fits.append((left, right))
        if len(fits) == 1:
            if match:
                return {}
            match = {k: (axis, level, sign) for k, axis, level, sign in zip((i, j), axes, fits[0], signs)}
    return match


def edits(image, parts, counts, status, placed):
    """Keep edits within the two source masks and preserve independently supported seeds."""
    from engine.visible_placement import MINIMUM_SAMPLES, _charts, _frontmost_cells, _sample_face

    if not placed or status.get('P') is None or not counts:
        return set(), set(), {}
    charts = _charts(image, '26.3', parts, counts)
    shape = tuple((n + (SAMPLE_STRIDE - 1)//2)//SAMPLE_STRIDE for n in image.size[::-1])
    masks = [observed_support(p['support'], shape, image.size) if p.get('support')
             else np.zeros(shape, bool) for p in parts]
    projection = np.asarray(status['P'], float)
    anchors = grounded_pairs(image, charts, parts, masks, projection)
    chosen = {r['id']: (r['axis'], r['level']) for r in status.get('plane_offsets', [])}
    if not anchors or any(chosen.get(parts[i]['id']) != plane[:2] for i, plane in anchors.items()):
        return set(), set(), {}
    carriers = _carriers(image, charts, masks, projection)
    by_mask = {id(m): plane for m, plane, _ in carriers}
    incompatible = {i for i, plane in anchors.items() if by_mask.get(id(masks[i])) not in (None, plane)}
    if not incompatible:
        return set(), set(), {}
    cache = {}
    old, _, _ = _blocks([masks[i] for i in incompatible], carriers, projection, cache)
    keep, _, _ = _blocks([m for i, m in enumerate(masks) if i not in incompatible], carriers, projection, cache)
    candidates = {}
    for i, plane in anchors.items():
        axis, level, sign = plane
        _, start, other_sign = next(p for j, p in anchors.items() if j != i)
        width, height = np.rint(charts[i]['pitch']['counts']).astype(int)
        end = start - other_sign * width
        grid = _cell_map(projection, shape, plane)
        covered = np.bincount(grid.inverse, weights=masks[i].ravel()[grid.valid_pixels], minlength=len(grid.cells))
        for coord in grid.cells[(covered >= MINIMUM_SAMPLES) & (covered >= .75 * grid.counts)]:
            cell = np.zeros(3, int)
            cell[axis] = level - int(sign > 0)
            cell[[k for k in range(3) if k != axis]] = coord
            if 0 <= cell[1] < height and min(start, end) <= cell[2-axis] < max(start, end):
                candidates.setdefault(tuple(cell.tolist()), []).append(i)
    remove = (placed & old) - keep - candidates.keys()
    eye = -np.linalg.solve(projection[:, :3], projection[:, 3])
    # A real step or neighboring surface may also have visible evidence on a
    # disputed cell. Preserve it when its first-hit samples reach another mask.
    others = [m for i, m in enumerate(masks) if i not in anchors]
    if remove and others:
        pair = np.logical_or.reduce([masks[i] for i in anchors])
        external = np.logical_or.reduce(others) & ~pair
        baseline = sorted(placed)
        owner = _frontmost_cells(baseline, projection, eye, shape)
        visible = owner[external & (owner >= 0)]
        supported = np.bincount(visible, minlength=len(baseline))
        remove.difference_update(baseline[k] for k in np.flatnonzero(supported >= MINIMUM_SAMPLES))
    trial = sorted((placed - remove) | candidates.keys())
    owner, axes = _frontmost_cells(trial, projection, eye, shape, face_axes=True)
    index = {cell: k for k, cell in enumerate(trial)}
    added = set()
    for cell in candidates.keys() - placed:
        for i in candidates[cell]:
            axis, _, sign = anchors[i]
            hit = _sample_face(projection, cell, axis, sign, shape)
            if hit is None:
                continue
            box, pixels, *_ = hit
            visible = (owner[box] == index[cell]) & (axes[box] == axis) & masks[i][box] & pixels
            if visible.sum() >= max(MINIMUM_SAMPLES, .75 * pixels.sum()):
                added.add(cell)
                break
    return added, remove, {parts[i]['id']: plane for i, plane in anchors.items()}

