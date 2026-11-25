"""Partition a merged wall with finite, junction-supported image boundaries."""
from itertools import chain

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.camera_lines import _runs
from engine.mask_first import _image_depth
from engine.normal_regions import _local_mask, _patch
from engine.surface_depth_split import _slope
from engine.surface_model import canonical
from engine.surface_wall_split import SETTINGS as WALL_SETTINGS, _score


# Distances are expressed in working mask pixels after mapping from the native screenshot.
SETTINGS = dict(detector='scalelsd', minimum_segment=9., minimum_chain=14., max_gap=5.,
                max_angle=6., collinear_distance=2., cross_angle=24., junction_radius=3.5,
                min_shared=10, min_covered=.78, endpoint_margin=3., barrier_width=3,
                barrier_extension=2., cell_core_fraction=.70)


def _tracks(evidence, scale):
    """Join collinear segments and require a crossing or meeting at both ends."""
    segments = np.asarray(evidence.get('segments', []), float).reshape(-1, 2, 2) / scale
    segments = segments[np.isfinite(segments).all(axis=(1, 2))]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    keep = lengths >= SETTINGS['minimum_segment']
    segments, lengths = segments[keep], lengths[keep]
    if not len(segments):
        return []
    directions = (segments[:, 1] - segments[:, 0]) / lengths[:, None]
    parallel = np.cos(np.deg2rad(SETTINGS['max_angle']))
    crossing = np.cos(np.deg2rad(SETTINGS['cross_angle']))
    used = set()
    tracks = []
    for i in np.argsort(-lengths):
        start, direction = segments[i, 0], directions[i]
        normal = np.array([-direction[1], direction[0]])
        relative = segments - start
        span = relative @ direction
        aligned = ((abs(directions @ direction) >= parallel)
                   & (np.max(abs(relative @ normal), axis=1) <= SETTINGS['collinear_distance']))
        lo, hi = 0., lengths[i]
        while True:
            group = np.flatnonzero(aligned & (span.min(axis=1) <= hi + SETTINGS['max_gap'])
                                   & (span.max(axis=1) >= lo - SETTINGS['max_gap']))
            first, last = span[group].min(), span[group].max()
            if first >= lo and last <= hi:
                break
            lo, hi = min(first, lo), max(last, hi)
        token = tuple(group)
        if token in used or hi - lo < SETTINGS['minimum_chain']:
            continue
        used.add(token)
        a, b = start + lo * direction, start + hi * direction

        def is_junction(point):
            along = np.sum((point - segments[:, 0]) * directions, axis=1).clip(0, lengths)
            distance = np.linalg.norm(point - segments[:, 0] - directions * along[:, None], axis=1)
            return np.any((distance <= SETTINGS['junction_radius']) & (abs(directions @ direction) < crossing))

        if is_junction(a) and is_junction(b):
            tracks.append((a, b, direction, normal))
    return tracks


def _proposals(mask, left, top, tracks):
    """Propose a child only if the finite track supports most of its new boundary."""
    y, x = np.indices(mask.shape)
    xy = np.stack((x + left + .5, y + top + .5), axis=-1)
    for a, b, direction, normal in tracks:
        if (np.any(np.maximum(a, b) < [left, top])
                or np.any(np.minimum(a, b) > [left + mask.shape[1], top + mask.shape[0]])):
            continue
        signed = (xy - a) @ normal
        shared = mask & (abs(signed) < 1.)
        if shared.sum() < SETTINGS['min_shared']:
            continue
        along = (xy - a) @ direction
        margin = SETTINGS['endpoint_margin']
        finite = (along >= -margin) & (along <= np.linalg.norm(b - a) + margin)
        if (shared & finite).sum() < SETTINGS['min_shared']:
            continue
        # An island on one side of a finite line can be one of several pillars under a beam.
        for positive in (True, False):
            for rows in _runs(mask & ((signed >= 0) == positive)):
                if sum(end - start for _, start, end in rows) < WALL_SETTINGS['min_area']:
                    continue
                child = np.zeros_like(mask)
                for row, start, end in rows:
                    child[row, start:end] = True
                rest = np.pad(mask & ~child, 1)
                border = child & (rest[1:-1, :-2] | rest[1:-1, 2:] | rest[:-2, 1:-1] | rest[2:, 1:-1])
                total = int(border.sum())
                if total >= SETTINGS['min_shared'] and (border & finite).sum() >= SETTINGS['min_covered'] * total:
                    yield child, [a.tolist(), b.tolist()]


def _closed_cells(mask, barrier):
    """Follow boundaries sealed by several tracks together and the original perimeter."""
    labels = np.zeros(mask.shape, np.int32)
    areas = []
    for i, component in enumerate(_runs(mask & ~barrier), 1):
        area = 0
        for row, start, end in component:
            labels[row, start:end] = i
            area += end - start
        areas.append(area)
    minimum = max(WALL_SETTINGS['min_area'], mask.sum() * WALL_SETTINGS['min_fraction'])
    core_area = minimum * SETTINGS['cell_core_fraction']
    if sum(area >= core_area for area in areas) < 2:
        return
    # Restore pixels underneath the rasterized tracks to their nearest reachable child.
    while np.any(mask & (labels == 0)):
        padded = np.pad(labels, 1)
        neighbor = np.maximum.reduce((padded[:-2, 1:-1], padded[2:, 1:-1],
                                      padded[1:-1, :-2], padded[1:-1, 2:]))
        add = (labels == 0) & mask & (neighbor > 0)
        if not add.any():
            return
        labels[add] = neighbor[add]
    for label, area in enumerate(areas, 1):
        if area >= core_area:
            yield labels == label, None


def refine(image, parts, lines, normals, depth):
    """Split saved mask supports. The upstream Wall Separator supplies each parent normal."""
    if not parts:
        return []
    size = tuple(parts[0]['support']['size'])
    scale = np.asarray(image.size) / size
    tracks = _tracks(lines[SETTINGS['detector']], scale)
    if not tracks:
        return list(parts)
    canvas = Image.new('1', size)
    draw = ImageDraw.Draw(canvas)
    extension = SETTINGS['barrier_extension']
    for a, b, direction, _ in tracks:
        draw.line(tuple(np.r_[a - extension * direction, b + extension * direction]),
                  fill=1, width=SETTINGS['barrier_width'])
    barriers = np.asarray(canvas)
    z = _image_depth(np.asarray(depth), size)
    small = image.convert('RGB').resize(size, Image.Resampling.BOX)
    rgb = np.asarray(small.filter(ImageFilter.GaussianBlur(1.25)), np.float32)
    gray = np.asarray(small.convert('L').filter(ImageFilter.GaussianBlur(.9)), np.float32)[..., None]
    output = []
    for part in parts:
        mask, left, top = _local_mask(np.asarray(part['support']['runs']))
        if min(mask.shape) < 12 or mask.sum() < 2 * WALL_SETTINGS['min_area']:
            output.append(part)
            continue
        box = np.s_[top:top + mask.shape[0], left:left + mask.shape[1]]
        local = z[box]
        y, x = np.indices(mask.shape)
        residual = local - _slope(local, mask, 0) * x - _slope(local, mask, 1) * y
        best = None
        sides = chain(_closed_cells(mask, barriers[box]), _proposals(mask, left, top, tracks))
        for side, line in sides:
            result = _score(mask, side, residual, local, rgb[box])
            if result is not None and (best is None or result[0] > best[0]):
                best = (*result, line)
        if best is None:
            output.append(part)
            continue
        _, components, evidence, line = best
        for i, component in enumerate(components):
            runs = [[v + top, start + left, end + left] for v, start, end in component]
            values = np.concatenate([z[v, start:end] for v, start, end in runs])
            child = dict(id=f"{part['id']}-junction-{i}", parent_id=part['id'],
                         support=dict(size=list(size), runs=runs), support_area=len(values),
                         normal_camera=part['normal_camera'], relative_inverse_depth=float(np.median(values)),
                         junction_split=dict(line=line, **evidence), geometry_status='mask only')
            chart = _patch(runs, gray, scale, size, np.asarray(child['normal_camera']))
            if chart is not None:
                child.update(quad=canonical(chart['quad']), geometry_status='mask with sampling chart')
            output.append(child)
    return output
