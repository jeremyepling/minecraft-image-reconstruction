"""Floor-free first-hit verification for fixed Floating depth options."""

from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmark.camera_labels import screen_matrix
from benchmark.expanded_baseline import read
from benchmark.relative_face_assembly import axis_vector
from benchmark.relative_face_scoring import _yaw
from engine.visible_placement import FACE_CORNERS, _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE, observed_support, rasterize


MIN_FACE_SAMPLES = 3
MIN_VISIBLE_SHARE = .55
MIN_REAL_GAIN = .035


def _project(matrix, points):
    homogeneous = points @ matrix[:, :3].T + matrix[:, 3]
    return homogeneous[..., :2] / homogeneous[..., 2, None], homogeneous[..., 2]


def _normalize(points):
    center = np.mean(points, axis=0)
    scale = np.sqrt(points.shape[1]) / np.linalg.norm(points - center, axis=1).mean()
    transform = np.eye(points.shape[1] + 1)
    transform[:-1, :-1] *= scale
    transform[:-1, -1] = -scale * center
    return (points - center) * scale, transform


def _dlt(world, screen):
    world, world_transform = _normalize(world)
    screen, screen_transform = _normalize(screen)
    ones = np.c_[world, np.ones(len(world))]
    rows = np.zeros((2 * len(world), 12))
    rows[::2, :4] = ones
    rows[1::2, 4:8] = ones
    rows[:, 8:] = -(screen[..., None] * ones[:, None, :]).reshape(-1, 4)
    _, singular, basis = np.linalg.svd(rows, full_matrices=False)
    if singular[-2] < singular[0] * 1e-4:
        return None
    matrix = np.linalg.solve(screen_transform, basis[-1].reshape(3, 4)) @ world_transform
    matrix /= np.linalg.norm(matrix[2, :3])
    if np.median(_project(matrix, np.zeros((1, 3)))[1]) < 0:
        matrix *= -1
    return matrix


def _anchors(chart, origin, excluded, shape):
    eligible = {(cell['u'], cell['v']) for cell in chart['cells'] if not cell.get('normal_offset')}
    # Keep one block of separation from the moving patch in the fitting image.
    withheld = {(u + du, v + dv) for u, v in excluded for du in (-1, 0, 1) for dv in (-1, 0, 1)}
    uv = np.array(sorted(eligible - withheld))
    if len(uv) < 8 or min(np.ptp(uv, axis=0)) < 2:
        return None
    steps = np.array([axis_vector(code) for code in chart['axes']])
    h = np.asarray(chart['H'], float)
    corners = np.unique((uv[:, None, :] + FACE_CORNERS).reshape(-1, 2), axis=0)
    pixels, _ = _project(np.c_[h[:, :2], np.zeros(3), h[:, 2]], np.c_[corners, np.zeros(len(corners))])
    margin = 4
    in_screen = (np.isfinite(pixels).all(axis=1) & (pixels >= margin).all(axis=1)
                 & (pixels < np.array(shape[::-1]) - margin).all(axis=1))
    corners, pixels = corners[in_screen], pixels[in_screen]
    if len(corners) < 12:
        return None
    # Spatially spread fitting points and independent interior corner checks.
    rays = np.array([(1, 1), (1, -1), (-1, -1), (-1, 1), (2, 1), (1, 2)])
    fit = np.unique(np.argmax(corners @ rays.T, axis=0))
    verify = np.setdiff1d(np.arange(len(corners)), fit)
    if len(fit) < 3 or len(verify) < 6:
        return None
    world = np.asarray(origin) + corners @ steps
    return world[fit], pixels[fit], world[verify], pixels[verify]


def fit_projection(charts, origins, excluded, shape):
    """Fit two assembled non-parallel source faces in an arbitrary local gauge."""
    choices = []
    for chart in charts:
        key = chart['id']
        if key not in origins:
            continue
        anchors = _anchors(chart, origins[key], excluded.get(key, ()), shape)
        if anchors is not None:
            choices.append((chart, anchors))
    fits = []
    for (a, aa), (b, ba) in combinations(choices, 2):
        if abs(a['normal_axis']) == abs(b['normal_axis']):
            continue
        world, screen, checks, observations = [np.concatenate(pair) for pair in zip(aa, ba, strict=True)]
        if len(world) < 8:
            continue
        anchor = world[0]
        try:
            matrix = _dlt(world - anchor, screen)
            if matrix is None:
                continue
            eye = np.linalg.solve(matrix[:, :3], -matrix[:, 3])
            camera, _ = cv2.RQDecomp3x3(matrix[:, :3])[1:3]
        except (np.linalg.LinAlgError, cv2.error):
            continue
        height, width = shape
        if (min(abs(camera[0, 0]), abs(camera[1, 1])) < .22 * width
                or min(abs(camera[0, 0]), abs(camera[1, 1])) > 4 * width
                or not .78 <= abs(camera[0, 0] / camera[1, 1]) <= 1.28
                or abs(camera[0, 1]) > .15 * width
                or not abs(abs(camera[0, 2]) - width / 2) < width * .24
                or not abs(abs(camera[1, 2]) - height / 2) < height * .24):
            continue
        if any(((eye + anchor - origins[c['id']]) @ axis_vector(c['normal_axis'])) <= 0 for c in (a, b)):
            continue
        observed, near = _project(matrix, checks - anchor)
        if not np.all(near > 0):
            continue
        errors = np.linalg.norm(observed - observations, axis=1)
        median, p90 = np.quantile(errors, [.5, .9])
        if not (median <= 3.0 and p90 <= 6.0):
            continue
        matrix[:, 3] -= matrix[:, :3] @ anchor
        fits.append((p90, matrix, eye + anchor, dict(charts=[a['id'], b['id']],
                     anchor=anchor.tolist(), fit_corners=len(world), held_out=len(checks),
                     median_error=float(median), p90_error=float(p90))))
    if not fits:
        return None
    _, matrix, eye, details = min(fits, key=lambda fit: fit[0])
    return matrix, eye, details


def _raster_grid(h, uv, shape):
    mask = np.zeros(shape, bool)
    for u, v in uv:
        corners = np.c_[FACE_CORNERS + (u, v), np.ones(4)] @ h.T
        if np.any(corners[:, 2] <= 0):
            continue
        corners[:, :2] /= SAMPLE_STRIDE
        raster = rasterize(corners, shape[1], shape[0])
        if raster is not None:
            box, inside, *_ = raster
            mask[box] |= inside
    return mask


def _face_seams(cells, axis, eye, matrix, gradient):
    """Compare projected face joins with native image gradients and off-edge probes."""
    tangents = [dimension for dimension in range(3) if dimension != axis]
    sign = int(eye[axis] > cells[0][axis])
    space = set(cells)
    endpoints = []
    for coord in cells:
        for across, along in (tangents, tangents[::-1]):
            nxt = np.array(coord)
            nxt[across] += 1
            if tuple(nxt) not in space:
                continue
            nxt[axis] += sign
            end = nxt.copy()
            end[along] += 1
            endpoints.append([nxt, end])
    if not endpoints:
        return None
    xy, w = _project(matrix, np.asarray(endpoints))
    vector = xy[:, 1] - xy[:, 0]
    lengths = np.linalg.norm(vector, axis=1)
    keep = (w > 0).all(axis=1) & (lengths >= 6) & np.isfinite(lengths)
    if keep.sum() < 3:
        return None
    vector, xy, lengths = vector[keep], xy[keep], lengths[keep]
    normal = vector[:, ::-1] * [-1, 1] / lengths[:, None]
    center = xy[:, :1, :] + np.array([.28, .5, .72])[None, :, None] * vector[:, None, :]
    probes = center[:, :, None, :] + np.array([0, -3, 3])[None, None, :, None] * normal[:, None, None, :]
    valid = ((probes >= 0) & (probes < np.array(gradient.shape[::-1]) - 1)).all(axis=(1, 2, 3))
    if valid.sum() < 3:
        return None
    x, y = probes[valid].astype(np.float32).reshape(-1, 2).T
    samples = cv2.remap(gradient, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1, 3, 3)
    center, nearby = samples[:, :, 0], samples[:, :, 1:].mean(axis=2)
    return float(np.median((center - nearby) / (center + nearby + 15.)))


def _patch_cells(chart, origin, uv, offset):
    u, v = [axis_vector(code) for code in chart['axes']]
    normal = axis_vector(chart['normal_axis'])
    cell_corner = np.asarray(origin) + np.minimum(0, u) + np.minimum(0, v) - np.maximum(0, normal)
    return [tuple(cell_corner + a * u + b * v + offset * normal) for a, b in uv]


def _edge_response(face, evidence):
    edge = face & ~cv2.erode(face.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
    if edge.sum() < 6:
        return None
    return float(evidence[edge].mean())


def _grade(cells, moved, axis, matrix, eye, observed, protected, original, opening, gradient, rim):
    if not cells or not moved or not set(moved).issubset(cells):
        return dict(eligible=False, reason='patch collisions or missing decoded cells')
    owner, axes = _frontmost_cells(cells, matrix, eye, observed.shape, face_axes=True)
    indices = {cell: i for i, cell in enumerate(cells)}
    hit = (owner >= 0) & np.isin(owner, [indices[cell] for cell in moved])
    face_hits = hit & (axes == axis) & observed
    counts = np.bincount(owner[face_hits], minlength=len(cells))
    total = np.bincount(owner[hit & (axes == axis)], minlength=len(cells))
    supported = all(counts[indices[cell]] >= MIN_FACE_SAMPLES and
                    counts[indices[cell]] >= MIN_VISIBLE_SHARE * total[indices[cell]] for cell in moved)
    owners_in_original = np.array([original.get(cell, -2) for cell in cells])
    neighbor_owner, neighbor_axis = protected
    covered = ((owner >= 0) & ~hit & (owners_in_original[owner] == neighbor_owner)
               & (axes == neighbor_axis)).sum()
    mask_overlap = face_hits.sum() / max(1, (observed | (hit & (axes == axis))).sum())
    # Native source gradients are independent of the chart homography. Measure
    # the boundary of the visible projected patch, including cutouts from occlusion.
    return dict(eligible=bool(supported), neighbors=int(covered), openings=int((hit & opening).sum()),
                support=float(mask_overlap), seen=int(np.count_nonzero(counts)), blocks=len(moved),
                rim=_edge_response(hit & (axes == axis), rim),
                seam=_face_seams(moved, axis, eye, matrix, gradient))


def _context(source):
    raw = read(Path(source['chart_packet']))
    charts = [dict(chart, **{key: source['oriented'][chart['id']][key] for key in ('axes', 'normal_axis')})
              for chart in raw['charts'] if chart['id'] in source['oriented']]
    return charts, np.asarray(Image.open(source['image']).convert('RGB'))


def _rank(source, projection, charts, rgb, depth_scores=None):
    if projection is None:
        return dict(choice='incumbent', reason='no reliable two-face projection')
    matrix, eye, fit = projection
    chart_ids = {chart['id']: chart for chart in charts}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.25)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=.25)
    gradient = np.hypot(dx, dy)
    shape = tuple(n // SAMPLE_STRIDE for n in gray.shape)
    rim = cv2.resize(gradient, shape[::-1], interpolation=cv2.INTER_AREA)
    rim = np.minimum(rim / 30., 1.)
    planes = {plane['id']: plane['support'] for plane in read(Path(source['surfaces']))['planes']}
    masks = {key: observed_support(data, shape, gray.shape[::-1]) for key, data in planes.items()}
    union = np.logical_or.reduce(list(masks.values()))
    opening = ~cv2.dilate(union.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
    options = source['options']
    original = sorted(tuple(xyz) for xyz, _ in read(Path(options['incumbent']['local_blocks'])))
    original_sites = {cell: i for i, cell in enumerate(original)}
    incumbent_owner, incumbent_axes = _frontmost_cells(original, matrix, eye, shape, face_axes=True)
    choice, highest, details = 'incumbent', 0., {}
    for name, option in options.items():
        if name == 'incumbent':
            continue
        changed = [(part, option['offsets'][part['id']]) for part in source['patches']
                   if option['offsets'][part['id']] != part['incumbent']]
        if len(changed) != 1:
            continue
        part, new_depth = changed[0]
        key = part['chart']
        if key not in source['origins'] or key not in chart_ids or key not in masks:
            continue
        chart = chart_ids[key]
        h = np.asarray(part['parent_H'], float)
        h = h * np.copysign(1, h[2, 2])
        observed = _raster_grid(h, part['old_uv'], shape) & masks[key]
        if observed.sum() < 20:
            continue
        before = _patch_cells(chart, source['origins'][key], part['uv'], part['incumbent'])
        after = _patch_cells(chart, source['origins'][key], part['uv'], new_depth)
        original_patch = set(before)
        if set(after).intersection(original_sites.keys() - original_patch):
            details[name] = dict(reason='depth option collides with frozen neighbors')
            continue
        base_indices = [i for i, cell in enumerate(original) if cell in original_patch]
        protected = (np.where(union & (incumbent_owner >= 0) & ~np.isin(incumbent_owner, base_indices),
                              incumbent_owner, -1), incumbent_axes)
        axis = abs(chart['normal_axis']) - 1
        base = _grade(original, before, axis, matrix, eye, observed, protected, original_sites, opening, gradient, rim)
        moved = sorted(tuple(xyz) for xyz, _ in read(Path(option['local_blocks'])))
        trial = _grade(moved, after, axis, matrix, eye, observed, protected, original_sites, opening, gradient, rim)
        record = dict(base=base, trial=trial)
        details[name] = record
        if not trial['eligible'] or 'support' not in base:
            continue
        if trial['neighbors'] < base['neighbors'] - max(2, .01 * base['neighbors']):
            continue
        if trial['openings'] > base['openings'] + 1:
            continue
        overlap_gain = trial['support'] - base['support']
        seam_gain = (trial['seam'] - base['seam']) if None not in (trial['seam'], base['seam']) else 0.
        rim_gain = (trial['rim'] - base['rim']) if None not in (trial['rim'], base['rim']) else 0.
        record.update(overlap_gain=float(overlap_gain), seam_gain=float(seam_gain), rim_gain=float(rim_gain))
        # Agreement with the old chart alone favors restoring its flat plane.
        # A move needs independent support in the photograph as well.
        if overlap_gain <= .015 or max(seam_gain, rim_gain) < .02 or min(seam_gain, rim_gain) < -.035:
            continue
        gain = .65 * overlap_gain + .35 * max(seam_gain, rim_gain)
        if depth_scores is not None:
            gain = .85 * gain + .15 * np.clip(depth_scores.get(name, 0.) - depth_scores.get('incumbent', 0.), -.25, .25)
        record['gain'] = float(gain)
        if gain > highest + 1e-8:
            choice, highest = name, gain
    competing = sorted((row['gain'] for row in details.values() if 'gain' in row), reverse=True)
    if highest < MIN_REAL_GAIN or (len(competing) > 1 and highest - competing[1] < .02):
        choice = 'incumbent'
    return dict(choice=choice, fit=fit, options=details)


def rank_options(source, depth_scores=None):
    if len(source['options']) == 1:
        return dict(choice='incumbent', reason='no eligible patch in published handoff')
    charts, rgb = _context(source)
    active = {}
    for part in source['patches']:
        active.setdefault(part['chart'], []).extend(part['old_uv'])
    projection = fit_projection(charts, source['origins'], active, rgb.shape[:2])
    return _rank(source, projection, charts, rgb, depth_scores)


def rank_reference_camera(source, frame, yaw_quarters=None, translation=None):
    if len(source['options']) == 1 or yaw_quarters is None or translation is None:
        return dict(choice='incumbent', reason='no options or no resolved incumbent registration')
    charts, rgb = _context(source)
    pose = read(Path(frame) / 'camera.json')
    camera = screen_matrix(pose)
    rotation = np.asarray([_yaw(axis, yaw_quarters) for axis in np.eye(3, dtype=int)]).T
    shift = np.asarray(translation) - np.asarray(pose['eye_world'])
    matrix = np.c_[camera[:, :3] @ rotation, camera[:, :3] @ shift + camera[:, 3]]
    eye = np.linalg.solve(matrix[:, :3], -matrix[:, 3])
    details = dict(reference=True, yaw_quarters=yaw_quarters, incumbent_translation=translation)
    return _rank(source, (matrix, eye, details), charts, rgb)
