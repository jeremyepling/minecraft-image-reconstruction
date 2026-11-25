"""Evaluation-only visible planes, continuous block spans and grid phase in native pixels."""
from collections import defaultdict
from itertools import groupby

import numpy as np

from benchmark.camera_labels import rasterize
from benchmark.camera_metrics import truth_camera
from engine.geometry import UNIT, project, transform


DOWNSAMPLE = 4
MIN_PATCH_FACES = 4
MIN_SUPPORT = 24  # quarter-resolution pixel centers
IOU_THRESHOLD = .5
PHASE_TOLERANCE = 3.  # source pixels
INSET_LOW, INSET_HIGH = np.array([.17, .23]), np.array([.26, .19])


def _components(cells):
    remaining = set(cells)
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, stack = [], [first]
        while stack:
            u, v = stack.pop()
            component.append((u, v))
            for neighbor in ((u-1, v), (u+1, v), (u, v-1), (u, v+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        yield component


def _largest_rect(cells):
    """Largest complete axis-aligned block rectangle with at least two cells per edge."""
    rows = defaultdict(set)
    for u, v in cells:
        rows[v].add(u)
    best, rank = None, (0, 0)
    for v0 in sorted(rows):
        shared = rows[v0].copy()
        v = v0+1
        while v in rows:
            shared.intersection_update(rows[v])
            if len(shared) < 2:
                break
            for _, members in groupby(enumerate(sorted(shared)), key=lambda a: a[1]-a[0]):
                run = [p for _, p in members]
                width, height = len(run), v-v0+1
                priority = (width*height, min(width, height))
                if width >= 2 and priority > rank:
                    best = (run[0], v0, run[-1]+1, v+1)
                    rank = priority
            v += 1
    return best


def _patch_chart(Q, axis, offset):
    free = [a for a in range(3) if a != axis]
    return np.column_stack((Q[:, free], Q[:, 3]+offset*Q[:, axis]))


def _rectangle(patch_id, H, cells):
    extent = _largest_rect(cells)
    if extent is None:
        return None
    lo, hi = np.asarray(extent[:2])+INSET_LOW, np.asarray(extent[2:])-INSET_HIGH
    uv = lo + UNIT*(hi-lo)
    q = project(H, uv)
    # Start at the image's upper left and wind clockwise, keeping UV indices paired.
    if (q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1)) < 0:
        q, uv = q[::-1], uv[::-1]
    start = int(np.argmin(q.sum(1)))
    q, uv = np.roll(q, -start, axis=0), np.roll(uv, -start, axis=0)
    counts = np.linalg.norm(np.diff(uv[:3], axis=0), axis=1)
    return dict(patch_id=patch_id, quad=q.tolist(), counts=counts.tolist(), block_bounds=list(extent))


def truth(pose, faces):
    """Partition adjacent coplanar visible block faces and z-buffer their quarter-scale support."""
    w, h = pose["image_size"]
    Q = truth_camera(pose)
    right = Q[0, :3] - w/2*Q[2, :3]
    down = Q[1, :3] - h/2*Q[2, :3]
    orientation = np.stack((right, down, Q[2, :3]))
    orientation /= np.linalg.norm(orientation, axis=1)[:, None]
    planes = defaultdict(dict)
    for face in faces:
        normal = tuple(face["normal"])
        axis = next(i for i, d in enumerate(normal) if d)
        level = face["world_corners"][0][axis]
        xy = tuple(p for i, p in enumerate(face["xyz"]) if i != axis)
        planes[(normal, level)][xy] = face
    scale = DOWNSAMPLE
    ids = np.zeros(((h+scale-1)//scale, (w+scale-1)//scale), np.int32)
    depths = np.zeros(ids.shape)
    patches, charts, vertices = {}, {}, {}
    for (normal, level), cells in sorted(planes.items()):
        axis = next(i for i, d in enumerate(normal) if d)
        H = _patch_chart(Q, axis, level)
        for component in _components(cells):
            pid = len(patches)+1
            full, visible_points = [], {}
            for cell in component:
                f = cells[cell]
                corners = np.asarray(f["world_corners"])
                homogeneous = corners @ Q[:, :3].T + Q[:, 3]
                homogeneous[:, :2] /= scale
                raster = rasterize(homogeneous, ids.shape[1], ids.shape[0])
                if raster is not None:
                    box, inside, inv_z, _, _ = raster
                    closer = inside & (inv_z > depths[box])
                    ids[box][closer] = pid
                    depths[box][closer] = inv_z[closer]
                if f["visible_fraction"] >= 1-1e-9 and all(f["corner_visible"]):
                    full.append(cell)
                for xyz, xy, vis in zip(corners, f["image_corners"], f["corner_visible"]):
                    if vis:
                        visible_points[tuple(xyz)] = xy
            vertices[pid] = np.array(list(visible_points.values()), dtype=float).reshape(-1, 2)
            charts[pid] = H
            patches[pid] = dict(id=pid, axis=axis, normal=list(normal), offset=level,
                                normal_camera=(orientation @ normal).tolist(), faces=len(component),
                                rectangle=_rectangle(pid, H, full))
    support = np.bincount(ids.ravel(), minlength=len(patches)+1)
    rectangles = []
    for pid, p in patches.items():
        p["support_px"] = int(support[pid])
        p["eligible"] = p["faces"] >= MIN_PATCH_FACES and p["support_px"] >= MIN_SUPPORT
        if p["eligible"] and p["rectangle"] is not None:
            rectangles.append(p["rectangle"])
    return dict(id_map=ids, image_size=[w,h], patches=patches, rectangles=rectangles,
                charts=charts, vertices=vertices)


def _one_to_one(pairs):
    # Start with strongest IoUs. Augment only unmatched proposals to resolve exact
    # threshold ties without sacrificing match cardinality or an uncontested match.
    options, held = defaultdict(list), {}
    matched = set()
    for iou, i, pid in sorted(pairs, key=lambda item: (-item[0], item[1], item[2])):
        options[i].append(pid)
        if i not in matched and pid not in held:
            held[pid] = i
            matched.add(i)
    def augment(i, seen):
        for pid in options[i]:
            if pid in seen:
                continue
            seen.add(pid)
            if pid not in held or augment(held[pid], seen):
                held[pid] = i
                return True
        return False
    for i in sorted(options.keys() - matched):
        augment(i, set())
    return sorted((i, pid) for pid, i in held.items())


def score_planes(t, proposals):
    """Match image-only quad proposals to eligible surface masks; measure union spill onto background."""
    ids, patches = t["id_map"], t["patches"]
    eligible = {pid for pid, p in patches.items() if p["eligible"]}
    union = np.zeros(ids.shape, dtype=bool)
    pairs, overlap = [], {}
    for i, prop in enumerate(proposals):
        q = np.asarray(prop["quad"], dtype=float)
        if q.shape != (4, 2) or not np.isfinite(q).all():
            continue
        homogeneous = np.column_stack((q/DOWNSAMPLE, np.ones(4)))
        try:
            raster = rasterize(homogeneous, ids.shape[1], ids.shape[0])
        except np.linalg.LinAlgError:
            continue
        if raster is None:
            continue
        box, inside = raster[:2]
        union[box] |= inside
        areas = np.bincount(ids[box][inside], minlength=len(patches)+1)
        area = int(inside.sum())
        for pid in eligible:
            hit = int(areas[pid])
            iou = hit / (area + patches[pid]["support_px"] - hit)
            if iou >= IOU_THRESHOLD:
                pairs.append((iou, i, pid))
                overlap[i, pid] = iou
    matches = []
    for i, pid in _one_to_one(pairs):
        quad = proposals[i]["quad"]
        center = project(transform(quad), [[.5, .5]])[0]
        xyz = np.linalg.solve(t["charts"][pid], [*center, 1.])
        depth = float(1 / xyz[2]) if xyz[2] > 0 else None
        matches.append(dict(proposal_index=i, patch_id=pid, iou=overlap[i, pid],
                            center_xy=center.tolist(), normal_camera=patches[pid]["normal_camera"],
                            depth_at_center=depth))
    pixels, spill = int(union.sum()), int(np.count_nonzero(union & (ids == 0)))
    return dict(eligible=len(eligible), proposed=len(proposals), matched=len(matches),
                precision=len(matches)/len(proposals) if proposals else None,
                recall=len(matches)/len(eligible) if eligible else None, matches=matches,
                proposed_pixels=pixels, spill_pixels=spill, spill_fraction=spill/pixels if pixels else None)


def _near_counts(expected, predicted):
    good = np.zeros(len(expected), dtype=bool)
    covered = 0
    for start in range(0, len(predicted), 128):
        delta = predicted[start:start+128, None, :] - expected[None, :, :]
        hit = np.sum(delta*delta, axis=2) <= PHASE_TOLERANCE**2
        covered += int(np.count_nonzero(hit.any(1)))
        good |= hit.any(0)
    return int(good.sum()), covered


def score_pitch(t, patch_id, quad, prediction):
    """Score edge lengths on the true plane and visible integer vertices inside the supplied quad."""
    q = np.asarray(quad, dtype=float)
    H = transform(q)
    uv = project(np.linalg.inv(t["charts"][patch_id]), q)
    lengths = np.linalg.norm(np.roll(uv, -1, axis=0) - uv, axis=1)
    expected = .5*(lengths[:2]+lengths[2:])
    target = t["vertices"][patch_id]
    crop = project(np.linalg.inv(H), target)
    target = target[np.all((crop >= -1e-7) & (crop <= 1+1e-7), axis=1)]
    counts = prediction.get("counts")
    grid = prediction.get("grid")
    points = np.empty((0, 2))
    count_error, half_both = None, False
    if counts is not None:
        counts = np.asarray(counts, dtype=float)
        if counts.shape != (2,) or not np.isfinite(counts).all() or np.any(counts <= 0):
            raise ValueError("Positive finite U and V block counts are required")
        error = abs(counts-expected)
        count_error = float(error.mean())
        half_both = bool(np.all(error <= .5+1e-9))
        if grid is not None:
            phase = np.asarray(grid["phase_blocks"], dtype=float) % 1.
            if phase.shape != (2,) or not np.isfinite(phase).all():
                raise ValueError("Two finite grid phases are required")
            axes = [(p+np.arange(int(np.floor(n-p+1e-8))+1))/n for n, p in zip(counts, phase)]
            u, v = np.meshgrid(*axes)
            points = project(H, np.column_stack((u.ravel(), v.ravel())))
        counts = counts.tolist()
    truth_hits, predicted_hits = _near_counts(target, points)
    precision = predicted_hits/len(points) if len(points) else None
    recall = truth_hits/len(target) if len(target) else None
    f1 = None
    if recall is not None:
        f1 = 2*precision*recall/(precision+recall) if precision and recall else 0.
    return dict(expected_counts=expected.tolist(), counts=counts, mean_count_error=count_error,
                half_block_both=half_both, truth_intersections=len(target), predicted_intersections=len(points),
                truth_within_3px=truth_hits, predicted_within_3px=predicted_hits,
                grid_precision=precision, grid_recall=recall, grid_f1=f1)
