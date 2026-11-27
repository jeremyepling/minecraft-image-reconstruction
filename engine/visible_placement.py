"""Place visible surface cells in the block grid estimated from one screenshot."""
import numpy as np

from engine.camera_metric import predict as seam_camera
from engine.masked_count import count
from engine.surface_model import intrinsic
from engine.surface_sample import sample_surface
from engine.visible_grid import _blocks, _carriers
from engine.visible_raster import SAMPLE_STRIDE, observed_support, rasterize


MINIMUM_SAMPLES = 4
MINIMUM_COVERAGE = .5
FACE_CORNERS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def _sample_face(projection, cell, axis, sign, shape):
    face = np.tile(cell, (4, 1))
    face[:, axis] += int(sign > 0)
    face[:, [i for i in range(3) if i != axis]] += FACE_CORNERS
    projected = face @ projection[:, :3].T + projection[:, 3]
    if projected[:, 2].min() <= 1e-5:
        return None
    projected[:, :2] /= SAMPLE_STRIDE
    try:
        return rasterize(projected, shape[1], shape[0])
    except np.linalg.LinAlgError:
        return None


def _frontmost_cells(cells, projection, eye, shape, *, face_axes=False):
    """Assign each sampled image ray to the nearest projected face in the candidate build."""
    inverse_depth = np.zeros(shape)
    owner = np.full(shape, -1, dtype=int)
    axes = np.full(shape, -1, dtype=np.int8) if face_axes else None
    occupied = set(cells)
    for index, cell in enumerate(cells):
        for axis in range(3):
            if cell[axis] <= eye[axis] <= cell[axis] + 1:
                continue
            sign = 1 if eye[axis] > cell[axis] else -1
            neighbor = list(cell)
            neighbor[axis] += sign
            if tuple(neighbor) in occupied:
                continue
            hit = _sample_face(projection, cell, axis, sign, shape)
            if hit is None:
                continue
            box, pixels, depth, _, _ = hit
            closer = pixels & (depth > inverse_depth[box])
            inverse_depth[box][closer] = depth[closer]
            owner[box][closer] = index
            if face_axes:
                axes[box][closer] = axis
    return (owner, axes) if face_axes else owner


def _estimate_plane_cells(image, parts, masks, projection, eye, select_level=None, *, partial=None, axes=None, walls_first=False, reconcile=None):
    """Fit provisional grid planes using floor contacts and relative surface depths."""
    focal = np.linalg.norm(projection[0, :3] - image.width / 2 * projection[2, :3])
    rotation = np.linalg.solve(intrinsic(image.size, focal), projection[:, :3])
    shape = masks[0].shape
    yy, xx = np.indices(shape)
    pixels = np.stack([(xx + .5) * SAMPLE_STRIDE, (yy + .5) * SAMPLE_STRIDE, np.ones(shape)], axis=-1)
    rays = pixels @ np.linalg.inv(projection[:, :3]).T
    union = np.logical_or.reduce(masks)
    bottom = np.where(union, yy, -1).max(axis=0)
    boundary = union & (yy == bottom) & (yy < shape[0] - 1)
    floor_depth = np.divide(-eye[1], rays[:, :, 1], out=np.full(shape, np.nan), where=abs(rays[:, :, 1]) > 1e-7)
    contacts = boundary & (floor_depth > 1) & (floor_depth < 150)
    distance = np.median(floor_depth[contacts]) if contacts.any() else focal / (image.height / 24)
    depths = [p['relative_inverse_depth'] for p in parts if (p.get('relative_inverse_depth') or 0) > 0]
    typical = np.median(depths) if depths else 1
    candidates = [set() for _ in parts]
    indices = range(len(parts))
    if walls_first:
        # Horizontal heights can use the independently observed wall levels.
        indices = sorted(indices, key=lambda i: parts[i].get('normal_camera') is not None
                         and np.argmax(abs(rotation.T @ parts[i]['normal_camera'])) == 1)
    selections = []
    for index in indices:
        part, mask = parts[index], masks[index]
        if not mask.any() or part.get('normal_camera') is None:
            continue
        axis = int(np.argmax(abs(rotation.T @ part['normal_camera'])))
        if axes is not None:
            axes[index] = axis
        contact = mask & contacts
        if axis != 1 and contact.sum() >= max(4, .12 * mask.any(axis=0).sum()):
            intercepts = eye[axis] + floor_depth[contact] * rays[contact, axis]
            level = int(np.rint(np.median(intercepts)))
        else:
            ratio = typical / max(part.get('relative_inverse_depth') or typical, .001)
            depth = distance * np.clip(ratio, .7, 1.4)
            level = int(np.rint(eye[axis] + np.median(rays[mask, axis]) * depth))
        if select_level is not None:
            level = select_level(part, mask, projection, eye, rays, axis, level, distance)
        selections.append((index, axis, level))
    if reconcile is not None:
        selections = reconcile(selections)
    for index, axis, level in selections:
        mask, cells = masks[index], candidates[index]
        depth = np.divide(level - eye[axis], rays[:, :, axis], out=np.zeros(shape), where=abs(rays[:, :, axis]) > 1e-7)
        inside = mask & (depth > 1.5) & (depth < 150)
        if inside.sum() < MINIMUM_SAMPLES:
            continue
        coords = np.floor(eye + depth[inside, None] * rays[inside]).astype(int)
        sign = 1 if eye[axis] > level else -1
        coords[:, axis] = level - int(sign > 0)
        coords, counts = np.unique(coords[coords[:, 1] >= 0], axis=0, return_counts=True)
        for cell in coords[counts >= MINIMUM_SAMPLES]:
            hit = _sample_face(projection, cell, axis, sign, shape)
            if hit is None:
                continue
            box, pixels, _, _, _ = hit
            supported = np.count_nonzero(pixels & mask[box])
            if supported >= max(MINIMUM_SAMPLES, MINIMUM_COVERAGE * pixels.sum()):
                cells.add(tuple(cell.tolist()))
            elif partial is not None and supported >= max(MINIMUM_SAMPLES, .22 * pixels.sum()):
                if partial(mask, projection, cell, axis, sign):
                    cells.add(tuple(cell.tolist()))
    return candidates


def _charts(image, version, parts, counts):
    saved = {row['id']: row for row in counts['surfaces']} if counts is not None else {}
    charts = []
    for part in parts:
        row = saved.get(part['id'])
        if row is None:
            sample = sample_surface(image, part)
            quad = sample['quad'] if sample is not None else None
            spans = count(sample, version=version)['counts']
        else:
            quad, spans = row.get('quad'), row['counts']
        charts.append(dict(id=part['id'], quad=quad, pitch=dict(counts=spans if None not in spans else None)))
    return charts


def place(image, version, parts, *, counts=None, atlas_path=None, joint_foreground=True, camera=None):
    """Return local (x, y, z) block coordinates and camera/placement diagnostics."""
    from engine.atlas_plane_depth import place_ranked
    image = image.convert('RGB')
    if camera is None:
        camera = seam_camera(image, version, atlas_path=atlas_path)
    return place_ranked(image, version, parts, counts, camera, atlas_path=atlas_path, joint_foreground=joint_foreground)


def place_with_camera(image, version, parts, counts, camera, select_level=None, *,
                      partial=None, refine_candidates=None, remove_occluded=False, walls_first=False, recover_edges=None,
                      reconcile=None, strict_output=None):
    status = dict(camera_status=camera['scale_status'], reason=camera['reason'], P=camera['P'],
                  strict_blocks=0, added_blocks=0, placed_blocks=0, duplicate_side_cells=0,
                  textured_side_cells=0, grounded_side_cells=0, shared_side_cells=0)
    status['camera_source'] = camera.get('camera_source')
    if camera.get('atlas_reason'):
        status['atlas_camera_reason'] = camera['atlas_reason']
    if camera['scale_status'] != 'supported' or camera['P'] is None or not parts:
        return set(), status
    projection = np.asarray(camera['P'])
    # Count sample centers strictly inside the original frame, including odd-sized images.
    shape = tuple((n + (SAMPLE_STRIDE - 1) // 2) // SAMPLE_STRIDE for n in image.size[::-1])
    masks = [observed_support(part['support'], shape, image.size) if part.get('support')
             else np.zeros(shape, dtype=bool) for part in parts]
    charts = _charts(image, version, parts, counts)
    carriers = _carriers(image, charts, masks, projection)
    if version == '26.3' and reconcile is not None:
        from engine.counted_ground_corner import grounded_pairs
        anchored = grounded_pairs(image, charts, parts, masks, projection)
        by_mask = {id(masks[i]): plane for i, plane in anchored.items()}
        carriers = [(m, by_mask.get(id(m), plane) if plane is not None else None, reason)
                    for m, plane, reason in carriers]
    strict, _, _ = _blocks(masks, carriers, projection, {})
    projection = projection / np.linalg.norm(projection[2, :3])
    eye = np.linalg.solve(projection[:, :3], -projection[:, 3])
    axes = [-1] * len(parts) if partial is not None else None
    candidates = _estimate_plane_cells(image, parts, masks, projection, eye, select_level,
                                       partial=partial, axes=axes, walls_first=walls_first, reconcile=reconcile)
    if refine_candidates is not None:
        status['secondary_depths'] = refine_candidates(parts, masks, projection, eye, candidates)
    cells = sorted(strict.union(*candidates))
    if axes is None:
        owner = _frontmost_cells(cells, projection, eye, shape)
    else:
        owner, front_axes = _frontmost_cells(cells, projection, eye, shape, face_axes=True)
    accepted = strict.copy()
    for index, (mask, evidence) in enumerate(zip(masks, candidates, strict=True)):
        if not evidence:
            continue
        visible = owner[mask if axes is None else mask & (front_axes == axes[index])]
        sample_counts = np.bincount(visible[visible >= 0], minlength=len(cells))
        accepted.update(cells[i] for i in np.flatnonzero(sample_counts >= MINIMUM_SAMPLES) if cells[i] in evidence)
    if accepted - strict:
        from engine.seam_ownership import corroborated_runs, textured_side_owners, resolve_side_owners
        from engine.seam_runs import grounded_side_owners
        textured = textured_side_owners(image, projection, eye, cells, owner, accepted, strict, _sample_face)
        runs = grounded_side_owners(image, accepted, strict, masks, projection, eye, _sample_face,
                                    _frontmost_cells, textured)
        grounded = corroborated_runs(runs, textured)
        duplicates = resolve_side_owners(textured, grounded, strict)
        accepted -= duplicates
        status.update(duplicate_side_cells=len(duplicates), textured_side_cells=len(duplicates & textured.keys()),
                      grounded_side_cells=len(duplicates & grounded.keys()),
                      shared_side_cells=len(duplicates & textured.keys() & grounded.keys()))
    if remove_occluded:
        from engine.visible_ray_filter import fully_occluded_cells
        occluded = fully_occluded_cells(image.size, camera['P'], accepted)
        accepted -= occluded
        status['fully_occluded_blocks'] = len(occluded)
    if recover_edges is not None:
        added = recover_edges(accepted)
        if added:
            accepted.update(added)
            if remove_occluded:
                occluded = fully_occluded_cells(image.size, camera['P'], accepted)
                accepted -= occluded
                status['fully_occluded_blocks'] += len(occluded)
        status['verified_edge_blocks'] = len(added & accepted)
    if strict_output is not None:
        strict_output.update(strict & accepted)
    status.update(strict_blocks=len(strict & accepted), added_blocks=len(accepted - strict), placed_blocks=len(accepted))
    return accepted, status
