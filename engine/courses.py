"""Find running-bond mortar on occupied, rectified parts of a single plane."""
import numpy as np

from engine.geometry import bilinear, box_mean, project


def _rectangle(valid, i, j, x0, y0):
    """Extend a contiguous column, then extend adjacent columns over the same rows."""
    rows, cols = valid.shape
    left, right, top, bottom = i, i + 1, j, j + 1
    while top > 0 and valid[top - 1, i]:
        top -= 1
    while bottom < rows and valid[bottom, i]:
        bottom += 1
    while left > 0 and valid[top:bottom, left - 1].all():
        left -= 1
    while right < cols and valid[top:bottom, right].all():
        right += 1
    return x0 + left, y0 + top, x0 + right, y0 + bottom


def _color_region(colors, occupied, i, j):
    rows, cols = occupied.shape
    x0, x1 = max(0, i - 2), min(cols, i + 3)
    y0, y1 = max(0, j - 3), min(rows, j + 4)
    delta = np.abs(colors[y0:y1, x0:x1] - colors[j, i])
    valid = occupied[y0:y1, x0:x1] & (delta[..., 0] <= 31) & (delta[..., 1:].max(2) <= 16)
    return _rectangle(valid, i - x0, j - y0, x0, y0)


def _correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))


def _course_lines(light, cell_height):
    """Locate the dominant repeat and dark horizontal line after removing a local gradient."""
    profile = light.mean(axis=1)
    profile -= box_mean(profile, 0, max(3, cell_height // 2))
    if profile.std() < 2:
        return None

    # Require four visible courses. Prefer the fundamental to its harmonics.
    limit = min(len(profile) // 4, round(cell_height * 1.5))
    first = max(3, round(cell_height * .23))
    shifts = np.arange(first - 1, limit + 2)
    if len(shifts) < 3:
        return None
    corrs = np.array([_correlation(profile[:-p], profile[p:]) for p in shifts])
    peaks = np.flatnonzero((corrs[1:-1] >= corrs[:-2]) & (corrs[1:-1] >= corrs[2:])) + 1
    if not len(peaks) or corrs[peaks].max() < .49:
        return None
    minimum = max(.49, corrs[peaks].max() - .13)
    index = next(k for k in peaks if corrs[k] >= minimum)
    before, center, after = corrs[index - 1:index + 2]
    curvature = min(-1e-6, 2 * (before - 2 * center + after))
    pitch = float(shifts[index] + np.clip((before - after) / curvature, -.5, .5))

    phase_bins = round(pitch)
    phases = (np.arange(len(profile)) % pitch) * phase_bins / pitch
    bins = np.rint(phases).astype(int) % phase_bins
    counts = np.bincount(bins, minlength=phase_bins)
    folded = np.bincount(bins, weights=profile, minlength=phase_bins) / np.maximum(1, counts)
    phase = folded.argmin() * pitch / phase_bins
    return profile, pitch, phase, float(center)


def _cross_joints(light, pitch, phase, cell_width):
    """Compare vertical ridges inside alternate courses, preserving their horizontal positions."""
    radius = max(1, round(cell_width / 8))
    if light.shape[1] < 2 * radius + 12:
        return None
    sides = (light[:, :-2*radius] + light[:, 2*radius:]) * .5
    ridge = sides - light[:, radius:-radius]
    profiles = []
    for start in np.arange(phase - pitch, len(light), pitch):
        lo, hi = int(np.ceil(start + pitch * .22)), int(np.ceil(start + pitch * .8))
        if lo >= 0 and hi <= len(light) and hi - lo >= 2:
            profiles.append(ridge[lo:hi].mean(axis=0))
    if len(profiles) < 4:
        return None
    profiles = np.array(profiles)
    if np.median(np.ptp(profiles, axis=1)) < 5:
        return None
    repeated = np.median([_correlation(a, b) for a, b in zip(profiles[:-2], profiles[2:])])
    if repeated < .35:
        return None

    even, odd = profiles[::2].mean(0), profiles[1::2].mean(0)
    width = len(even)
    limit = min(width - 8, int(width * .65), cell_width * 2)
    shifts = range(max(2, round(cell_width * .3)), limit + 1)
    # A square grid or a vertical panel joint already agrees at zero shift.
    scores = [max(_correlation(even[:-s], odd[s:]), _correlation(even[s:], odd[:-s])) for s in shifts]
    if not scores:
        return None
    best = max(scores)
    contrast = best - _correlation(even, odd)
    if best < .52 or contrast < .35:
        return None
    half_width = next(s for s, score in zip(shifts, scores) if score >= best - .055)
    return half_width * 2, float(min(repeated, best)), float(contrast)


def _aligned_cells(profiles, reference, threshold):
    """Test every cell for the common mortar phase, including cells beside flush trim."""
    profiles = profiles - profiles.mean(axis=1, keepdims=True)
    reference = reference - reference.mean(axis=1, keepdims=True)
    rms, target_rms = profiles.std(axis=1), reference.std(axis=1)
    corr = (profiles * reference).mean(axis=1) / np.maximum(1e-6, rms * target_rms)
    return (rms >= np.maximum(1.8, .3 * target_rms)) & (corr >= threshold)


def _coherent_regions(light, target_rows, supports, sx, sy):
    """Split color-consistent regions at texture changes before testing for stagger."""
    regions = set()
    for left, top, right, bottom in supports:
        patch = light[top*sy:bottom*sy, left*sx:right*sx]
        lines = _course_lines(patch, sy)
        if lines:
            reference = lines[0].reshape(bottom - top, sy, 1)
            valid = _aligned_cells(target_rows[top:bottom, :, left:right], reference, .55)
            for j, i in zip(*np.where(valid)):
                regions.add(_rectangle(valid, i, j, left, top))
    return sorted(regions)


def _scan(rgb, transform, ids, x0, y0, sx, sy):
    rows, cols = ids.shape
    yy, xx = np.mgrid[:rows*sy, :cols*sx]
    grid = np.stack([x0 + (xx + .5) / sx, y0 + (yy + .5) / sy], axis=2)
    rect = bilinear(rgb, project(transform, grid)).astype(np.float32)
    med = np.median(rect.reshape(rows, sy, cols, sx, 3), axis=(1, 3))
    luma = np.array([.213, .715, .072], dtype=np.float32)
    colors = np.stack([med @ luma, med[..., 0] - med[..., 1], med[..., 2] - med[..., 1]], axis=-1)
    light = rect @ luma
    occupied = ids >= 0
    supports = {_color_region(colors, occupied, i, j) for j, i in zip(*np.where(occupied))}
    target_rows = light.reshape(rows, sy, cols, sx).mean(axis=3)
    target_rows -= box_mean(target_rows, 1, max(3, sy // 2))
    found = {}
    for left, top, right, bottom in _coherent_regions(light, target_rows, supports, sx, sy):
        patch = light[top*sy:bottom*sy, left*sx:right*sx]
        lines = _course_lines(patch, sy)
        if not lines:
            continue
        row, pitch, phase, course_corr = lines
        joints = _cross_joints(patch, pitch, phase, sx)
        if not joints:
            continue
        width, repeat, contrast = joints
        evidence = dict(
            confidence=round(min(course_corr, repeat), 4),
            course_height_cells=round(pitch / sy, 4),
            brick_width_cells=round(width / sx, 4),
            course_correlation=round(course_corr, 4),
            stagger_contrast=round(contrast, 4),
            patch_uv=[x0 + int(left), y0 + int(top), x0 + int(right), y0 + int(bottom)],
        )
        reference = row.reshape(bottom - top, sy, 1)
        local = _aligned_cells(target_rows[top:bottom, :, left:right], reference, .6)
        region_ids = ids[top:bottom, left:right]
        for k in region_ids[local & (region_ids >= 0)].tolist():
            if k not in found or found[k]['confidence'] < evidence['confidence']:
                found[k] = evidence
    return found


def course_evidence(rgb, plane, cells):
    """Return local cell indices with repeated course lines and staggered cross-joints.

    patch_uv is an occupied rectangle in plane grid coordinates. Course heights
    and full brick widths use grid units along their respective plane axes.
    """
    if len(cells) < 4:
        return {}
    cols, rows = plane['columns'], plane['rows']
    quad = np.asarray(plane['quad'])
    widths = np.linalg.norm(quad[[1, 2]] - quad[[0, 3]], axis=1)
    heights = np.linalg.norm(quad[[2, 3]] - quad[[1, 0]], axis=1)
    xscale, yscale = widths.mean() / cols, heights.mean() / rows
    if xscale < 3 or yscale < 4 or rows < 2:
        return {}
    sx, sy = (min(32, round(scale)) for scale in (xscale, yscale))
    transform = np.asarray(plane['grid_to_image'])
    ids = np.full((rows, cols), -1, dtype=np.int32)
    facing = plane.get('facing')
    normal_axis = {'east': 0, 'west': 0, 'up': 1, 'down': 1, 'north': 2, 'south': 2}.get(facing)
    origin = plane.get('origin')
    for k, cell in enumerate(cells):
        # Depth refinement can reuse a facade's ij for a return, a recessed
        # window or a parallel setback. Its original transform fits none of those.
        if cell.get('depth_surface') or (facing and cell.get('facing', facing) != facing):
            continue
        if normal_axis is not None and origin is not None and 'xyz' in cell:
            if cell['xyz'][normal_axis] != origin[normal_axis]:
                continue
        i, j = cell['ij']
        ids[j, i] = k

    # Two overlapping local neighborhoods span at most four columns and six
    # rows beyond a target. The halo produces the same interior result at a
    # chunk boundary while keeping rectification memory bounded on tall walls.
    result = {}
    for y in range(0, rows, 24):
        for x in range(0, cols, 24):
            if not (ids[y:y+24, x:x+24] >= 0).any():
                continue
            x0, y0 = max(0, x - 4), max(0, y - 6)
            chunk = ids[y0:y+30, x0:x+28]
            for k, match in _scan(rgb, transform, chunk, x0, y0, sx, sy).items():
                i, j = cells[k]['ij']
                if x <= i < x + 24 and y <= j < y + 24:
                    result[k] = match
    return result
