"""Score dimensionless wall width/height for labeled endpoints and automatic charts."""
from collections import defaultdict
from itertools import permutations, product

import numpy as np

from benchmark.camera_metrics import observation, proposal_scores, registration


def _pairs(spans):
    """Join uninterrupted perpendicular runs only at an endpoint on their shared wall."""
    endpoints = defaultdict(lambda: ([], []))
    for normal_axis, direction, a, b in spans:
        if normal_axis == 1:
            continue
        for corner, end in ((a, b), (b, a)):
            horizontal, vertical = endpoints[normal_axis, corner]
            (vertical if direction == 1 else horizontal).append(end)
    for (axis, corner), (horizontal, vertical) in sorted(endpoints.items()):
        for width_end, height_end in product(sorted(horizontal), sorted(vertical)):
            yield axis, corner, width_end, height_end


def _optics(prediction):
    """Recover a centered, square-pixel pinhole from three image-only Manhattan VPs."""
    vps = np.asarray(prediction.get("axes_homogeneous", []), dtype=float)
    size = np.asarray(prediction.get("image_size", []), dtype=float)
    if vps.shape != (3, 3) or size.shape != (2,) or min(size) <= 0:
        return None
    if not np.isfinite(vps).all() or not np.isfinite(size).all():
        return None
    center, unit = size / 2, max(size)
    shifted = vps.copy()
    shifted[:, :2] = (shifted[:, :2] - shifted[:, 2, None] * center) / unit
    lengths = np.linalg.norm(shifted, axis=1)
    if np.any(lengths < 1.e-12):
        return None
    shifted /= lengths[:, None]
    i, j = np.triu_indices(3, 1)
    # For each orthogonal pair (u,v), ux*vx + uy*vy + f²*uw*vw = 0.
    a = (shifted[i, :2] * shifted[j, :2]).sum(1)
    b = shifted[i, 2] * shifted[j, 2]
    focal_sq = -(a @ b) / (b @ b) if b @ b > 1.e-12 else None
    focal = unit * np.sqrt(focal_sq) if focal_sq is not None and focal_sq > 0 else None
    return vps, center, focal


def _measure(optics, points):
    """Use only VPs and the three supplied image endpoints. The plane distance cancels."""
    if optics is None:
        return None, None, "missing_vanishing_axes"
    vps, center, focal = optics
    corner = points[0]
    edges = points[1:] - corner
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) < 1:
        return None, None, "unresolved_projected_span"
    to_vps = vps[:, :2] - vps[:, 2, None] * corner
    directions = to_vps / np.maximum(1.e-12, np.linalg.norm(to_vps, axis=1, keepdims=True))
    agreement = np.abs((edges / lengths[:, None]) @ directions.T)
    # A finite visible edge cannot cross its own vanishing point.
    far_end = vps[None, :, :2] - points[1:, None, :] * vps[None, :, 2, None]
    agreement[(far_end * to_vps).sum(2) <= 0] = 0
    alignments = [(min(agreement[0, i], agreement[1, j]), agreement[0, i] + agreement[1, j], i, j)
                  for i, j in permutations(range(3), 2)]
    quality, _, width_axis, height_axis = max(alignments)
    axes = [width_axis, height_axis]
    if quality < np.cos(np.radians(12)):
        return None, axes, "axes_disagree_with_endpoint_directions"

    selected = vps[axes]
    if focal is None:
        # Two axes at infinity make this wall frontoparallel. Width and height
        # then have identical isotropic magnification at every point on it.
        parallel = abs(selected[:, 2]) * max(center) <= 1.e-8 * np.linalg.norm(selected[:, :2], axis=1)
        if np.all(parallel) and abs(directions[width_axis] @ directions[height_axis]) < 1.e-5:
            return float(lengths[0] / lengths[1]), axes, None
        return None, axes, "focal_length_unresolved"

    camera_dirs = np.column_stack(((selected[:, :2] - selected[:, 2, None] * center) / focal,
                                   selected[:, 2]))
    camera_dirs /= np.linalg.norm(camera_dirs, axis=1, keepdims=True)
    normal = np.cross(*camera_dirs)
    rays = np.column_stack(((points - center) / focal, np.ones(3)))
    depths = rays @ normal
    if np.min(abs(depths)) < 1.e-7 or np.any(depths * depths[0] <= 0):
        return None, axes, "wall_intersection_unresolved"
    # Place the shared corner at unit optical depth. No observer height,
    # block pitch, true camera, or physical length enters this measurement.
    xyz = rays * (depths[0] / depths)[:, None]
    width, height = np.linalg.norm(xyz[1:] - xyz[0], axis=1)
    if min(width, height) < 1.e-10:
        return None, axes, "degenerate_wall_span"
    return float(width / height), axes, None


def _case(expected, predicted, reason, **fields):
    error = 100 * abs(predicted / expected - 1) if predicted is not None else None
    return dict(**fields, expected_ratio=float(expected), predicted_ratio=predicted, abs_error_pct=error,
                reason=reason)


def summarize_cases(cases):
    errors = [case["abs_error_pct"] for case in cases if case["abs_error_pct"] is not None]
    return dict(total=len(cases), measured=len(errors), abstained=len(cases) - len(errors),
                mean_abs_error_pct=float(np.mean(errors)) if errors else None, cases=cases)


def score(prediction, pose, faces):
    """Score labeled-endpoint rectification and separately score the image-only chart extents."""
    vertices, spans, patches = observation(faces)
    optics = _optics(prediction)
    diagnostic = []
    for axis, corner, width_end, height_end in _pairs(spans):
        pixels = np.array([vertices[p] for p in (corner, width_end, height_end)])
        value, vp_indices, reason = _measure(optics, pixels)
        expected = np.linalg.norm(np.subtract(width_end, corner)) / abs(height_end[1] - corner[1])
        diagnostic.append(_case(expected, value, reason, corner_xyz=list(corner), normal_axis=axis,
                                width_end_xyz=list(width_end), height_end_xyz=list(height_end),
                                corner_px=pixels[0].tolist(), width_end_px=pixels[1].tolist(),
                                height_end_px=pixels[2].tolist(), axis_indices=vp_indices))

    # Registration uses evaluator camera metadata only to associate chart axes
    # and origins with their labeled dominant walls. Ratios use predicted spans.
    T = None
    if prediction.get("P") is not None and prediction.get("planes"):
        T = registration(np.asarray(prediction["P"]), pose)
    automatic = []
    for chart in proposal_scores(prediction, T, patches):
        horizontal = 1 if chart["axis"] == 0 else 0
        vertical = 1 - horizontal
        expected = chart["expected"][horizontal] / chart["expected"][vertical]
        value, reason = None, "no_automatic_chart"
        if "predicted" in chart and min(chart["predicted"]) > 0:
            value = chart["predicted"][horizontal] / chart["predicted"][vertical]
            reason = None
        plane_error = chart.get("plane_error_blocks") if prediction.get("scale_status") == "supported" else None
        automatic.append(_case(expected, value, reason, axis=chart["axis"], plane=chart["plane"],
                              plane_error_blocks=plane_error))
    return dict(endpoint_diagnostic=summarize_cases(diagnostic), automatic_charts=summarize_cases(automatic))
