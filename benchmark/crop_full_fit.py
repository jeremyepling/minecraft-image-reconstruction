"""Benchmark-only crop expansion with the baseline plane and mask held fixed."""
import numpy as np

from engine.geometry import UNIT, transform
from engine.surface_perimeter import _native_alpha
from engine.surface_sample import sample_surface


def _project(matrix, points, anchor):
    """Reject a footprint that reaches or crosses the chart's projective horizon."""
    homogeneous = points @ matrix[:, :2].T + matrix[:, 2]
    denominator = homogeneous[:, 2]
    anchor_depth = matrix[2, :2] @ anchor + matrix[2, 2]
    if not np.isfinite(homogeneous).all() or np.any(denominator * np.sign(anchor_depth) <= abs(anchor_depth) * 1.e-6):
        return None
    return homogeneous[:, :2] / denominator[:, None]


def _row_edges(owned, left, top):
    rows = np.flatnonzero(owned.any(axis=1))
    start = left + np.argmax(owned[rows], axis=1)
    stop = left + owned.shape[1] - np.argmax(owned[rows, ::-1], axis=1)
    y = rows + top
    return np.column_stack((np.r_[start, stop, start, stop], np.r_[y, y, y + 1, y + 1]))


def expand_sample(image, part, select_bounds=None):
    """Expand the baseline chart to native mask edges, or custom bounds from UV centers."""
    original = sample_surface(image, part)
    support = part.get('support')
    if original is None or support is None or not original['mask'].any():
        return original
    native = _native_alpha(support, image.size)
    if native is None:
        return original
    alpha, (left, top, _, _) = native
    owned = np.asarray(alpha, dtype=bool)
    if select_bounds is None:
        # Extrema of a linear-fractional map on each scanline are at its ends.
        points = _row_edges(owned, left, top)
    else:
        rows, columns = np.nonzero(owned)
        points = np.column_stack((columns + left + .5, rows + top + .5))
    quad = np.asarray(original['quad'])
    try:
        chart = transform(quad)
        inverse = np.linalg.inv(chart)
    except np.linalg.LinAlgError:
        return original
    uv = _project(inverse, points, quad.mean(axis=0))
    if uv is None:
        return original
    bounds = (uv.min(axis=0), uv.max(axis=0)) if select_bounds is None else select_bounds(uv)
    if bounds is None:
        return original
    low, high = np.minimum(0., bounds[0]), np.maximum(1., bounds[1])
    corners = low + UNIT * (high - low)
    expanded = _project(chart, corners, np.array([.5, .5]))
    if expanded is None or np.max(abs(expanded - quad)) < .5:
        return original
    # A far-off corner offers no extra observed pixels and destroys sampling resolution.
    if np.any(expanded < -np.array(image.size)) or np.any(expanded > 2 * np.array(image.size)):
        return original
    result = sample_surface(image, dict(part, quad=expanded.tolist()))
    if result is None or not result['mask'].any():
        return original
    result['chart_source'] = original['chart_source'] + ' + expanded bounds'
    return result


def sample(image, part):
    return expand_sample(image, part)
