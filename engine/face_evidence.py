"""Visible face thickness in original screenshot pixels, before rectification."""
import numpy as np

MIN_FACE_WIDTH = 5.0


def effective_width(quad):
    """Projected area divided by the longer mean pair of opposite edges."""
    points = np.asarray(quad, dtype=float)
    points = points - points[0]
    following = np.roll(points, -1, axis=0)
    twice_area = abs(np.sum(points[:, 0]*following[:, 1] - points[:, 1]*following[:, 0]))
    a, b, c, d = np.linalg.norm(following - points, axis=1)
    edge_sum = max(a+c, b+d)
    return float(twice_area / edge_sum) if edge_sum else 0.0
