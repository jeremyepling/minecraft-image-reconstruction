"""Fit the four sides of an observed face to two projected Minecraft axes."""
from itertools import permutations

import numpy as np


def fit_face(quad, axes):
    corners = np.asarray(quad, dtype=float)
    frames = np.asarray(axes, dtype=float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if frames.shape != (3, 3) or np.min(lengths) < 4 or not np.isfinite(frames).all():
        return None

    # A vanishing point with w=0 represents parallel lines in the screenshot.
    centers = corners + .5 * edges
    rays = frames[None, :, :2] - centers[:, None, :] * frames[None, :, 2:]
    normals = np.stack((-rays[:, :, 1], rays[:, :, 0]), axis=2)
    ray_lengths = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= np.maximum(1.e-12, ray_lengths)
    # Residual is perpendicular displacement at an endpoint, measured in
    # pixels. Allow a little more angular uncertainty for a short observed edge.
    residual = .5 * abs((edges[:, None] * normals).sum(2))
    tolerance = 1.7 + .5 * lengths * np.sin(np.deg2rad(3.))
    normalized = np.where(ray_lengths[:, :, 0] > 1.e-10, residual/tolerance[:, None], np.inf)
    def error(pair):
        return np.max(normalized[np.arange(4), list(pair)*2])
    a,b = min(permutations(range(3), 2), key=error)
    if error((a,b)) > 1.:
        return None
    # Opposite sides share a direction. Adjacent sides switch direction at
    # the intersection of two image-supported lines through their midpoints.
    ns = normals[np.arange(4), [a,b,a,b]]
    lines = np.column_stack((ns, -(ns*centers).sum(1)))
    vertices = np.cross(np.roll(lines, 1, axis=0), lines)
    if np.min(abs(vertices[:, 2])) < 1.e-6:
        return None
    adjusted = vertices[:, :2] / vertices[:, 2, None]
    movement = np.linalg.norm(adjusted-corners, axis=1)
    if (movement > 2. + .055*np.minimum(lengths, np.roll(lengths,1))).any():
        return None
    return adjusted, dict(status='supported', face_axes=[a,b], third_axis=3-a-b,
                          max_corner_shift_px=round(float(movement.max()),2))
