"""Rank observed edge quadrilaterals using one image's predicted surface normals."""
import numpy as np

from engine.geometry import bilinear
from engine.surface_edges import _sample_grid, detect_planes


SETTINGS = dict(interior_tolerance_deg=20., minimum_consistency=.74,
                shoulder_min_px=4., shoulder_max_px=25.,
                boundary_angle_range_deg=[10., 50.], min_confidence=.68,
                rgb_weight=.25, interior_weight=.20, boundary_weight=.55)


def _unit(v):
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=np.isfinite(length) & (length > 1.e-6))


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=-1), -1., 1.)))


def _evidence(quad, base, field, scale):
    def sample(points):
        xy = points * scale
        on_image = np.all((xy >= 0) & (xy < field.shape[1::-1]), axis=-1)
        return _unit(bilinear(field, xy)) * on_image[..., None]

    interior = sample(_sample_grid(quad))
    valid = np.any(interior, axis=2)
    if valid.mean() < .75:
        return None
    median = _unit(np.median(interior[valid], axis=0))
    if not median.any():
        return None
    aligned = valid & (_angle(interior, median) <= SETTINGS['interior_tolerance_deg'])
    consistency = float(aligned.mean())
    if consistency < SETTINGS['minimum_consistency']:
        return None

    # Look on both sides of each edge. Sampling the same angular change on
    # two spatial scales admits softened corners and narrow side returns.
    edges = np.roll(quad, -1, axis=0)-quad
    length = np.linalg.norm(edges, axis=1)
    inward = np.stack([-edges[:, 1], edges[:, 0]], axis=1) / length[:, None]
    # Distance to the opposite corner limits the inward shoulder on narrow
    # faces, including roofs that project into thin trapezoids.
    depth = ((np.roll(quad, -2, axis=0)-quad)*inward).sum(1)
    spread = np.clip(.12*depth, SETTINGS['shoulder_min_px'], SETTINGS['shoulder_max_px'])
    t = np.linspace(.12, .88, 15)
    edge_samples = quad[:, None, :] + t[None, :, None]*edges[:, None, :]
    votes = np.zeros((4, len(t)))
    for distance in (.6, 1.5):
        inside_step = inward * np.minimum(.33 * depth, distance * spread)[:, None]
        outside_step = inward * (distance * spread)[:, None]
        inside = sample(edge_samples + inside_step[:, None, :])
        outside = sample(edge_samples - outside_step[:, None, :])
        turn = np.minimum(_angle(inside, outside), _angle(median, outside))
        low, high = SETTINGS['boundary_angle_range_deg']
        change = np.clip((turn-low)/(high-low), 0., 1.)
        good = np.any(inside, axis=2) & np.any(outside, axis=2) & (_angle(median, inside) < 35.)
        votes = np.maximum(votes, np.where(good, change, 0.))
    sides = votes.mean(1)
    # A single strong silhouette cannot establish a patch. The two middle
    # sides tolerate one junction with little visible normal change.
    boundary = float(np.sort(sides)[1:3].mean())
    confidence = (SETTINGS['rgb_weight'] * base + SETTINGS['interior_weight'] * consistency
                  + SETTINGS['boundary_weight'] * boundary)
    details = dict(interior_normal_agreement=round(consistency, 4),
                   rgb_confidence=round(float(base), 4),
                   side_orientation_support=sides.round(4).tolist(),
                   median_normal_camera=median.round(5).tolist())
    return float(confidence), details


def detect(image, normals, version='26.3'):
    """Return observed patches in native screenshot coordinates; use no camera or labels."""
    field = np.asarray(normals, np.float32)
    if field.ndim != 3 or field.shape[2] != 3 or min(field.shape[:2]) < 2:
        raise ValueError('Expected a height × width × 3 normal field for this screenshot')
    scale = np.array(field.shape[1::-1])/image.size

    def rank(quad, base):
        result = _evidence(quad, base, field, scale)
        return result if result is not None and result[0] >= SETTINGS['min_confidence'] else None

    return detect_planes(image, version, shading=True, rank_candidate=rank)
