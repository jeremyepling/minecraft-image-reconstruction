"""Locate missing grid offsets in a visible surface at its measured block spacing."""
from math import isfinite

import numpy as np

from engine.atlas_count import _shortlist


ATLAS_SHA256 = '49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950'
MAX_PIXELS = 2400
COLOR_CANDIDATES = 24
MIN_CORRELATION = .65
RIVAL_GAP = .04
MAX_PHASE_DISTANCE = .20  # Fraction of one block, accounting for wraparound.


def _offset_scores(coords, values, counts, atlas, candidates):
    """Correlate every sprite orientation and offset using only observed pixels."""
    texels = np.floor(coords * counts * 16).astype(np.int32) % 16
    bins = 16 * texels[:, 1] + texels[:, 0]
    weights = np.bincount(bins, minlength=256).reshape(16, 16)
    sums = np.bincount(bins, weights=values, minlength=256).reshape(16, 16)
    occupancy = np.fft.rfft2(weights)
    templates = atlas.spectra[candidates].conj()
    squares = np.fft.rfft2(atlas.patterns[candidates] ** 2).conj()

    def inverse(data):
        return np.fft.irfft2(data, s=(16, 16), axes=(-2, -1))

    cross = inverse(np.fft.rfft2(sums) * templates)
    means = inverse(occupancy * templates)
    variance = inverse(occupancy * squares) - means ** 2 / len(values)
    return cross / np.sqrt(np.maximum(1., variance) * len(values))


def register_phase(sample, result, atlas):
    """Fill only missing offsets whose near-best texture fits agree at the measured spacing."""
    counts = result['counts']
    phase = list(result['phase'])
    if (atlas is None or sample is None or None not in phase
            or any(n is None or not isfinite(n) or n <= 0 for n in counts)):
        return result
    if atlas.provenance['atlas_sha256'] != ATLAS_SHA256:
        raise ValueError('Grid offset registration requires the calibrated vanilla Java 26.3 atlas.')

    mask = sample['mask']
    rgb = sample['rgb']
    rows, cols = np.nonzero(mask)
    if len(rows) < 110 or min(np.ptp(cols), np.ptp(rows)) < 5:
        return result
    if len(rows) > MAX_PIXELS:
        # Regular strides can alias repeating artwork.
        chosen = np.random.default_rng(47).choice(len(rows), MAX_PIXELS, replace=False)
        rows, cols = rows[chosen], cols[chosen]
    gray = rgb[rows, cols].mean(axis=1)
    deviation = gray.std()
    if deviation < 1.8:
        return result
    coords = (np.column_stack((cols, rows)) + .5) / mask.shape[::-1]
    shortlist, _ = _shortlist(rgb, mask, atlas)
    scores = _offset_scores(coords, (gray - gray.mean()) / deviation, counts,
                            atlas, shortlist[:COLOR_CANDIDATES])
    best = scores.max()
    if not isfinite(best):
        raise ValueError('Nonfinite atlas correlation during grid offset registration.')
    if best < MIN_CORRELATION:
        return result

    # Frozen tie rule: prefer the largest (horizontal, vertical) offset.
    _, _, best_y, best_x = np.nonzero(scores == best)
    origin = max(zip(best_x, best_y))
    _, _, near_y, near_x = np.nonzero(scores >= best - RIVAL_GAP)
    for axis, rivals in enumerate((near_x, near_y)):
        if phase[axis] is None:
            distance = abs((origin[axis] - rivals + 8) % 16 - 8) / 16.
            if distance.max() <= MAX_PHASE_DISTANCE:
                phase[axis] = float(origin[axis] / 16.)
    return dict(result, phase=phase) if phase != list(result['phase']) else result
