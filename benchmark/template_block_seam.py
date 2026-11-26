"""Register visible, rectified RGB to repeated vanilla sprites to locate missing grid axes."""
from functools import lru_cache
from itertools import product

import numpy as np

from engine.atlas_count import _shortlist, load_atlas


SETTINGS = dict(color_candidates=24, retained_pitches=10, minimum_pitch_px=8.,
                max_count=24., coarse_count_step=.20, max_pixels=2400,
                minimum_correlation=.65, ambiguous_correlation_gap=.04,
                max_seam_difference=.20)
ATLAS_SHA256 = '49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950'


@lru_cache(maxsize=1)
def _atlas():
    atlas = load_atlas(version='26.3')
    if atlas.provenance['atlas_sha256'] != ATLAS_SHA256:
        raise ValueError('The registration experiment requires the frozen 26.3 atlas.')
    return atlas


def _pairs(baseline, size):
    """Use texture peaks and scan chart spans compatible with a square source block."""
    current = baseline['counts']
    if None not in current:
        return [tuple(current)]
    axis_peaks = []
    evidence = baseline.get('evidence', {})
    aspect = evidence.get('aspect')
    for n, peaks in zip(current, evidence.get('alternatives', [[], []])):
        best = sorted(peaks, key=lambda p: p['agreement'] + p['native_support'], reverse=True)
        axis_peaks.append([n] if n is not None else [p['count'] for p in best[:5]])
    pairs = list(product(*axis_peaks))
    if aspect and aspect > 0:
        for axis in (0, 1):
            other = 1 - axis
            if current[other] is not None:
                continue
            length = size[axis]
            upper = min(SETTINGS['max_count'], length / SETTINGS['minimum_pitch_px'])
            anchors = axis_peaks[axis]
            if current[axis] is None:
                # Scan on the longer axis to bound accumulated phase error
                # after refining the best pitch. Short axes still contribute repeat peaks.
                if (aspect >= 1) == (axis == 0):
                    anchors = [*anchors, *np.arange(.8, upper+.01, SETTINGS['coarse_count_step'])]
            for anchor in anchors:
                partner = anchor / aspect if axis == 0 else anchor * aspect
                for ratio in (.94, 1., 1.06):
                    pair = (anchor, partner * ratio)
                    pairs.append(pair if axis == 0 else pair[::-1])
    return list(dict.fromkeys((float(u), float(v)) for u,v in pairs if u > 0 and v > 0
                              and size[0] / u >= 5 and size[1] / v >= 5))


def _bins(coords, counts):
    ij = np.floor(coords * counts * 16).astype(np.int32) % 16
    return 16 * ij[:,1] + ij[:,0]


def _folded(coords, values, counts):
    cells = _bins(coords, counts)
    weights = np.bincount(cells, minlength=256)
    sums = np.bincount(cells, weights=values, minlength=256)
    return weights, sums


def _short_pitches(pairs, coords, values):
    """Prefer pitches that explain repeated observations at the same atlas texel."""
    if len(pairs) <= SETTINGS['retained_pitches']:
        return pairs
    merits = []
    for pair in pairs:
        weights, sums = _folded(coords, values, pair)
        used = weights > 0
        # Penalize sparse bins: a period that assigns each pixel its own bin
        # cannot demonstrate repetition.
        explained = np.sum(sums[used]**2 / weights[used])
        bias = used.sum() / len(values)
        merits.append((explained / len(values) - bias) / max(.01, 1-bias))
    selected = []
    for index in np.argsort(merits)[::-1]:
        pair = pairs[index]
        if all(np.max(abs(np.subtract(pair,q))) > .09 for q in selected):
            selected.append(pair)
        if len(selected) == SETTINGS['retained_pitches']:
            break
    return selected


def _correlations(pair, coords, values, conjugates, squares):
    weights, sums = _folded(coords, values, pair)
    occupancy = np.fft.rfft2(weights.reshape(16,16))
    field = np.fft.rfft2(sums.reshape(16,16))

    def inverse(data):
        return np.fft.irfft2(data, s=(16,16), axes=(-2,-1))

    cross = inverse(field * conjugates)
    mean = inverse(occupancy * conjugates)
    variance = inverse(occupancy * squares) - mean**2 / len(values)
    return cross / np.sqrt(np.maximum(1., variance) * len(values))


def register(sample, baseline, atlas):
    counts = list(baseline['counts'])
    phase = list(baseline['phase'])
    output = dict(counts=counts, phase=phase)
    if sample is None or all(p is not None for p in phase):
        return output
    mask = sample['mask']
    rgb = sample['rgb']
    rows, cols = np.nonzero(mask)
    if len(rows) < 110 or min(np.ptp(cols), np.ptp(rows)) < 5:
        return output
    size = mask.shape[::-1]
    pairs = _pairs(baseline, size)
    if not pairs:
        return output
    if len(rows) > SETTINGS['max_pixels']:
        # Uniform random sampling avoids regular strides that alias periodic artwork.
        chosen = np.random.default_rng(47).choice(len(rows), SETTINGS['max_pixels'], replace=False)
        rows, cols = rows[chosen], cols[chosen]
    coords = (np.column_stack((cols,rows))+.5) / size
    gray = rgb[rows,cols].mean(axis=1)
    if np.std(gray) < 1.8:
        return output
    values = (gray - gray.mean()) / gray.std()
    shortlist, _ = _shortlist(rgb, mask, atlas)
    indices = shortlist[:SETTINGS['color_candidates']]
    conjugates = atlas.spectra[indices].conj()
    squares = np.fft.rfft2(atlas.patterns[indices]**2).conj()
    if None in counts:
        # A small piece of a sprite can resemble a lighting gradient. Compare
        # novel spacings only if a whole cycle on each axis and two cycles
        # on one axis could actually be observed in this mask.
        extent = np.ptp(coords, axis=0)
        pairs = [pair for pair in pairs if min(extent * pair) >= .90 and max(extent * pair) >= 1.9]
    candidates = _short_pitches(pairs, coords, values)
    fits = []

    def record(pair, template=None):
        spec = conjugates if template is None else conjugates[template:template+1]
        sq = squares if template is None else squares[template:template+1]
        scores = _correlations(pair, coords, values, spec, sq)
        best = float(scores.max())
        if best < SETTINGS['minimum_correlation'] - .10:
            return
        # Keep the competing *geometry*, including alignments on other
        # materials or mirrored versions with virtually equal correlations.
        near = np.argwhere(scores >= best - SETTINGS['ambiguous_correlation_gap'])
        for i, rot, y, x in near:
            fits.append((float(scores[i,rot,y,x]), pair,
                         (x / 16., y / 16.), int(i if template is None else template)))

    for pair in candidates:
        record(pair)
    if not fits:
        return output
    _, coarse, _, template = max(fits)
    offsets = [[0.] if n is not None else [-.12, -.06, 0., .06, .12] for n in counts]
    for du, dv in product(*offsets):
        if du or dv:
            pair = (coarse[0]+du, coarse[1]+dv)
            if min(pair) > 0:
                record(pair, template)
    best, pitch, origin, _ = max(fits)
    if best < SETTINGS['minimum_correlation']:
        return output
    if None in counts and best < .80:
        return output
    output['correlation'] = best
    rivals = [r for r in fits if r[0] >= best - SETTINGS['ambiguous_correlation_gap']]
    for axis in (0, 1):
        if phase[axis] is not None:
            continue
        if max(abs(r[1][axis]-pitch[axis]) for r in rivals) > .27:
            continue
        counts[axis] = float(pitch[axis])
        for _, other_pitch, other_origin, _ in rivals:
            # Grids must agree at both ends of the chart after wrapping the
            # arbitrary integer at the origin once.
            delta = (origin[axis] - other_origin[axis] + .5) % 1. - .5
            if max(abs(delta), abs(delta - pitch[axis] + other_pitch[axis])) > SETTINGS['max_seam_difference']:
                break
        else:
            phase[axis] = float(origin[axis])
    return output


def predict(sample, baseline):
    if sample is None or None not in baseline['phase']:
        return dict(counts=list(baseline['counts']), phase=list(baseline['phase']))
    return register(sample, baseline, _atlas())
