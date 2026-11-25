"""Measure block pitch on a rectified full-block surface using default 16-texel art."""
import numpy as np

from engine.geometry import UNIT, bilinear, project, transform


SETTINGS = dict(texels_per_block=16, scanlines=40, oversampling=2.5, maximum_samples=3072,
                minimum_native_period_px=1.65, minimum_color_step=3.,
                minimum_effective_texel_bins=9., phase_bins=96)


def _samples(rgb, h, axis, length):
    n = int(np.clip(np.ceil(length * SETTINGS["oversampling"]), 48, SETTINGS["maximum_samples"]))
    t = (np.arange(n) + .5) / n
    cross = (np.arange(SETTINGS["scanlines"]) + .5) / SETTINGS["scanlines"]
    uv = np.stack(np.meshgrid(t, cross), axis=-1)
    if axis:
        uv = uv[:, :, ::-1]
    xy = project(h, uv)
    colors = bilinear(rgb, xy)
    valid = ((xy[..., 0] >= .5) & (xy[..., 0] <= rgb.shape[1] - .5)
             & (xy[..., 1] >= .5) & (xy[..., 1] <= rgb.shape[0] - .5))
    # Exclude the bilinear footprint of boundary pixels so surrounding floor,
    # sky or a neighboring wall cannot provide extra texture evidence.
    quad = project(h, UNIT)
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        edge = b - a
        distance = abs(edge[0] * (xy[..., 1] - a[1]) - edge[1] * (xy[..., 0] - a[0]))
        valid &= distance > 1.42 * np.linalg.norm(edge)
    colors[~valid] = 0
    return colors, valid


def _agreement(detail, valid, lag):
    if not 0 < lag < .8 * detail.shape[1]:
        return None
    a, b = detail[:, lag:], detail[:, :-lag]
    ok = valid[:, lag:] & valid[:, :-lag]
    if ok.mean() <= .4:
        return None
    n = np.maximum(1, ok.sum(1))[:, None, None]
    weight = ok[:, :, None]
    a = (a - (a * weight).sum(1, keepdims=True) / n) * weight
    b = (b - (b * weight).sum(1, keepdims=True) / n) * weight
    numerator = (a * b).sum((1, 2))
    denominator = np.sqrt((a**2).sum((1, 2)) * (b**2).sum((1, 2)))
    return float(np.median(numerator / np.maximum(1., denominator)))


def _axis(colors, valid, length):
    """Find the native pixel step, requiring activity on many of the 16 texel boundaries."""
    n = colors.shape[1]
    # A two-source-pixel difference measures usable contrast even in oversampled input.
    probe = min(n // 6, max(2, round(n / length * 2)))
    contrast = np.abs(colors[:, probe:] - colors[:, :-probe]).mean(2)
    contrast = np.where(valid[:, probe:] & valid[:, :-probe], contrast, 0.)
    lines = np.quantile(contrast, .90, axis=1) >= SETTINGS["minimum_color_step"]
    if lines.sum() < 8:
        return dict(count=None, reason="insufficient surface contrast")
    colors, valid = colors[lines], valid[lines]
    dx = np.abs(np.diff(colors, axis=1)).mean(2)
    dx = np.where(valid[:, :-1] & valid[:, 1:], dx, 0.)
    # Limit a narrow mortar edge or an aperture's border to the same weight as
    # several texture transitions. Pool rows before looking for a shared texel phase.
    cap = np.maximum(.35, np.quantile(dx, .86, axis=1, keepdims=True))
    dx = np.minimum(dx, cap) / cap
    edge = dx.mean(0)
    taper = np.hanning(n+1)[1:-1]
    offsets = np.arange(1, n) / n
    centered = (edge - np.average(edge, weights=taper)) * taper
    fft_length = 8 * n
    spec = np.fft.rfft(centered, n=fft_length)
    freq = np.fft.rfftfreq(fft_length, d=1/n)
    amp = abs(spec) / max(1.e-9, (edge*taper).sum())
    peaks = np.flatnonzero((amp[1:-1] >= amp[:-2]) & (amp[1:-1] > amp[2:])) + 1
    peaks = peaks[(freq[peaks] >= 12) & (freq[peaks] <= length / SETTINGS["minimum_native_period_px"])]
    peaks = sorted(peaks, key=lambda k: -amp[k])[:24]
    # Removing the row-wide brightness profile isolates texture shared between
    # blocks from a repeated mortar stripe or gradual lighting across the plane.
    detail = colors - np.median(colors, axis=0, keepdims=True)
    proposals = []
    for k in peaks:
        f = freq[k]
        if amp[k] < .055:
            break
        # Offset by one sample because the derivative lives between pixel centers.
        native_phase = (f / n - np.angle(spec[k]) / (2*np.pi)) % 1.
        cycles = f * offsets - native_phase
        indices = np.rint(cycles).astype(int)
        near = np.abs(cycles - indices) < .22
        far = np.abs(cycles - indices) > .34
        background = np.average(edge[far], weights=taper[far])
        energy = np.maximum(edge - background, 0.) * taper
        bins = np.bincount(indices[near] % 16, energy[near], minlength=16)
        effective = bins.sum()**2 / max(1.e-12, bins @ bins)
        if effective < 6:
            continue
        # A native texel grid has a single concentration per cycle. Proposals
        # at half the real frequency mix transitions half a cycle out of phase.
        alignment = np.average(edge[near], weights=taper[near]) / max(.01, background)
        if alignment < 1.25:
            continue
        lag = 16 * n / f
        repeat = _agreement(detail, valid, round(lag))
        subperiods = [_agreement(detail, valid, round(lag/d)) for d in (2, 3, 4, 6, 8)]
        # If precisely the same detailed motif recurs well inside a proposed
        # block, coarse trim and masonry courses can counterfeit a 16-texel grid.
        if any(r is not None and r > .80 and (repeat is None or r >= repeat - .07) for r in subperiods):
            continue
        if effective < SETTINGS["minimum_effective_texel_bins"] and (repeat or 0.) < .65:
            continue
        if repeat is not None and repeat < .05:
            continue
        score = float(amp[k] * min(1., effective/13) * (1 + .45 * max(0., repeat or 0.)))
        proposals.append(dict(count=float(f / 16), native_phase=float(native_phase), strength=score,
                              aligned_energy=float(alignment), effective_bins=float(effective), repeat=repeat))
    if not proposals:
        return dict(count=None, reason="no resolved native texel lattice")
    proposals.sort(key=lambda p: -p["strength"])
    best = proposals[0]
    return dict(count=best["count"], candidates=proposals[:8], reason="native 16-texel lattice")


def _aspect(H, size):
    """Width/height of a rectangular patch from its two orthogonal vanishing directions."""
    w, h = size
    vectors = H[:, :2].T.copy()
    vectors[:, :2] -= vectors[:, 2, None] * [.5*w, .5*h]
    product = vectors[0, 2] * vectors[1, 2]
    slopes = abs(vectors[:, 2]) * h / np.maximum(1.e-12, np.linalg.norm(vectors[:, :2], axis=1))
    if max(slopes) < 1.e-5:
        return float(np.linalg.norm(vectors[0, :2]) / np.linalg.norm(vectors[1, :2]))
    if min(slopes) < .008:
        return None
    f2 = -(vectors[0, :2] @ vectors[1, :2]) / product
    if not (.3*h)**2 < f2 < (3*h)**2:
        return None
    vectors[:, 2] *= np.sqrt(f2)
    return float(np.linalg.norm(vectors[0]) / np.linalg.norm(vectors[1]))


def _select(axes, aspect):
    candidates = [a.get("candidates", []) for a in axes]
    if aspect:
        pairs = [(min(a["strength"], b["strength"]) + .3 * max(a["strength"], b["strength"]), a, b)
                 for a in candidates[0] for b in candidates[1]
                 if abs(np.log(a["count"]/b["count"]/aspect)) < .075 and max(a["strength"], b["strength"]) > .13]
        pairs.sort(key=lambda row: -row[0])
        if pairs and (len(pairs) == 1 or pairs[1][0] < .85 * pairs[0][0]):
            return list(pairs[0][1:])
        if pairs:
            return None
        # The spacing measured on the better-resolved axis transfers to the
        # other via the independently recovered metric aspect of the plane.
        reliable = []
        for i, choices in enumerate(candidates):
            if not choices:
                continue
            best = choices[0]
            if (best["strength"] > .20 and best["effective_bins"] > 10 and
                (best["repeat"] or 0.) > .45 and
                (len(choices) == 1 or choices[1]["strength"] < .7 * best["strength"])):
                reliable.append((i, best))
        if len(reliable) == 1:
            i, best = reliable[0]
            count = best["count"] * (aspect if i == 1 else 1/aspect)
            other = dict(count=count, strength=best["strength"] * .75, source="rectangular plane aspect")
            return [best, other] if i == 0 else [other, best]
        return None
    if all(c and (len(c) == 1 or c[1]["strength"] < .75*c[0]["strength"]) for c in candidates):
        return [c[0] for c in candidates]
    return None


def _phase(colors, valid, count):
    """Look for one coherent dark joint per measured block. Equal courses remain ambiguous."""
    n = colors.shape[1]
    if count < 1.8:
        return None
    # Use a rectified local background to suppress smooth lighting and require
    # a dark center between two brighter shoulders across independent scanlines.
    lag = max(2, round(.15*n/count))
    if 2*lag + 4 >= n:
        return None
    gray = colors.mean(2)
    dark = np.minimum(gray[:, :-2*lag], gray[:, 2*lag:]) - gray[:, lag:-lag]
    ok = valid[:, :-2*lag] & valid[:, 2*lag:] & valid[:, lag:-lag]
    dark = np.where(ok, np.clip(dark, 0., 45.), 0)
    x = (np.arange(lag, n-lag) + .5) / n * count
    phases = np.arange(SETTINGS["phase_bins"]) / SETTINGS["phase_bins"]
    signal = []
    for phase in phases:
        delta = (x - phase + .5) % 1. - .5
        near = abs(delta) < .037
        signal.append(float(np.mean(dark[:, near])) if np.any(near) else 0.)
    signal = np.asarray(signal)
    i = int(signal.argmax())
    peak = signal[i]
    separation = np.abs((phases - phases[i] + .5) % 1. - .5)
    competitor = signal[separation > .15].max()
    if peak < 5. or peak < 1.6 * max(1., competitor) or peak < 2.0 * max(1., np.median(signal)):
        return None
    return float(phases[i])


def fit(image, quad, version="26.3"):
    """Return continuous patch spans in blocks and a grid only when both joint phases resolve.

    A boundary on axis i lies at normalized quad coordinate (phase_blocks[i]+k)/counts[i].
    phase_blocks measures forward from the supplied quad origin; a boundary between dark
    edge texels is estimated at the stripe center. A repeating motif alone has unknown phase.
    """
    if version != "26.3":
        raise ValueError("Surface pitch is configured for default Java 26.3 full-block textures")
    quad = np.asarray(quad, dtype=float)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        raise ValueError("Expected four finite plane corners")
    edges = np.roll(quad, -1, axis=0) - quad
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    lengths = np.linalg.norm(edges, axis=1)
    dimensions = [(lengths[0]+lengths[2])/2, (lengths[1]+lengths[3])/2]
    result = dict(counts=None, grid=None, confidence=0., reason="insufficient plane resolution", evidence={})
    if min(dimensions) < 12 or not (np.all(turns > 0) or np.all(turns < 0)):
        return result
    try:
        H = transform(quad)
    except np.linalg.LinAlgError:
        return result
    rgb = np.array(image.convert("RGB"))
    samples = [_samples(rgb, H, axis, length) for axis, length in enumerate(dimensions)]
    axes = [_axis(*sample, length) for sample, length in zip(samples, dimensions)]
    aspect = _aspect(H, image.size)
    result["evidence"] = dict(axes=axes, patch_aspect=aspect,
                              assumption="one block spans sixteen resolved vanilla texture texels")
    selected = _select(axes, aspect)
    if selected is None:
        result["reason"] = "insufficient native texture evidence on both axes"
        return result
    counts = np.array([a["count"] for a in selected])
    result["evidence"]["selected"] = selected
    result.update(counts=counts.tolist(), confidence=min(1., min(a["strength"] for a in selected)/.3),
                  reason="pitch supported; joint phase unresolved")
    phase_values = [_phase(*sample, count) for sample, count in zip(samples, counts)]
    result["evidence"]["phase_by_axis"] = phase_values
    if any(p is None for p in phase_values):
        return result
    phases = np.array(phase_values)
    UV = np.diag([1/counts[0], 1/counts[1], 1.])
    UV[:2, 2] = phases/counts
    lines = [(np.arange(0, np.floor(n - phase) + 1) + phase)/n for n,phase in zip(counts, phases)]
    result.update(grid=dict(phase_blocks=phases.tolist(), grid_to_image=(H @ UV).tolist(),
                            u_lines=lines[0].tolist(), v_lines=lines[1].tolist()),
                  reason="native texel pitch and a unique repeated dark joint on each axis")
    return result
