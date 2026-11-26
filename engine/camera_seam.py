"""Estimate an upright metric camera from two periodic families of ground seams."""
import numpy as np
from PIL import Image


SETTINGS = {
    "seed_width_limit": 768,
    "horizon_trials": 7,
    "floor_rows": 88,
    "center_fit_iterations": 14,
    "minimum_supported_lines": 4,
    "maximum_height_disagreement": 0.08,
    "maximum_normalized_residual": 0.04,
}


def _sample(field, x, y):
    """Bilinear sampling in pixel-center coordinates (top-left pixel is .5,.5)."""
    x = np.clip(x - 0.5, 0, field.shape[1] - 1.00001)
    y = np.clip(y - 0.5, 0, field.shape[0] - 1.00001)
    i, j = x.astype(int), y.astype(int)
    tx, ty = x - i, y - j
    return ((1 - ty) * ((1 - tx) * field[j, i] + tx * field[j, i + 1])
            + ty * ((1 - tx) * field[j + 1, i] + tx * field[j + 1, i + 1]))


def _pencils(rgb):
    """A fronto-parallel ground warp makes both pencils periodic in two dimensions."""
    h, w = rgb.shape[:2]
    reference = 0.86 * h
    edges = np.median(np.concatenate((rgb[:, :w // 12], rgb[:, -w // 12:]), axis=1), axis=1)
    chroma = edges[:, 2] - edges[:, 0]
    boundary = np.arange(h // 8, 2 * h // 3)
    drops = chroma[boundary - 3] - chroma[boundary + 3]
    if np.max(drops) < 10:
        return None
    sky_edge = boundary[np.argmax(drops)]
    gray = rgb.mean(2)
    u = np.arange(-0.38 * w, 0.38 * w)
    best = None
    for horizon in sky_edge + h * np.linspace(-0.15, 0.015, SETTINGS["horizon_trials"]):
        distance = reference - horizon
        v_limits = distance ** 2 / (h * np.array([0.982, 0.735]) - horizon) - distance
        v = np.arange(np.ceil(v_limits[0]), np.floor(v_limits[1]))
        y = horizon + distance ** 2 / (distance + v)
        x = w / 2 + u[None, :] * (y[:, None] - horizon) / distance
        patch = _sample(gray, x, y[:, None])
        # A few foreground pixels and shadows can otherwise swamp the floor texture.
        center = np.median(patch)
        patch = np.clip(center - patch, -22, 45)
        patch -= patch.mean(1, keepdims=True)
        window = np.hanning(len(v))[:, None] * np.hanning(len(u))[None, :]
        patch *= window
        frequency = np.fft.rfft2(patch, s=(4 * len(v), 2 * len(u)), axes=(0, 1))
        amplitude = np.abs(frequency)
        fu = np.fft.rfftfreq(2 * len(u))[None, :]
        fv = np.fft.fftfreq(4 * len(v))[:, None]
        slopes = -fv / np.maximum(1.e-9, fu)
        allowed = ((fu >= 1 / (0.55 * w)) & (fu < 0.13) &
                   (np.abs(fv) > 0.002) & (np.abs(fv) < 0.20))
        families = []
        strengths = []
        for sign in (-1, 1):
            keep = allowed & (sign * slopes > 0.20) & (sign * slopes < 16)
            if not keep.any():
                break
            j, i = np.unravel_index(np.argmax(np.where(keep, amplitude, -1)), amplitude.shape)
            # A narrow trough can put more power in a harmonic. Prefer its
            # fundamental only if the subharmonic carries comparable power.
            for k in (4, 3, 2):
                jj = round(float(fv[j, 0]) * len(fv) / k) % len(fv)
                ii = round(i / k)
                if keep[jj, ii] and amplitude[jj, ii] > 0.64 * amplitude[j, i]:
                    j, i = jj, ii
                    break
            ax, ay = float(fu[0, i]), float(fv[j, 0])
            period, m = 1 / ax, -ay / ax
            # At the reference row v=0, convert FFT phase into the x of seam 0.
            phase = np.angle(frequency[j, i]) / (2 * np.pi)
            intercept = w / 2 + (ax * u[0] + ay * v[0] - phase) * period
            intercept += round((w / 2 - intercept) / period) * period
            vp = w / 2 + m * distance
            families.append([intercept, (intercept - vp) / distance, period, period / distance])
            strengths.append(float(amplitude[j, i] / np.sqrt((patch ** 2).sum() * window.sum() + 1)))
        if len(families) == 2:
            quality = min(strengths)
            if best is None or quality > best[0]:
                best = quality, families
    if best is None or best[0] < 0.12:
        return None
    return np.array(best[1])


def _observations(gray, reference, model):
    h, w = gray.shape
    intercept, tilt, pitch, growth = model
    y = np.linspace(0.70 * h, 0.985 * h, SETTINGS["floor_rows"])
    dy = y - reference
    spacing = pitch + growth * dy
    if np.min(spacing) < 5:
        return None
    nmax = int(np.ceil(w / min(spacing) + abs(tilt * (y[0] - reference)) / pitch)) + 3
    n, dy = np.meshgrid(np.arange(-nmax, nmax + 1), dy)
    width = pitch + growth * dy
    center = intercept + tilt * dy + n * width
    valid = (center > 0.22 * width) & (center < w - 0.22 * width)
    n, dy, center, width = (arr[valid] for arr in (n, dy, center, width))
    if len(n) < 24:
        return None
    # Estimate the middle of the dark stripe from its neighboring bright faces.
    # We fit at actual pixel centers; this matters at magnified building corners.
    sample_positions = np.linspace(-0.21, 0.21, 43)
    profiles = _sample(gray, center[:, None] + width[:, None] * sample_positions,
                       reference + dy[:, None])
    shoulders = np.minimum(np.median(profiles[:, :5], axis=1), np.median(profiles[:, -5:], axis=1))
    weights = np.maximum(0, shoulders[:, None] - profiles - 3)
    total = weights.sum(1)
    peak = weights.max(1)
    delta = (weights @ sample_positions) / np.maximum(1, total)
    good = (peak > 9) & (total > 26) & (np.abs(delta) < 0.16)
    design = np.column_stack([np.ones_like(n), dy, n, n * dy])
    return design[good], (center + delta * width)[good], width[good]


def _fit_family(gray, reference, model):
    for _ in range(SETTINGS["center_fit_iterations"]):
        result = _observations(gray, reference, model)
        if result is None:
            return None
        design, targets, widths = result
        if len(design) < 40 or len(np.unique(design[:, 2])) < SETTINGS["minimum_supported_lines"]:
            return None
        # Robust regression rejects other pencil crossings and object pixels.
        for _ in range(4):
            residuals = (targets - design @ model) / widths
            weight = 1 / (1 + (residuals / 0.035) ** 2)
            model = np.linalg.lstsq(weight[:, None] * design, weight * targets, rcond=None)[0]
        if model[2] <= 5 or model[3] <= 0:
            return None
    residual = np.abs(targets - design @ model) / widths
    good = residual < SETTINGS["maximum_normalized_residual"]
    if np.mean(good) < 0.65 or len(np.unique(design[good, 2])) < SETTINGS["minimum_supported_lines"]:
        return None
    if np.ptp(design[good, 1]) < 0.18 * gray.shape[0]:
        return None
    return model, design[good], targets[good], float(np.median(residual))


def _perspective(fits, w, h):
    reference = 0.86 * h
    horizons = [reference - f[0][2] / f[0][3] for f in fits]
    if abs(horizons[0] - horizons[1]) > 0.035 * h:
        return None
    # Solve the common horizon from both pencils. At a fixed horizon all other
    # seam parameters are linear. Minimize per-observation subpixel residuals.
    def constrained(horizon):
        loss, models = 0., []
        for _, design, targets, _ in fits:
            dy, n = design[:, 1:3].T
            matrix = np.column_stack((np.ones_like(n), dy, n * (1 + dy / (reference - horizon))))
            model = np.linalg.lstsq(matrix, targets, rcond=None)[0]
            loss += np.mean(np.minimum((matrix @ model - targets) ** 2, 9)) / model[2] ** 2
            models.append(model)
        return loss, models

    candidates = np.linspace(min(horizons) - 2, max(horizons) + 2, 31)
    horizon = min(candidates, key=lambda v: constrained(v)[0])
    models = constrained(horizon)[1]
    vps = [center + tilt * (horizon - reference) for center, tilt, _ in models]
    focal_sq = -(vps[0] - w / 2) * (vps[1] - w / 2) - (horizon - h / 2) ** 2
    if not (0.35 * h) ** 2 < focal_sq < (3 * h) ** 2:
        return None
    focal = np.sqrt(focal_sq)
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    directions = np.linalg.solve(K, np.array([[v, horizon, 1] for v in vps]).T).T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    up = np.cross(*directions)
    up *= np.sign(-up[1])
    heights, coords = [], []
    for index, (center, _, period) in enumerate(models):
        rays = np.linalg.solve(K, np.array([[center, reference, 1], [center + period, reference, 1]]).T).T
        ground = -rays / (rays @ up)[:, None]
        coord = ground @ directions[1 - index]
        heights.append(1 / abs(coord[1] - coord[0]))
        coords.append(coord[0])
    if 1 - min(heights) / max(heights) > SETTINGS["maximum_height_disagreement"]:
        return None
    height = np.mean(heights)
    offsets = height * np.array(coords[::-1])
    # Choose the lattice intersection near the lower middle of the building.
    ray = np.linalg.solve(K, [w / 2, max(0.67 * h, horizon + 0.12 * h), 1])
    near = (-height * ray / (ray @ up)) @ directions.T
    offsets += np.round(near - offsets)
    origin = directions.T @ offsets - height * up
    basis = np.column_stack((directions[0], up, directions[1], origin))
    return (K @ basis).tolist(), dict(horizon_px=float(horizon), focal_px=float(focal),
                                      height_blocks=float(height), axis_heights_blocks=heights,
                                      median_seam_residual_blocks=[f[3] for f in fits],
                                      seam_samples=[len(f[1]) for f in fits])


def predict(image, version="26.3"):
    """One source image in, a 3x4 local unit-grid projection or an abstention out."""
    def uncertain(reason):
        return dict(P=None, scale_status="uncertain", reason=reason, settings=SETTINGS)

    if version != "26.3":
        raise ValueError("The seam camera is benchmarked for Java 26.3")
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 128:
        return uncertain("Insufficient resolution for two ground-seam families")
    seed_width = min(w, SETTINGS["seed_width_limit"])
    seed_height = round(h * seed_width / w)
    if seed_height < 64:
        return uncertain("Insufficient vertical coverage for perspective fitting")
    seed_image = image.resize((seed_width, seed_height), Image.Resampling.BOX)
    seeds = _pencils(np.asarray(seed_image, dtype=float))
    if seeds is None:
        return uncertain("No paired periodic floor signal")
    seeds *= np.array([w / seed_width, (w / seed_width) / (h / seed_height),
                       w / seed_width, (w / seed_width) / (h / seed_height)])
    gray = np.asarray(image, dtype=float).mean(2)
    fits = [_fit_family(gray, 0.86 * h, model) for model in seeds]
    if any(fit is None for fit in fits):
        return uncertain("Insufficient support for repeated seam centers")
    camera = _perspective(fits, w, h)
    if camera is None:
        return uncertain("Ground seams disagree on metric perspective")
    P, evidence = camera
    return dict(P=P, scale_status="supported", reason="Two perpendicular seam families agree assuming one block per floor tile",
                settings=SETTINGS, evidence=evidence)
