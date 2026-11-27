"""Follow a wall's photographed color through a narrow gap to its open sky edge."""
import numpy as np

from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha


METHOD = 'observed-wall-skyline-v1'
MAX_GAP = 4
MIN_SPAN = 12


def _horizon(part, height):
    # For an upright, roll-free Minecraft image a vertical wall is
    # perpendicular to camera-space gravity (0, gy, gz). Its normal bounds
    # the horizon across a conservative 50–90 degree vertical FOV interval.
    n = part.get('normal_camera')
    if n is None or len(n) != 3 or n[2] >= -.30 or not -.30 < n[1] < -.03:
        return None
    return min(height * (.5 + .5 * n[1] / n[2] / np.tan(np.radians(fov / 2)))
               for fov in (50, 90))


def refine(image, parts, sky):
    h, w = image.height, image.width
    horizons = [_horizon(p, h) for p in parts]
    if min(w, h) < 16 or all(v is None for v in horizons) or not sky[0].any():
        return parts, []
    # Track only the uppermost supported pixel in each column, with ties
    # marked ambiguous. A foreground occluder or another mask retains ownership.
    first = np.full(w, h, dtype=int)
    owner = np.full(w, -1, dtype=int)
    bounds = []
    for i, part in enumerate(parts):
        patch = _native_alpha(part['support'], image.size) if part.get('support') else None
        bounds.append(patch)
        if not patch:
            continue
        alpha, (l, t, r, _) = patch
        local = np.asarray(alpha, dtype=bool)
        top = np.where(local.any(0), t + local.argmax(0), h)
        old = first[l:r]
        owner[l:r] = np.where(top < old, i, np.where(top == old, -1, owner[l:r]))
        np.minimum(old, top, out=old)
    sky_end = (~sky).argmax(0)
    distance = first - sky_end
    eligible = sky[0] & (distance >= 1) & (distance <= MAX_GAP) & (first >= 7) & (first + 3 < h)
    if eligible.sum() < MIN_SPAN:
        return parts, []
    rgb = np.asarray(image.convert('RGB'), dtype=np.int16)
    result = list(parts)
    log = []
    for i, horizon in enumerate(horizons):
        if horizon is None or bounds[i] is None:
            continue
        xs = np.flatnonzero(eligible & (owner == i) & (first + h * .04 < horizon))
        if len(xs) < MIN_SPAN:
            continue
        alpha, (l, t, r, b) = bounds[i]
        mask = np.asarray(alpha, dtype=bool)
        y = first[xs]
        deep = y + 2 < b
        xs, y = xs[deep], y[deep]
        top = sky_end[xs]
        use = mask[y + 1 - t, xs - l] & mask[y + 2 - t, xs - l]
        use &= np.max(abs(rgb[top, xs] - rgb[top - 1, xs]), axis=1) >= 28
        inside, neighbor = rgb[y, xs], rgb[y + 1, xs]
        for step in range(1, MAX_GAP + 1):
            color = rgb[y - step, xs]
            error = np.maximum(abs(color - inside), abs(color - neighbor))
            supported = ~sky[y - step, xs] & (error.max(axis=1) <= 4)
            use &= (step > distance[xs]) | supported
        xs = xs[use]
        groups = [g for g in np.split(xs, np.flatnonzero(np.diff(xs) != 1) + 1)
                  if len(g) >= MIN_SPAN and np.max(abs(np.diff(sky_end[g]))) <= 1]
        if not groups:
            continue
        xs = np.concatenate(groups)
        new_top = min(t, int(sky_end[xs].min()))
        updated = np.zeros((b - new_top, r - l), dtype=bool)
        updated[t - new_top:] = mask
        for step in range(1, MAX_GAP + 1):
            use = xs[distance[xs] >= step]
            updated[first[use] - step - new_top, use - l] = True
        runs = (np.asarray(row_runs(updated)) + [new_top, l, l]).tolist()
        copy = {k:v for k,v in parts[i].items()
                if k not in ('outline', 'source_crop', 'context_crop', 'sampling_chart')}
        copy.update(support=dict(size=list(image.size), runs=runs), support_area=int(updated.sum()))
        result[i] = copy
        log.append(dict(surface=copy['id'], added_pixels=int(updated.sum() - mask.sum())))
    return (result if log else parts), log
