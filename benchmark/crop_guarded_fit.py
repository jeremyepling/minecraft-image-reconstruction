"""Expand a clipped chart conservatively using predicted owned pixel centers."""
import numpy as np

from benchmark.crop_full_fit import expand_sample


SETTINGS = {
    "skip_at_or_above_inside_fraction": 0.90,
    "tail_fraction_per_axis": 0.01,
    "max_growth_per_side_in_original_spans": 0.50,
}


def guarded_bounds(uv):
    """Return extent in the original chart coordinate system, or keep it."""
    inside = ((uv >= 0) & (uv <= 1)).all(axis=1)
    if not len(inside) or inside.mean() >= SETTINGS["skip_at_or_above_inside_fraction"]:
        return None

    tail = SETTINGS["tail_fraction_per_axis"]
    low, high = np.quantile(uv, [tail, 1 - tail], axis=0)
    limit = SETTINGS["max_growth_per_side_in_original_spans"]
    return np.clip(low, -limit, 0), np.clip(high, 1, 1 + limit)


def sample(image, part):
    """Keep the working chart and four-tap ownership test from Surface Sample."""
    return expand_sample(image, part, guarded_bounds)
