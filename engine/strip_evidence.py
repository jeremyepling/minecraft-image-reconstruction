"""Require local photographic evidence before removing a proposed narrow extension."""
import numpy as np
from PIL import Image, ImageFilter

from engine.entrance_regions import _components


SETTINGS = dict(outside_radius=2, body_radius=8, minimum_pixels=8,
                minimum_color_advantage=6., maximum_outside_error=18.)


def neighborhood(mask, radius):
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).filter(
        ImageFilter.MaxFilter(2 * radius + 1)), dtype=bool)


def palette_distance(first, second):
    # Several quantiles keep shading and block texture in the comparison.
    quantiles = [.2, .5, .8]
    return abs(np.quantile(first, quantiles, axis=0)
               - np.quantile(second, quantiles, axis=0)).mean()


def gate(rgb_local, owned, removed):
    """Accept a whole attached tail when its colors agree with its surroundings."""
    accepted = np.zeros_like(removed)
    radius = SETTINGS['body_radius']
    for coords in _components(removed):
        start = np.maximum(coords.min(axis=0) - radius, 0)
        end = np.minimum(coords.max(axis=0) + radius + 1, owned.shape)
        box = np.s_[start[0]:end[0], start[1]:end[1]]
        tail = np.zeros_like(owned[box])
        y, x = (coords-start).T
        tail[y,x] = True
        outside = neighborhood(tail, SETTINGS['outside_radius']) & ~owned[box]
        body = neighborhood(tail, radius) & owned[box] & ~removed[box]
        if min(len(coords), outside.sum(), body.sum()) < SETTINGS['minimum_pixels']:
            continue
        rgb = rgb_local[box]
        outer_error = palette_distance(rgb[tail], rgb[outside])
        body_error = palette_distance(rgb[tail], rgb[body])
        if (outer_error <= SETTINGS['maximum_outside_error'] and
                body_error - outer_error >= SETTINGS['minimum_color_advantage']):
            accepted[box] |= tail
    return accepted
