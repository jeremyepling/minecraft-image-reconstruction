"""Reject surface proposals supported entirely by border-connected sky."""
import numpy as np

from engine.camera_lines import _runs
from engine.surface_perimeter import _native_alpha

METHOD = "border-connected-whole-sky-v1"


def sky_support(image):
    """Compare each pixel to its row's border sky and plausible whitening by cloud."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    left, right = rgb[:, 0], rgb[:, -1]
    # Require independent agreement from two sides, and a blue, unsaturated-red sky.
    row_ok = ((np.abs(left - right).max(axis=1) <= 2) &
              (left[:, 2] >= 230) & (left[:, 2] - left[:, 0] >= 40) &
              (left[:, 1] - left[:, 0] >= 15))
    base = left[:, None, :]
    delta = rgb - base
    white_mix = delta[:, :, 0] / np.maximum(1, 255 - base[:, :, 0])
    expected_green = base[:, :, 1] + white_mix * (255 - base[:, :, 1])
    # A pale cloud must retain the sky's blue tint and similar mixing in R/G.
    compatible = (row_ok[:, None] & (delta[:, :, 0] >= -2) &
                  (np.abs(rgb[:, :, 1] - expected_green) <= 6) &
                  (np.abs(delta[:, :, 2]) <= 4) & (rgb[:, :, 2] - rgb[:, :, 0] >= 7))
    result = np.zeros_like(compatible)
    width = image.width
    for runs in _runs(compatible):
        if any(a == 0 or b == width for _, a, b in runs):
            for y, a, b in runs:
                result[y, a:b] = True
    return result


def filter_proposals(image, planes):
    sky = sky_support(image)
    kept, rejected = [], []
    for plane in planes:
        native = _native_alpha(plane["support"], image.size)
        if native is not None:
            alpha, (x0, y0, x1, y1) = native
            pixels = np.asarray(alpha) != 0
            if pixels.any() and sky[y0:y1, x0:x1][pixels].all():
                rejected.append(plane["id"])
                continue
        kept.append(plane)
    return kept, dict(input_count=len(planes), kept_count=len(kept), rejected_ids=rejected)
