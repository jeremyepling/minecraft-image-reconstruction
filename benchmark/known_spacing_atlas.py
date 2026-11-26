"""Locate missing grid offsets by matching vanilla art at measured block spacing."""
from math import isfinite

from benchmark import template_block_seam


def predict(sample, baseline):
    counts = list(baseline['counts'])
    phase = list(baseline['phase'])
    result = dict(counts=counts, phase=phase)
    if (sample is None or None not in phase
            or any(n is None or not isfinite(n) or n <= 0 for n in counts)):
        return result

    # With both counts measured, the template matcher tests just that pair at
    # shifted origins. Its normal correlation and ambiguity checks still apply.
    registration = template_block_seam.predict(sample, baseline)
    for axis, offset in enumerate(registration['phase']):
        if phase[axis] is None:
            phase[axis] = offset
    return result
