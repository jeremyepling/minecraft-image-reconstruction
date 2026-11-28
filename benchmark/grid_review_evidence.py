"""Conservative first-pass comparisons against recorded, visible grid evidence."""
import math


def grid_preference(a, b):
    """Compare wall geometry on the same native faces, independently of export."""
    if min(a.get("faces", 0), b.get("faces", 0)) < 4:
        return dict(preferred=None, reason="insufficient visible reference faces")
    for option in (a, b):
        values = [option.get(key) for key in ("covered", "phase", "cells")]
        if any(value is None or not math.isfinite(value) for value in values) or option["cells"] <= 0:
            return dict(preferred=None, reason="invalid visible grid measurement")
    errors = (
        ("coverage", a["covered"] - b["covered"], 2.0),
        ("corner alignment", b["phase"] - a["phase"], .04),
        ("block scale", abs(math.log(b["cells"])) - abs(math.log(a["cells"])), math.log(1.15)),
    )
    better_a = [name for name, delta, margin in errors if delta >= margin - 1e-9]
    better_b = [name for name, delta, margin in errors if delta <= -margin + 1e-9]
    if better_a and better_b:
        preferred, reason = None, "reference measures disagree"
    elif better_a or better_b:
        preferred, reason = ("A", "A fits the visible reference better") if better_a else (
            "B", "B fits the visible reference better")
    else:
        preferred, reason = None, "both grids are close on these measures"
    return dict(preferred=preferred, reason=reason, evidence_for_a=better_a, evidence_for_b=better_b)
