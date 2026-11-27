"""Apply the frozen surface-only chooser to an already finalized strict and texture baseline."""
from benchmark.inset_mask_chooser import refine
from benchmark.inset_outline_count_guard import apply


def supplemental(image, original_parts, camera, baseline_parts, strict_events, *, atlas=None):
    if not camera or camera.get('P') is None:
        return baseline_parts,[]
    provisional, events=refine(image,original_parts,camera,baseline_parts,strict_events)
    return apply(image,baseline_parts,provisional,events,camera,atlas,collar=True)
