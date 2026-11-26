"""Ablation of the final visible-edge check using the same plane decisions."""
from engine.atlas_plane_depth import _place


def without_edges(image, version, parts, counts, camera):
    return _place(image, version, parts, counts, camera, None, remove_occluded=True, recover_edges=False)
