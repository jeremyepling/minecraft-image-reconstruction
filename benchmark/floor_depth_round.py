"""Offline adapter for calibrated relative-depth placement. Only RGB indexes the model cache."""
from unittest.mock import patch

from engine import ground_depth, visible_placement
from benchmark.floor_depth_fields import field


def run(image,version,parts,counts,camera,kind,anchored=False):
    original = visible_placement._estimate_plane_cells
    def estimated(img,parts,masks,P,eye):
        levels = ground_depth.plane_levels(img,parts,masks,P,eye,field(img,kind),inverse=kind=='inverse',anchored=anchored)
        return original(img,parts,masks,P,eye,levels)
    with patch.object(visible_placement,'seam_camera',return_value=camera), \
         patch.object(visible_placement,'_estimate_plane_cells',side_effect=estimated):
        return visible_placement.place(image,version,parts,counts=counts)


def predict(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'inverse')


def points(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'points')


def anchored(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'inverse',True)
