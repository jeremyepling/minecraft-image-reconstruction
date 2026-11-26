"""Replay secondary visible depths above the frozen standalone ranker."""
import numpy as np

from engine import visible_placement as placer
from engine.atlas_plane_depth import frozen_weights,options
from engine.camera_atlas_floor import _atlas
from engine.plane_offset_features import ranked_choice
from engine.secondary_plane_depth import wrap_selector


def predict(image,version,parts,counts,camera):
    if camera['P'] is None or not parts:
        return placer.place_with_camera(image,version,parts,counts,camera)
    rgb=np.asarray(image)
    atlas=_atlas(version)
    weights=frozen_weights()
    if atlas is None or weights is None:
        return placer.place_with_camera(image,version,parts,counts,camera)
    def select(part,mask,P,eye,rays,axis,baseline,scene):
        opts=options(mask,P,eye,rays,axis,baseline,rgb,atlas,scene)
        return ranked_choice(opts,baseline,weights)
    select, refine = wrap_selector(select,rgb,atlas)
    return placer.place_with_camera(image,version,parts,counts,camera,select,refine)
