"""Replay photographed pillar rows followed by one verified local depth exchange."""
from benchmark.framed_geometry_order import ATLAS
from engine.camera_atlas_floor import _atlas
from engine import local_panel_depth,pillar_rows


def combined(image,parts,counts,camera,prior):
    posts,post_info=pillar_rows.recover(image,parts,camera,prior,_atlas('26.3',ATLAS))
    result,panel_info=local_panel_depth.recover(image,parts,counts,camera,posts,atlas_path=ATLAS)
    return result,dict(post=post_info,panel=panel_info)
