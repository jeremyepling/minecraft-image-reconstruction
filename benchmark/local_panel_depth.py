"""Replay the geometric panel selector with the shared benchmark atlas."""
from benchmark.texture_camera_eval import STABLE
from engine.camera_atlas_floor import _atlas
from engine import local_panel_depth as panel
from engine.local_panel_depth import choose, materialize


def atlas():
    return _atlas('26.3',STABLE/'runs/26.3/atlas')


def trials(image,parts,camera,prior):
    return panel.trials(image,parts,camera,prior,atlas())


def recover(image,parts,counts,camera,prior):
    scene,options=trials(image,parts,camera,prior)
    picked=choose(options)
    result=materialize(image,scene,prior,picked)
    return result,dict(options=len(options),eligible=picked is not None,changed=result!=prior)
