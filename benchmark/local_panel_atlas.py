"""Compare the native depth verifier with and without local ambiguity resolution."""
from benchmark.texture_camera_eval import STABLE
from engine import local_panel_depth


def recover(image,parts,counts,camera,prior):
    return local_panel_depth.recover(image,parts,counts,camera,prior,atlas_path=STABLE/'runs/26.3/atlas',resolve_ambiguous=False)


def ambiguous(image,parts,counts,camera,prior):
    return local_panel_depth.recover(image,parts,counts,camera,prior,atlas_path=STABLE/'runs/26.3/atlas')
