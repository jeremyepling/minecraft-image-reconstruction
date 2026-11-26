"""Apply one frozen challenger to the ordinary sealed control placement."""
import os
from pathlib import Path

import engine
from engine.atlas_plane_depth import place_ranked
from engine.camera_atlas_floor import _atlas


def _overlay():
    external = str(Path(os.environ['MCIR_BLIND_OVERLAY']).resolve()/'engine')
    if external not in engine.__path__:
        engine.__path__.append(external)


def nearby(image,version,parts,counts,camera):
    cells,status=place_ranked(image,version,parts,counts,camera)
    _overlay()
    from engine.nearby_face_decision import recover
    return recover(image,parts,camera,cells),status


def thin(image,version,parts,counts,camera):
    cells,status=place_ranked(image,version,parts,counts,camera)
    _overlay()
    from engine.thin_atlas_faces import recover
    return recover(image,parts,camera,cells,_atlas(version)),status
