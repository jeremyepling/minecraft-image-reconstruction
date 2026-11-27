"""Use this candidate's placer when the comparison process preloads the control module."""
from importlib import reload
from pathlib import Path

from benchmark.inset_return_grid import refine
from engine import atlas_plane_depth

wanted=Path(__file__).resolve().parents[1]/'engine/atlas_plane_depth.py'
if Path(atlas_plane_depth.__file__).resolve()!=wanted:
    reload(atlas_plane_depth)
if Path(atlas_plane_depth.__file__).resolve()!=wanted:
    raise RuntimeError('Depth-handoff comparison loaded the wrong placement revision')
