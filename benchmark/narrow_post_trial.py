"""Replay the three native boundary tolerances for pillar rows."""
from benchmark.framed_geometry_order import ATLAS
from engine.camera_atlas_floor import _atlas
from engine.pillar_rows import recover


def placed(image,parts,counts,camera,prior,*,border=0):
    return recover(image,parts,camera,prior,_atlas("26.3",ATLAS),border=border)


def one_pixel(image,parts,counts,camera,prior):
    return placed(image,parts,counts,camera,prior,border=1)


def two_pixels(image,parts,counts,camera,prior):
    return placed(image,parts,counts,camera,prior,border=2)
