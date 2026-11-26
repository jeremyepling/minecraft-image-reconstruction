"""Test the two sequences for verifying a shared roof patch and front-facing vertical blocks."""
from benchmark.facade_placement_order import ATLAS, strict_repeat
from engine.visible_roof_rectangle import recover


def rectangle(image,parts,counts,camera,cells):
    return recover(image,parts,counts,camera,cells,texture=False,unclaimed=True,shared_patch=True,atlas_path=ATLAS)


def walls_then_roof(image,parts,counts,camera,prior):
    walls,s1=strict_repeat(image,parts,counts,camera,prior)
    result,s2=rectangle(image,parts,counts,camera,walls)
    return result,dict(walls=s1,roof=s2)


def roof_then_walls(image,parts,counts,camera,prior):
    roof,s1=rectangle(image,parts,counts,camera,prior)
    result,s2=strict_repeat(image,parts,counts,camera,roof)
    return result,dict(roof=s1,walls=s2)
