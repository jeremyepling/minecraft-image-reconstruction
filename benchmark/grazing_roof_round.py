"""Native-pixel texture trial after the previously selected plane and edge stages."""
from benchmark.canopy_composition import with_edges
from engine.camera_atlas_floor import _atlas
from engine.grazing_faces import verified
from engine.visible_ray_filter import fully_occluded_cells


def run(image,version,parts,counts,camera,union,oriented=False):
    cells,status=with_edges(image,version,parts,counts,camera)
    if offsets:=status.get('plane_offsets'):
        new=verified(image,parts,camera['P'],cells,offsets,_atlas(version),union=union,oriented=oriented)
        if new:
            cells=cells|new
            cells-=fully_occluded_cells(image.size,camera['P'],cells)
    return cells,status


def horizontal(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,False)


def union(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,True)


def oriented(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,True,True)
