from engine.atlas_plane_depth import place_ranked
from engine.camera_atlas_floor import _atlas
from engine.visible_edge_faces import extend


def run(image,version,parts,counts,camera,strict,search=False,creases=False):
    cells,status=place_ranked(image,version,parts,counts,camera)
    if offsets:=status.get('plane_offsets'):
        added=extend(image,parts,camera['P'],cells,offsets,_atlas(version),strict=strict,search=search,creases=creases)
        cells=cells|added
        status.update(edge_faces=len(added),placed_blocks=len(cells))
    return cells,status


def broad(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,False)


def guarded(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,True)


def alternate(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,False,True)


def crease(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,False,True,True)
