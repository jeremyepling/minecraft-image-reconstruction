"""Ablate the last image-evidence stage on top of the combined plane selectors."""
from engine.atlas_plane_depth import place_ranked
from engine.camera_atlas_floor import _atlas
from engine.visible_edge_faces import extend
from engine.visible_ray_filter import fully_occluded_cells


def with_edges(image,version,parts,counts,camera):
    cells,status=place_ranked(image,version,parts,counts,camera)
    if offsets:=status.get('plane_offsets'):
        added=extend(image,parts,camera['P'],cells,offsets,_atlas(version),search=True,creases=True)
        if added:
            cells=cells|added
            cells-=fully_occluded_cells(image.size,camera['P'],cells)
    return cells,status
