"""Compare placement refinements on identical upstream image measurements."""
from engine import nearby_face_decision, thin_atlas_faces
from engine.atlas_plane_depth import place_ranked
from engine.camera_atlas_floor import _atlas


def compose(image, version, parts, counts, camera, *, nearby=True, thin=True):
    # This experiment is pinned to the second-pass placer at 2b436b2.
    cells, status = place_ranked(image, version, parts, counts, camera)
    if not cells or status.get('foreground_depth',{}).get('method') != 'joint':
        return cells,status
    if nearby:
        cells = nearby_face_decision.recover(image,parts,camera,cells)
    if thin:
        cells = thin_atlas_faces.recover(image,parts,camera,cells,_atlas(version))
    return cells,status


def combined(image,version,parts,counts,camera):
    return compose(image,version,parts,counts,camera)


def atlas_and_second_pass(image,version,parts,counts,camera):
    return compose(image,version,parts,counts,camera,nearby=False)
