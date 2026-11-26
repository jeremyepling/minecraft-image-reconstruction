"""Compare bounded iterations of the existing native-pixel evidence checks."""
from engine import neighbor_photo_faces, visible_horizontal_faces
from engine.atlas_plane_depth import place_ranked


def extend(image, parts, camera, cells, rounds, photo):
    for _ in range(rounds):
        prior = cells
        if photo:
            cells = cells | neighbor_photo_faces.recover(image, parts, camera['P'], cells)
        cells = visible_horizontal_faces.recover(image, parts, camera, cells)
        if cells == prior:
            break
    return cells


def _run(image, version, parts, counts, camera, rounds, photo):
    cells, status = place_ranked(image, version, parts, counts, camera)
    if not cells or status.get('foreground_depth',{}).get('method') != 'joint':
        return cells,status
    if not photo and not status['dense_horizontal_blocks']:
        return cells,status
    return extend(image, parts, camera, cells, rounds, photo),status


def horizontal_once(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,1,False)


def horizontal_four(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,4,False)


def photo_twice(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,2,True)
