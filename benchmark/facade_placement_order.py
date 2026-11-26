"""Compare the timing of the same visible vertical-face verifier on ordinary image-only inputs."""
from unittest.mock import patch

from benchmark.texture_camera_eval import STABLE
from engine import atlas_plane_depth, neighbor_photo_faces, vertical_atlas_faces
from engine.camera_atlas_floor import _atlas


ATLAS = STABLE / 'runs/26.3/atlas'


def facade_first(image, parts, counts, camera, prior):
    """Move vertical atlas verification to immediately before dense horizontal recovery."""
    neighbor = neighbor_photo_faces.recover
    verify = vertical_atlas_faces.recover
    atlas = _atlas('26.3', ATLAS)
    added = []

    def verify_after_neighbors(photo, surfaces, projection, cells):
        proposals = neighbor(photo, surfaces, projection, cells)
        provisional = cells | proposals
        checked = verify(photo, surfaces, camera, provisional, atlas)
        added.append(len(checked - provisional))
        return proposals | (checked - provisional)

    def skip_late(photo, surfaces, projection, cells, atlas):
        return cells

    with patch.object(neighbor_photo_faces, 'recover', verify_after_neighbors), \
         patch.object(vertical_atlas_faces, 'recover', skip_late):
        cells, status = atlas_plane_depth.place_ranked(image, '26.3', parts, counts, camera, atlas_path=ATLAS)
    return cells, dict(vertical_at_first_pass=added, height_revisions=status.get('roof_height_revisions', {}))


def facade_last(image, parts, counts, camera, prior):
    """Give newly exposed vertical evidence a second check after the two roof refinements."""
    if not prior or camera.get('P') is None:
        return prior, dict(verified_after_roof=0)
    atlas = _atlas('26.3', ATLAS)
    cells = vertical_atlas_faces.recover(image, parts, camera, prior, atlas)
    return cells, dict(verified_after_roof=len(cells-prior))


def facade_repeat(image, parts, counts, camera, prior):
    """Check newly exposed vertical neighbors up to three times after the roof pass."""
    if not prior or camera.get('P') is None:
        return prior, dict(verified_per_pass=[])
    atlas = _atlas('26.3', ATLAS)
    current = prior
    increments = []
    for _ in range(3):
        checked = vertical_atlas_faces.recover(image, parts, camera, current, atlas)
        increments.append(len(checked - current))
        if checked == current:
            break
        current = checked
    return current, dict(verified_per_pass=increments)
