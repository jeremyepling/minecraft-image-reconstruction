"""Check elevated cubes along sightlines that show the exterior pavement grid."""
import numpy as np
from PIL import Image, ImageFilter

from engine.pillar_seams import native_mask, orientation
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def _obstructing_faces(ordered, projection, eye, masks):
    owner, axes = _frontmost_cells(ordered, projection, eye, masks[0].shape, face_axes=True)
    visible = owner >= 0
    index = 3 * owner[visible] + axes[visible]
    face_samples = np.bincount(index, minlength=len(ordered) * 3).reshape(-1, 3)
    blocked = set()
    for mask in masks:
        # Fragments from separate openings cannot combine into evidence against a post.
        hit = mask & visible
        hits = np.bincount(3 * owner[hit] + axes[hit], minlength=len(ordered) * 3).reshape(-1, 3)
        # Four samples, a majority of one visible face, and a fifth of the entire
        # visible cube must lie inside this one verified opening.
        faces = (hits >= 4) & (2 * hits > face_samples)
        enough = faces.any(axis=1) & (5 * hits.sum(axis=1) >= face_samples.sum(axis=1))
        blocked.update(cell for cell, supported in zip(ordered, enough) if supported and cell[1] >= 1)
    return blocked


def window_obstructions(size, camera, opening_parts, cells):
    """Reproject after each removal. Flush pavement and the first block course are protected."""
    if not cells or not opening_parts or not camera or camera.get('P') is None:
        return set()
    projection, _ = orientation(camera, size)
    eye = -np.linalg.solve(projection[:, :3], projection[:, 3])
    # Match the placement raster. A partial stride counts only if its sample
    # center still lies inside the image.
    x = np.arange(SAMPLE_STRIDE // 2, size[0], SAMPLE_STRIDE)
    y = np.arange(SAMPLE_STRIDE // 2, size[1], SAMPLE_STRIDE)
    masks = []
    for part in opening_parts:
        # Exclude three source pixels at the rim, where mask rounding can put
        # the edge of a legitimate adjoining post inside the opening.
        alpha = Image.fromarray(native_mask(part, size).astype(np.uint8))
        interior = np.asarray(alpha.filter(ImageFilter.MinFilter(7)), dtype=bool)
        masks.append(interior[np.ix_(y, x)])
    remaining = set(cells)
    while remaining:
        blocked = _obstructing_faces(sorted(remaining), projection, eye, masks)
        if not blocked:
            break
        remaining.difference_update(blocked)
    return set(cells) - remaining
