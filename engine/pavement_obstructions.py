"""Test whether visible cube faces interrupt a source-verified pavement opening."""
import numpy as np

from engine.pillar_seams import native_mask, orientation
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def first_obstructions(size, camera, opening_parts, cells):
    """Find above-ground cubes whose visible face falls mainly inside the opening."""
    if not cells or not opening_parts or not camera or camera.get('P') is None:
        return set()
    projection, _ = orientation(camera, size)
    eye = -np.linalg.solve(projection[:, :3], projection[:, 3])
    ordered = sorted(cells)
    # Match the placement raster. A partial stride counts only if its sample
    # center still lies inside the image.
    shape = tuple((n + (SAMPLE_STRIDE - 1) // 2) // SAMPLE_STRIDE for n in size[::-1])
    owner, axes = _frontmost_cells(ordered, projection, eye, shape, face_axes=True)
    x = np.arange(shape[1]) * SAMPLE_STRIDE + SAMPLE_STRIDE // 2
    y = np.arange(shape[0]) * SAMPLE_STRIDE + SAMPLE_STRIDE // 2
    blocked = set()
    # Assess each opening independently. Tiny fragments from unrelated windows
    # cannot combine into enough evidence against a legitimate post.
    visible = owner >= 0
    index = 3 * owner[visible] + axes[visible]
    face_samples = np.bincount(index, minlength=len(ordered) * 3).reshape(-1, 3)
    for part in opening_parts:
        mask = native_mask(part, size)[np.ix_(y, x)]
        hit = mask & visible
        hits = np.bincount(3 * owner[hit] + axes[hit], minlength=len(ordered) * 3).reshape(-1, 3)
        # The ground plane is at y=0. Retain flush tiles and the entire first
        # block course. Require four visible samples and majority support on
        # one face, plus at least a fifth of the cube's visible footprint.
        faces = (hits >= 4) & (2 * hits > face_samples)
        enough = faces.any(axis=1) & (5 * hits.sum(axis=1) >= face_samples.sum(axis=1))
        blocked.update(cell for cell, supported in zip(ordered, enough) if supported and cell[1] >= 1)
    return blocked


def window_obstructions(size, camera, opening_parts, cells):
    """Reproject as each removal reveals the next actual face toward the ground."""
    kept = set(cells)
    while kept:
        blocked = first_obstructions(size, camera, opening_parts, kept)
        if not blocked:
            break
        kept.difference_update(blocked)
    return set(cells) - kept
