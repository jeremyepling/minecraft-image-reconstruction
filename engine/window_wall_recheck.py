"""Recheck a measured wall face after an elevated pavement hypothesis was cleared."""
import numpy as np

from engine.pillar_seams import orientation
from engine.visible_placement import MINIMUM_SAMPLES, _estimate_plane_cells, _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE, observed_support


def recover(image, parts, status, opening_ids, placed):
    """Reproject measured wall candidates previously hidden by a confirmed opening's plane."""
    levels = {r['id']: r for r in status.get('plane_offsets', ())}
    planes = [p for p in parts if p.get('support') and p['id'] in levels]
    if not placed or not opening_ids.intersection(levels) or not planes:
        return set()
    projection, _ = orientation(status, image.size)
    eye = -np.linalg.solve(projection[:, :3], projection[:, 3])
    shape = tuple((n + (SAMPLE_STRIDE - 1) // 2) // SAMPLE_STRIDE for n in image.size[::-1])
    masks = [observed_support(p['support'], shape, image.size) for p in planes]

    def selected(part, mask, projection, eye, rays, axis, baseline, distance):
        return levels[part['id']]['level']

    candidates = _estimate_plane_cells(image, planes, masks, projection, eye, selected)
    obstructions = set()
    for part, cells in zip(planes, candidates):
        if part['id'] in opening_ids and levels[part['id']]['axis'] == 1:
            obstructions.update(c for c in cells if c[1] > 0 and c not in placed)
    if not obstructions:
        return set()

    wall_faces = []
    for part, mask, cells in zip(planes, masks, candidates):
        axis = levels[part['id']]['axis']
        if axis == 1:
            continue
        for cell in cells - placed:
            x, y, z = cell
            if (x, y - 1, z) not in placed:
                continue
            sides = ((x - 1, y, z), (x + 1, y, z)) if axis == 2 else ((x, y, z - 1), (x, y, z + 1))
            if not any(side in placed for side in sides):
                continue
            sign = 1 if eye[axis] > cell[axis] + .5 else -1
            hit = _sample_face(projection, cell, axis, sign, shape)
            if hit is not None:
                box, pixels, *_ = hit
                wall_faces.append((cell, axis, box, pixels, mask[box]))

    if not wall_faces:
        return set()
    tentative = sorted(placed | {face[0] for face in wall_faces})
    with_opening = sorted(set(tentative) | obstructions)
    before = _frontmost_cells(with_opening, projection, eye, shape)
    obscured = np.isin(before, [i for i, cell in enumerate(with_opening) if cell in obstructions])
    after, axes = _frontmost_cells(tentative, projection, eye, shape, face_axes=True)
    positions = {cell: i for i, cell in enumerate(tentative)}
    additions = set()
    for cell, axis, box, pixels, mask in wall_faces:
        # The same complete wall face must change from hidden to clearly visible.
        # Require a majority of its whole footprint, not just a few rim pixels.
        visible = pixels & mask & (after[box] == positions[cell]) & (axes[box] == axis)
        recovered = visible & obscured[box]
        visible_before = (visible & ~obscured[box]).sum()
        if visible_before < MINIMUM_SAMPLES and recovered.sum() >= max(MINIMUM_SAMPLES, .75 * pixels.sum()):
            additions.add(cell)
    return additions
