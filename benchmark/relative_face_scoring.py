"""Grade camera-free, signed voxel output against one frame's visible blocks."""

from operator import index

import numpy as np

from benchmark.camera_labels import screen_matrix
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.expanded_scores import ASSIGNMENT_FIELDS, target_breakdown
from benchmark.typed_structures import geometry_score, score


def _yaw(position, quarters):
    x, y, z = position
    return ((x, y, z), (-z, y, x), (-x, y, -z), (z, y, -x))[quarters]


def yaw_from_anchor_normal(normal_camera, local_normal_axis, pose):
    """Register one declared outward wall normal using the recorded camera axes.

    Camera axes are right, down and forward. A source normal close to a yaw
    boundary or without enough horizontal evidence leaves the orientation
    unresolved. No voxel targets or predicted block coordinates enter this step.
    """
    local = tuple(map(index, local_normal_axis))
    if len(local) != 3 or local[1] != 0 or abs(local[0]) + abs(local[2]) != 1:
        raise ValueError('The local anchor must be a signed horizontal unit axis')
    normal = np.asarray(normal_camera, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.linalg.norm(normal):
        raise ValueError('Expected a finite nonzero camera-space anchor normal')
    normal = normal / np.linalg.norm(normal)
    w, h = pose['image_size']
    basis = screen_matrix(pose)[:, :3]
    basis[:2] -= np.asarray([w / 2, h / 2])[:, None] * basis[2]
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    possible_world = np.asarray([_yaw(local, q) for q in range(4)])
    similarities = possible_world @ basis.T @ normal
    best, second = sorted(range(4), key=lambda q: -similarities[q])[:2]
    margin = float(similarities[best] - similarities[second])
    resolved = similarities[best] > np.sqrt(.5) and margin >= .10
    return dict(yaw_quarters=best if resolved else None,
                cosine=float(similarities[best]), cosine_margin=margin,
                local_normal=list(local), normal_camera=normal.tolist(),
                best_world_normal=possible_world[best].tolist())


def evaluate_relative_voxels(predicted, reference, *, yaw_quarters=None, relaxed_yaw=False, assigned=None):
    """Return scores and aligned IDs for the fixed orientation, or the relaxed winner.

    A declared yaw is fixed before scoring. The optional relaxed score searches
    the four whole-build upright quarter turns. Every score chooses one integer
    XYZ translation using strict geometry, including its deterministic tie rule,
    and uses that same translation for facade, exact IDs and target breakdowns.
    """
    predicted = {tuple(map(index, (x, y, z))): name for (x, y, z), name in predicted.items()}
    if assigned is not None and not assigned <= predicted.keys():
        raise ValueError('Material assignment outside the predicted voxel set')
    if yaw_quarters is not None:
        yaw_quarters = index(yaw_quarters)
        if yaw_quarters not in range(4):
            raise ValueError('Expected a yaw of 0 through 3 quarters')

    def align(quarters):
        mapping = {cell: _yaw(cell, quarters) for cell in predicted}
        offset, matched = best_translation_fast(mapping.values(), reference.truth)
        dx, dy, dz = offset
        mapping = {cell: (x + dx, y + dy, z + dz) for cell, (x, y, z) in mapping.items()}
        aligned = {world: predicted[cell] for cell, world in mapping.items()}
        assigned_world = {mapping[cell] for cell in assigned or ()}
        geometry = geometry_score(aligned, reference.truth, reference.optional)
        scores = score(aligned, reference.truth, reference.optional, assigned_world)
        if assigned is None:
            scores.update(dict.fromkeys(ASSIGNMENT_FIELDS))
        if matched != geometry['strict']['correct']:
            raise ValueError('Whole-build translation does not match its measured overlap')
        return dict(yaw_quarters=quarters, translation=list(offset), geometry=geometry, scores=scores,
                    targets=target_breakdown(aligned, reference)), aligned

    fixed, fixed_ids = align(yaw_quarters) if yaw_quarters is not None else (None, {})
    relaxed, relaxed_ids = None, {}
    if relaxed_yaw:
        orientations = ((fixed, fixed_ids) if q == yaw_quarters else align(q) for q in range(4))
        relaxed, relaxed_ids = max(orientations, key=lambda result: (
            result[0]['geometry']['strict']['correct'],
            result[0]['yaw_quarters'] == yaw_quarters,
            -result[0]['yaw_quarters']))
    return dict(fixed=fixed, relaxed_yaw=relaxed, exported=len(predicted),
                visible_truth=len(reference.truth), required_truth=len(reference.truth) - len(reference.optional)), (
                    fixed_ids if fixed is not None else relaxed_ids)
