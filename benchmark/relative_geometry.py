"""Compare visible voxel sets with a free origin and a fixed orientation and scale."""
from collections import Counter
from operator import index


PRIMARY_METRIC = 'summary.relative_geometry.f1'


def best_translation(predicted, target):
    """Return the integer offset to add to every prediction and the resulting overlap.

    Each target-minus-prediction difference casts one vote. For a given offset,
    its vote count equals the number of distinct occupied cells that coincide.
    Ties prefer the smallest squared displacement, then lexicographic XYZ.
    The target is scoring truth and this offset belongs only to the evaluator.
    """
    predicted = {(index(x), index(y), index(z)) for x, y, z in predicted}
    target = {(index(x), index(y), index(z)) for x, y, z in target}
    origin_overlap = len(predicted & target)
    if origin_overlap == min(len(predicted), len(target)):
        return (0, 0, 0), origin_overlap

    votes = Counter()
    for px, py, pz in predicted:
        votes.update((tx - px, ty - py, tz - pz) for tx, ty, tz in target)
    best_count = max(votes.values())
    if best_count == origin_overlap:
        return (0, 0, 0), best_count
    tied = (offset for offset, count in votes.items() if count == best_count)
    winner = min(tied, key=lambda xyz: (sum(v * v for v in xyz), xyz))
    return winner, best_count


def _aligned(cells, target):
    offset, _ = best_translation(cells, target)
    if offset == (0, 0, 0):
        return cells, offset
    dx, dy, dz = offset
    return {(x + dx, y + dy, z + dz) for x, y, z in cells}, offset


def _metrics(cells, before, truth, structure):
    target = truth.keys()
    correct = cells & target
    extras = cells - target
    added, removed = cells - before, before - cells
    return dict(target=len(target), placed=len(cells), correct=len(correct), extras=len(extras),
                missed=len(target - cells), hidden_extras=len(extras & structure),
                gained=len(added & target), lost=len(removed & target),
                extra_added=len(added - target), extra_removed=len(removed - target),
                hidden_added=len((added - target) & structure), hidden_removed=len((removed - target) & structure),
                joint_correct=sum(truth[p] == 'minecraft:smooth_stone' for p in correct))


def placement_scores(world, before, truth, structure):
    """Score decoded smooth-stone geometry in both the camera and best whole-build frames."""
    absolute = _metrics(world, before, truth, structure)
    aligned, offset = _aligned(world, truth)
    prior, prior_offset = (aligned, offset) if before == world else _aligned(before, truth)
    return dict(metrics=absolute, relative_geometry=_metrics(aligned, prior, truth, structure),
                translation=offset, baseline_translation=prior_offset)
