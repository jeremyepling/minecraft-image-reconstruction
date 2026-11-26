"""Compare visible voxel sets with a free origin and a fixed orientation and scale."""
from collections import Counter
from operator import index


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
