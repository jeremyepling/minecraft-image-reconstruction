import itertools
import random
import unittest

from benchmark.relative_geometry import best_translation, placement_scores
from benchmark.visible_depth_score import aggregate


def translate(cells, offset):
    dx, dy, dz = offset
    return {(x + dx, y + dy, z + dz) for x, y, z in cells}


class RelativeGeometryTests(unittest.TestCase):
    def test_whole_disconnected_build_can_move_any_distance(self):
        target = {(0, 0, 0), (0, 1, 0), (0, 2, 0), (6, 2, 1), (7, 2, 1)}
        offset = (27000017, -240, 19000031)
        predicted = translate(target, offset)
        self.assertEqual(best_translation(predicted, target),
                         (tuple(-v for v in offset), len(target)))

    def test_moving_roof_alone_loses_internal_spacing(self):
        pillars = {(x, y, 0) for x in (0, 4) for y in (0, 1, 2)}
        roof = {(x, 3, 0) for x in range(5)}
        target = pillars | roof
        predicted = pillars | translate(roof, (0, 0, 1))
        self.assertEqual(best_translation(predicted, target), ((0, 0, 0), 6))

    def test_primary_benchmark_honors_one_shared_origin_for_roof_and_pillars(self):
        pillars = {(x, y, 0) for x in (0, 4) for y in (0, 1, 2)}
        roof = {(x, 3, 0) for x in range(5)}
        truth = dict.fromkeys(pillars | roof, 'minecraft:smooth_stone')
        original = set(truth)
        shifted = translate(original, (0, 0, 1))
        partial = pillars | translate(roof, (0, 0, 1))
        whole = placement_scores(shifted, original, truth, original)
        roof_only = placement_scores(partial, shifted, truth, original)
        self.assertEqual(whole['metrics']['correct'], 0)
        self.assertEqual(whole['relative_geometry']['correct'], 11)
        self.assertEqual(whole['relative_geometry']['lost'], 0)
        self.assertEqual(roof_only['relative_geometry']['lost'], 5)
        self.assertEqual(roof_only['relative_geometry']['extras'], 5)
        rows = [dict(camera_supported=True, **s) for s in (whole, roof_only)]
        self.assertEqual(aggregate(rows)['relative_geometry']['f1'], 17 / 22)

    def test_extra_and_missing_cells_remain_errors_after_alignment(self):
        target = {(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)}
        imperfect = (target - {(1, 0, 0)}) | {(0, 9, 5)}
        predicted = translate(imperfect, (-18, 22, -35))
        self.assertEqual(best_translation(predicted, target), ((18, -22, 35), 3))

    def test_equal_overlap_uses_shortest_offset_then_lexicographic(self):
        self.assertEqual(best_translation([(0, 0, 0)], [(7, 0, 0), (0, 0, 0)]),
                         ((0, 0, 0), 1))
        self.assertEqual(best_translation([(0, 0, 0)], [(1, 0, 0), (-1, 0, 0), (-2, 0, 0)]),
                         ((-1, 0, 0), 1))

    def test_duplicate_coordinates_and_empty_inputs(self):
        self.assertEqual(best_translation([(3, 4, 5)] * 3, [(5, 2, 0)] * 2), ((2, -2, -5), 1))
        for predicted, target in [([], []), ([(1, 2, 3)], []), ([], [(1, 2, 3)])]:
            self.assertEqual(best_translation(predicted, target), ((0, 0, 0), 0))

    def test_sparse_vote_matches_exhaustive_search_on_small_shapes(self):
        rng = random.Random(3192)
        universe = list(itertools.product(range(-1, 2), repeat=3))
        offsets = list(itertools.product(range(-2, 3), repeat=3))
        for _ in range(40):
            predicted = set(rng.sample(universe, rng.randrange(10)))
            target = set(rng.sample(universe, rng.randrange(10)))
            overlap = {d: len(translate(predicted, d) & target) for d in offsets}
            best = min(offsets, key=lambda d: (-overlap[d], sum(v*v for v in d), d))
            self.assertEqual(best_translation(predicted, target), (best, overlap[best]))
