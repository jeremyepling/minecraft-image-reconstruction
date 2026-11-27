import unittest

import numpy as np

from benchmark.expanded_scores import FrameReference
from benchmark.relative_face_scoring import evaluate_relative_voxels, yaw_from_anchor_normal


STONE = 'minecraft:stone'
BRICKS = 'minecraft:bricks'
TRUTH = {
    (21, 10, 0): STONE, (21, 11, 0): STONE, (21, 12, 0): STONE,
    (22, 10, 0): BRICKS, (23, 10, 0): BRICKS, (22, 10, 1): BRICKS,
    (26, 14, 2): BRICKS, (27, 14, 2): BRICKS, (26, 14, 3): BRICKS,
}


def _reference(optional=frozenset()):
    return FrameReference(TRUTH, optional, {}, {'full_cube': set(TRUTH)}, {})


def _pose():
    # Project world +X right, +Y up and +Z forward with an arbitrary focal length.
    projection = np.diag([1.3, 1.6, 1., 0.])
    projection[3, 2] = 1.
    return dict(image_size=[1440, 900], view_rotation_projection=projection.T.ravel().tolist())


class RelativeFaceScoringTests(unittest.TestCase):
    def test_translates_signed_voxels_together_and_retains_exact_ids(self):
        predicted = {(x - 37, y - 18, z - 20): name for (x, y, z), name in TRUTH.items()}
        row, aligned = evaluate_relative_voxels(predicted, _reference(), yaw_quarters=0, assigned=set(predicted))
        self.assertEqual(row['fixed']['translation'], [37, 18, 20])
        self.assertEqual(row['fixed']['geometry']['strict']['f1'], 1)
        self.assertEqual(row['fixed']['scores']['strict']['f1'], 1)
        self.assertEqual(row['fixed']['scores']['assigned_correct'], len(TRUTH))
        self.assertEqual(aligned, TRUTH)

    def test_offsets_between_disconnected_parts_are_scored(self):
        predicted = {(x - 37, y - 18 + (x >= 26), z - 20): name for (x, y, z), name in TRUTH.items()}
        row, _ = evaluate_relative_voxels(predicted, _reference(), yaw_quarters=0)
        geometry = row['fixed']['geometry']['strict']
        self.assertEqual((geometry['correct'], geometry['missed'], geometry['extras']), (6, 3, 3))
        self.assertEqual(row['fixed']['translation'], [37, 18, 20])

    def test_relaxed_orientation_is_separate_and_has_no_reflection(self):
        predicted = {(z - 5, y - 15, -x + 2): name for (x, y, z), name in TRUTH.items()}
        row, _ = evaluate_relative_voxels(predicted, _reference(), yaw_quarters=0, relaxed_yaw=True)
        self.assertLess(row['fixed']['geometry']['strict']['f1'], 1)
        self.assertEqual(row['relaxed_yaw']['yaw_quarters'], 1)
        self.assertEqual(row['relaxed_yaw']['geometry']['strict']['f1'], 1)
        reflected = {(-x, y, z): name for (x, y, z), name in TRUTH.items()}
        row, _ = evaluate_relative_voxels(reflected, _reference(), relaxed_yaw=True)
        self.assertIsNone(row['fixed'])
        self.assertLess(row['relaxed_yaw']['geometry']['strict']['f1'], 1)

    def test_facade_neutral_position_keeps_strict_id_penalty(self):
        predicted = dict(TRUTH)
        predicted[(21, 10, 0)] = BRICKS
        predicted[(26, 13, 2)] = BRICKS
        row, _ = evaluate_relative_voxels(predicted, _reference({(21, 10, 0)}), yaw_quarters=0)
        self.assertEqual(row['fixed']['geometry']['strict']['extras'], 1)
        self.assertEqual(row['fixed']['scores']['strict']['extras'], 2)
        self.assertEqual(row['fixed']['scores']['facade']['extras'], 1)
        self.assertEqual(row['fixed']['scores']['facade']['missed'], 0)

    def test_anchor_uses_its_signed_camera_normal_and_can_be_ambiguous(self):
        match = yaw_from_anchor_normal([2, -.05, .1], [0, 0, 1], _pose())
        self.assertEqual(match['yaw_quarters'], 3)
        self.assertEqual(match['best_world_normal'], [1, 0, 0])
        match = yaw_from_anchor_normal([1, 0, 1], [0, 0, 1], _pose())
        self.assertIsNone(match['yaw_quarters'])
        self.assertAlmostEqual(match['cosine_margin'], 0)


if __name__ == '__main__':
    unittest.main()
