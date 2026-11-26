import unittest

import numpy as np

from benchmark.crop_guarded_fit import guarded_bounds


class GuardedFitTest(unittest.TestCase):
    def test_nine_tenths_inside_preserves_original_crop(self):
        uv = np.array([[0.5, 0.5]] * 90 + [[1.2, 0.5]] * 10)
        self.assertIsNone(guarded_bounds(uv))

    def test_outlier_ignored_and_other_axis_kept(self):
        uv = np.array([[0.5, 0.5]] * 68 + [[-0.25, 0.5]] * 130 + [[-8., 9.]])
        low, high = guarded_bounds(uv)
        np.testing.assert_allclose(low, [-0.25, 0])
        np.testing.assert_allclose(high, [1, 1])

    def test_both_sides_limited_independently(self):
        uv = np.array([[-3., 1.1], [4., 1.1]] * 20)
        low, high = guarded_bounds(uv)
        np.testing.assert_allclose(low, [-0.5, 0])
        np.testing.assert_allclose(high, [1.5, 1.1])
