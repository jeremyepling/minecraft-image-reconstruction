from itertools import product
import unittest

import numpy as np

from benchmark.floating_local_reprojection_projection import (
    _grade, _patch_cells, _project, fit_projection,
)


class LocalReprojectionTest(unittest.TestCase):
    def setUp(self):
        self.eye = np.array([14, 9, 17], float)
        forward = np.array([3, 3, 3], float) - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        rotation = np.stack([right, np.cross(forward, right), forward])
        intrinsic = np.array([[820., 0, 640], [0, 820, 360], [0, 0, 1]])
        linear = intrinsic @ rotation
        self.matrix = np.c_[linear, -linear @ self.eye]
        self.origins = {'front': [0, 0, 6], 'right': [6, 0, 0]}
        self.charts = [self.chart('front', [1, 2], 3), self.chart('right', [3, 2], 1)]

    def chart(self, key, axes, normal):
        origin = self.origins[key]
        direction = np.eye(3)[:, np.array(axes) - 1]
        h = np.c_[self.matrix[:, :3] @ direction, self.matrix[:, :3] @ origin + self.matrix[:, 3]]
        return dict(id=key, axes=axes, normal_axis=normal, H=h,
                    cells=[dict(u=u, v=v) for u, v in product(range(6), repeat=2)])

    def test_two_faces_recover_relative_projection_and_ignore_arbitrary_origin(self):
        sites = np.array([[3, 1, 6], [6, 5, 2], [2, 2, 4], [6, 4, 6]])
        expected, _ = _project(self.matrix, sites)
        for translation in (np.zeros(3, int), np.array([30, -19, 5])):
            origins = {key: np.array(xyz) + translation for key, xyz in self.origins.items()}
            matrix, eye, receipt = fit_projection(self.charts, origins, {}, (720, 1280))
            novel, _ = _project(matrix, sites + translation)
            np.testing.assert_allclose(novel, expected, atol=1e-7)
            np.testing.assert_allclose(eye, self.eye + translation, atol=1e-7)
            self.assertGreater(receipt['held_out'], 6)

    def test_single_plane_and_small_anchor_abstain(self):
        self.assertIsNone(fit_projection(self.charts[:1], self.origins, {}, (720, 1280)))
        excluded = {'front': [(u, v) for u, v in product(range(5), repeat=2)]}
        self.assertIsNone(fit_projection(self.charts, self.origins, excluded, (720, 1280)))

    def test_unsupported_and_hidden_moved_cells_are_rejected(self):
        front = self.charts[0]
        back = _patch_cells(front, self.origins['front'], [(3, 3)], 0)
        moved = _patch_cells(front, self.origins['front'], [(3, 3)], 1)
        shape = (180, 320)
        protected = np.full(shape, -1)
        empty = np.zeros(shape, bool)
        gradient = np.zeros((720, 1280), np.float32)
        args = (protected, protected), {}, empty, gradient, np.zeros(shape, np.float32)
        support = np.ones(shape, bool)
        self.assertTrue(_grade(moved, moved, 2, self.matrix, self.eye, support, *args)['eligible'])
        self.assertFalse(_grade(moved, moved, 2, self.matrix, self.eye, empty, *args)['eligible'])
        self.assertFalse(_grade(moved + back, back, 2, self.matrix, self.eye, support, *args)['eligible'])
