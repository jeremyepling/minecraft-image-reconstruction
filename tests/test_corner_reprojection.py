import unittest

import numpy as np
from PIL import Image

from engine import corner_reprojection
from engine.plane_consensus import row_runs
from engine.projected_appearance import Scene, STEP
from engine.surface_model import intrinsic


class CornerReprojectionTests(unittest.TestCase):
    def setUp(self):
        size = (360, 240)
        self.image = Image.new('RGB', size)
        eye = np.array([-10., 5., -14.])
        forward = np.array([1., 2., 1.]) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        self.rotation = np.array([right, np.cross(right, forward), forward])
        self.camera = dict(P=intrinsic(size, 330) @ np.column_stack((self.rotation, -self.rotation @ eye)))
        corner = {(0, z) for z in range(4)} | {(x, 0) for x in range(4)}
        step = {(-1, z) for z in range(-1, 4)} | {(x, -1) for x in range(-1, 4)}
        self.flat = {(x, y, z) for y in range(5) for x,z in corner}
        self.stepped = {(x, y, z) for y in range(5) for x,z in (step if 0 < y < 4 else corner)}

    def observed_faces(self, real):
        scene = Scene(self.image, [], self.camera, real)
        parts = []
        for axis in range(3):
            visible = scene.axes.reshape(scene.shape) == axis
            mask = visible.repeat(STEP, axis=0).repeat(STEP, axis=1)
            parts.append(dict(id=str(axis), normal_camera=self.rotation[:, axis], support=dict(
                size=self.image.size, runs=row_runs(mask))))
        return parts

    def test_actual_projecting_middle_courses_are_preserved(self):
        self.assertEqual(len(corner_reprojection.proposals(self.stepped)), 1)
        result, evidence = corner_reprojection.recheck(self.image, self.observed_faces(self.stepped), self.camera, self.stepped)
        self.assertIsNone(evidence)
        self.assertEqual(result, self.stepped)

    def test_recover_only_a_corner_with_both_observed_faces(self):
        parts = self.observed_faces(self.flat)
        result, evidence = corner_reprojection.recheck(self.image, parts, self.camera, self.stepped)
        self.assertEqual(result, self.flat)
        self.assertEqual(evidence['wall_masks'], ['0', '2'])
        result, evidence = corner_reprojection.recheck(self.image, parts[1:], self.camera, self.stepped)
        self.assertIsNone(evidence)
        self.assertEqual(result, self.stepped)
