import unittest

import numpy as np

from engine.cube_faces import fit_face


class CubeFaceTest(unittest.TestCase):
    def test_oblique_right_angles_keep_their_projected_shape(self):
        # Two vanishing directions meet the actual edges of this trapezoid.
        quad = np.array([[60.,30.], [150.,55.], [138.,120.], [54.,103.]])
        corners = np.column_stack((quad, np.ones(4)))
        sides = np.cross(corners, np.roll(corners,-1,axis=0))
        axes = [np.cross(sides[0],sides[2]),np.cross(sides[1],sides[3]),[1.,-1.,0.]]
        fitted, evidence = fit_face(quad, axes)
        np.testing.assert_allclose(fitted, quad)
        self.assertEqual(evidence['third_axis'], 2)

    def test_parallel_axes_and_inconsistent_corner(self):
        axes = [[1,0,0],[0,1,0],[230,60,1]]
        face = np.array([[25,25],[145,25],[145,135],[25,135]])
        self.assertIsNotNone(fit_face(face,axes))
        face[2] = [110,115]
        self.assertIsNone(fit_face(face,axes))
