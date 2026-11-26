import unittest

import numpy as np
from PIL import Image

from engine.roof_span_contacts import rectified_bounds, recover, support_courses


class RoofContactTests(unittest.TestCase):
    def test_support_requires_separated_columns_with_two_observed_vertical_courses(self):
        points = [(0,0,0),(0,1,0),(3,0,0),(3,1,0),(5,0,0),(5,1,0)]
        owner = np.repeat(np.arange(len(points))[None,:],4,axis=0)
        axes = np.zeros_like(owner)
        axes[:,5] = 1
        oriented = np.ones((3,*axes.shape),dtype=bool)
        courses = support_courses(points,owner,axes,oriented)
        self.assertEqual({tuple(row) for row in courses[2]}, {(0,0),(3,0)})
        oriented[0] = False
        self.assertFalse(support_courses(points,owner,axes,oriented))

    def test_partial_diagonal_silhouette_cannot_establish_rectangle(self):
        x, y = np.meshgrid(np.linspace(.02,6.98,80),np.linspace(.02,.98,24))
        uv = np.column_stack((x.ravel(),y.ravel()))
        bounds = rectified_bounds(uv)
        np.testing.assert_equal(bounds[0],[0,0])
        np.testing.assert_equal(bounds[1],[7,1])
        self.assertIsNone(rectified_bounds(uv[uv[:,0]/7+uv[:,1]<1]))

    def test_input_without_estimated_camera_preserves_build(self):
        before={(0,0,0)}
        self.assertEqual(recover(Image.new('RGB',(64,32)),[],{},dict(P=None),before),before)
