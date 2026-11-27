import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine.jagged_texture_join import refine
from engine.plane_consensus import row_runs


class JaggedTextureJoinTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB',(180,120),(88,88,88))
        y,x = np.indices((120,180))
        shared = (12<x)&(x<168)&(12<y)&(y<108)
        edge = x < 80+np.rint(15*np.sin(y/11))
        self.parts = []
        for name,mask in [('a',shared & edge),('b',shared & ~edge)]:
            self.parts.append(dict(id=name,normal_camera=[0.,0.,-1.],
                                   support=dict(size=[180,120],runs=row_runs(mask))))

    def test_missing_calibration_on_a_jagged_pair_returns_original_with_single_estimate(self):
        with patch('engine.visible_placement.seam_camera',return_value=dict(P=None)) as fit:
            result, joins, camera = refine(self.image,self.parts,'26.3')
            fit.assert_called_once()
            self.assertIs(result,self.parts)
            self.assertEqual(joins,[])
            result, joins, _ = refine(self.image,self.parts,'26.3',camera=camera)
            fit.assert_called_once()
            self.assertIs(result,self.parts)

    def test_no_camera_work_for_an_existing_junction(self):
        parts = [dict(self.parts[0],junction_split={'edge_support':.9}),self.parts[1]]
        with patch('engine.visible_placement.seam_camera',side_effect=AssertionError('unexpected fit')):
            result, joins, camera = refine(self.image,parts,'26.3')
        self.assertIs(result,parts)
        self.assertEqual(joins,[])
        self.assertIsNone(camera)
