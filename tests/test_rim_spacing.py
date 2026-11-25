import unittest

import numpy as np

from engine.rim_spacing import local_spacing, texture_detail


class LocalSpacingTests(unittest.TestCase):
    def test_repeat_comes_from_wall_outside_inlay(self):
        rng=np.random.default_rng(27)
        # One block repeats every 34 horizontal and 35 vertical pixels.
        sprite=rng.integers(0,255,(35,34,3),dtype=np.uint8)
        rect=np.tile(sprite,(8,9,1))
        rect[70:175,102:204]=19
        wall=np.ones(rect.shape[:2],bool)
        wall[65:180,97:209]=False
        scales=local_spacing(texture_detail(rect),wall,np.array([102,70]),np.array([204,175]),36)
        for record,expected in zip(scales,(34,35)):
            self.assertAlmostEqual(record['ratio']*36,expected,delta=.15)

    def test_textureless_wall_abstains(self):
        rect=np.ones((160,180,3),np.uint8)*90
        found=local_spacing(texture_detail(rect),np.ones(rect.shape[:2],bool),
                            np.array([48,48]),np.array([114,114]),36)
        self.assertEqual(found,[None,None])
