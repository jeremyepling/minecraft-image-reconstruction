"""Image-only local spacing regressions."""
import unittest

import numpy as np

from engine.local_pitch import estimate_pitch


class LocalPitchTests(unittest.TestCase):
    def test_repeat_in_an_unaligned_crop(self):
        tex = np.random.default_rng(720).integers(30,210,(16,16,3),dtype=np.uint8)
        wall = np.tile(tex,(8,9,1)).repeat(3,axis=0).repeat(3,axis=1)
        crop = wall[17:276,9:323]
        pitch = estimate_pitch(crop,np.ones(crop.shape[:2],dtype=bool))
        np.testing.assert_allclose(pitch,(48,48),rtol=.04)

    def test_empty_and_uniform_regions_abstain(self):
        rgb = np.zeros((90,110,3),np.uint8)
        for occupied in (False,True):
            self.assertIsNone(estimate_pitch(rgb,np.full(rgb.shape[:2],occupied)))
