import unittest

import numpy as np

from benchmark.attached_strip import appendages


class AttachedStripTest(unittest.TestCase):
    def test_only_terminal_ribbon_is_cut(self):
        mask = np.zeros((100,100),bool)
        mask[5:50,5:95] = True
        mask[50:83,30:33] = True
        mask[12:27,12:27] = False
        removed = appendages(mask,4)
        np.testing.assert_array_equal(removed, mask & (np.indices(mask.shape)[0]>=50))

    def test_thin_surface_is_preserved(self):
        mask = np.ones((3,120),bool)
        self.assertFalse(appendages(mask,4).any())

    def test_narrow_bridge_between_broad_areas_is_preserved(self):
        mask = np.zeros((70,90),bool)
        mask[10:60,:30] = True
        mask[10:60,65:] = True
        mask[30:33,30:65] = True
        self.assertFalse(appendages(mask,4).any())
