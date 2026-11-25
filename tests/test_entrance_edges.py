"""Unsigned lintel contrast and bounded vertical jambs seed the depth stage."""
import unittest
import numpy as np
from engine.entrance_edges import boxes


class EntranceTests(unittest.TestCase):
    def test_bright_and_dark_rims(self):
        for wall,insert in ((48,180),(180,48)):
            with self.subTest(wall=wall):
                rgb=np.full((180,260,3),wall,np.uint8)
                rgb[45:135,70:160]=insert
                found=boxes(rgb,30)
                self.assertEqual(len(found),1)
                lo,hi=found[0]
                np.testing.assert_allclose(lo*30,[70,45],atol=3)
                np.testing.assert_allclose(hi*30,[160,135],atol=3)

    def test_continuous_stripes_have_no_jambs(self):
        rgb=np.full((180,260,3),110,np.uint8)
        rgb[45:135]=50
        self.assertEqual(boxes(rgb,30),[])
