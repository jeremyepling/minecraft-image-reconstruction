"""Geometry proposal controls for joined shadows and low-frequency illumination."""
import unittest

import numpy as np
from engine.entrance_regions import boxes


class EntranceRegionTests(unittest.TestCase):
    def test_lintels_separate_openings_touching_in_shadow(self):
        cells=np.full((9,15),150,np.uint8)
        cells[3:7,2:5]=80
        cells[3:7,7:10]=82
        cells[6,5:7]=88
        field=cells.repeat(20,axis=0).repeat(20,axis=1)
        rect=np.repeat(field[...,None],3,axis=2)
        predicted=[(tuple(lo),tuple(hi)) for lo,hi in boxes(rect,20)]
        self.assertIn(((2,3),(5,7)),predicted)
        self.assertIn(((7,3),(10,7)),predicted)

    def test_uniform_plane_has_no_entrances(self):
        self.assertEqual(boxes(np.full((128,256,3),82,np.uint8),16),[])
