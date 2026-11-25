import unittest
import numpy as np

from engine.remap import elevation
from engine.boundary_evidence import boundary_evidence


class RemapTests(unittest.TestCase):
    def test_front_elevation_keeps_openings_and_disconnected_pillars(self):
        mask=np.zeros((160,200),bool)
        mask[32:48,32:144]=True
        mask[48:112,48:64]=True
        mask[48:112,112:128]=True
        _,cells=elevation(np.zeros((*mask.shape,3),np.uint8),mask,(16,16))
        solid={tuple(c['xyz']) for c in cells}
        self.assertEqual(len(solid),15)
        self.assertEqual({x for x,y,z in solid if y==0},{1,5})
        self.assertEqual({x for x,y,z in solid if y==4},set(range(7)))

    def test_no_evidence_between_separate_cells(self):
        cells=[dict(plane='p',ij=[0,0],facing='south'),dict(plane='p',ij=[2,0],facing='south')]
        self.assertEqual(boundary_evidence(None,cells),[])
