import unittest
from unittest.mock import patch

from engine import roof_span_texture as roof


class RoofSpanTextureTests(unittest.TestCase):
    def setUp(self):
        self.match = dict(split=.90, depth=.30, phase=.15, color=4, oriented=.9)

    def test_texture_runs_keep_unmeasured_hole_unknown(self):
        prior = {(0,4,0),(6,4,0)}
        cells = {(i,4,0) for i in (1,2,4,5)}
        rows = {(p,1):self.match for p in cells}
        self.assertEqual(roof.continuous(rows,prior,'guarded'), cells)

    def test_roof_extent_requires_an_observed_contact(self):
        cells = {(i,4,0) for i in range(3)}
        rows = {(p,1):self.match for p in cells}
        self.assertEqual(roof.continuous(rows,set(),'guarded'),set())
        self.assertEqual(roof.continuous(rows,{(1,3,0)},'guarded'),cells)

    def test_atlas_or_camera_absence_keeps_prior(self):
        prior = {(1,4,3)}
        with patch.object(roof,'_atlas',return_value=None):
            for camera in ({'P':None}, {'P':[]}):
                retained,_ = roof.recover(None,[{'support':True}],{},camera,prior)
                self.assertEqual(retained,prior)
