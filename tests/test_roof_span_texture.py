import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from engine import roof_span_texture as roof
from benchmark.mask_blocks import _write_read


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

    def test_ray_candidates_round_trip_through_minecraft_export(self):
        shape=(8,8)
        ctx=SimpleNamespace(union=np.ones(shape,bool),axes=np.zeros(shape,int),
                            masks=np.ones((3,*shape),bool),depth=np.zeros(shape),eye=np.array([.5,2.,.5]),
                            _rays=lambda x,y: np.tile([0.,-1.,0.],(len(x),1)))
        cells={c for c,axis in roof.visible_candidates(ctx,{(1,0):{(0,0,0)}})}
        self.assertEqual(cells,{(0,0,0)})
        with TemporaryDirectory() as temp:
            self.assertEqual(_write_read(Path(temp)/'roof.zip',cells),cells)
