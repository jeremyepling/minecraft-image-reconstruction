from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from engine import roof_height_atlas as height
from engine.thin_atlas_faces import _mips


class WholePatchHeightTests(TestCase):
    def test_same_photo_samples_choose_the_generating_integer_height(self):
        rng=np.random.default_rng(31)
        tiles=rng.integers(40,220,size=(32,16,16,3),dtype=np.uint8)
        atlas=SimpleNamespace(tiles=tiles,means=tiles.mean(axis=(1,2)))
        maps=_mips(atlas)
        eye=np.array([1.3,8.6,-4.3])
        coords=rng.uniform((-4,0),(5,9),size=(2000,2))
        rays=np.column_stack((coords[:,0]-eye[0],np.full(len(coords),4-eye[1]),coords[:,1]-eye[2]))
        xy=np.floor(16*coords).astype(int)%16
        colors=tiles[3,xy[:,1],xy[:,0]]
        result=height.rank_planes(coords[:,0],coords[:,1],rays,colors,eye,range(1,8),atlas,maps)
        self.assertEqual(result[0]['plane_y'],4)
        self.assertEqual(result[0]['sprite'],3)
        self.assertGreater(result[0]['value']-result[1]['value'],.5)

    def test_flat_patch_has_no_plane_evidence(self):
        x=np.arange(100)
        result=height.rank_planes(x,x,np.zeros((100,3)),np.ones((100,3)),None,[1,2],None,None)
        self.assertEqual(result,[])

    def test_missing_calibration_preserves_prior_and_abstains(self):
        prior={(2,3,4)}
        with patch.object(height,'_atlas',return_value=None):
            for camera in ({'P':None},{'P':np.eye(3,4)}):
                result,report=height.recover(None,[{}],None,camera,prior)
                self.assertEqual(result,prior)
                self.assertIsNone(report['primary_height'])
