import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import local_panel_depth as panel


class LocalPanelDepthTest(unittest.TestCase):
    def test_removing_two_neighbors_restores_third_first_hit(self):
        class Scene:
            cells=[(0,0,i) for i in (2,3,4)]
            owner=np.array([0,0,-1])
            axes=np.array([2,2,-1])
            dep=np.array([.5,.5,0.])
            def faces(self,c):
                rays=np.array([0,1] if c[2]==2 else [0])
                return rays,np.full(len(rays),2),np.full(len(rays),1/c[2])
        scene=Scene()
        owner,axes,z=panel.Stack(scene).uncover(scene,np.array([0,1,2]),[0,1])
        np.testing.assert_array_equal(owner,[2,-1,-1])
        np.testing.assert_array_equal(axes,[2,-1,-1])
        np.testing.assert_array_equal(z,[.25,0.,0.])

    def test_missing_or_unverified_atlas_keeps_prior(self):
        prior={(0,0,1),(0,1,1)}
        image=Image.new('RGB',(16,16))
        with patch.object(panel,'trials',side_effect=AssertionError('unverified input')):
            with patch.object(panel,'_atlas',return_value=None):
                self.assertEqual(panel.recover(image,[{}],{},dict(P=np.eye(3)),prior)[0],prior)
            with patch.object(panel,'_atlas',return_value=SimpleNamespace(provenance={'atlas_sha256':'wrong'})):
                self.assertEqual(panel.recover(image,[{}],{},dict(P=np.eye(3)),prior)[0],prior)

    def test_unresolved_camera_skips_even_atlas_preparation(self):
        prior={(0,0,1),(0,1,1)}
        with patch.object(panel,'_atlas',side_effect=AssertionError('unexpected atlas load')):
            self.assertEqual(panel.recover(None,[{}],{},dict(P=None),prior)[0],prior)
