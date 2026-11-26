import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine.visible_cell_decision import FEATURES, cell_features
from engine.visible_floor_evidence import floor_features
from engine.visible_ray_filter import fully_occluded_cells


class CellEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(96,96),'gray')
        focal=75
        k=np.array([[focal,0,48],[0,focal,48],[0,0,1]])
        axes=np.diag([1,-1,1])
        eye=np.array([.5,.5,-5])
        self.P=k @ np.column_stack((axes,-axes@eye))
        self.cells={(0,0,0),(0,0,1)}
        self.mask=np.ones((24,24),dtype=bool)

    def test_front_cell_owns_rays_through_occluded_candidate(self):
        parts=[dict(normal_camera=[0.,0.,-1.])]
        xyz,support,_=cell_features(self.image,parts,self.P,self.cells,masks=[self.mask],support_only=True)
        self.assertEqual(xyz.tolist(),[[0,0,0],[0,0,1]])
        self.assertGreater(support[0],.7)
        self.assertEqual(support[1],0)
        with patch('engine.visible_cell_decision._atlas',return_value=None):
            _,all_features,available=cell_features(self.image,parts,self.P,self.cells,masks=[self.mask])
        self.assertFalse(available)
        np.testing.assert_array_equal(support,all_features[:,FEATURES.index('visible_matching_mask')])

    def test_wrong_axis_cannot_support_the_front_face(self):
        _,support,_=cell_features(self.image,[dict(normal_camera=[0.,1.,0.])],self.P,self.cells,
                                  masks=[self.mask],support_only=True)
        np.testing.assert_array_equal(support,np.zeros(2))

    def test_native_raster_culls_a_back_block_and_preserves_front_block(self):
        self.assertEqual(fully_occluded_cells(self.image.size,self.P,self.cells),{(0,0,1)})
        self.assertFalse(fully_occluded_cells(self.image.size,self.P,{(0,0,0)}))

    def test_floor_evidence_abstains_without_sufficient_observed_reference(self):
        self.assertTrue(np.all(floor_features(self.image,self.P,self.cells)==0))
        self.assertEqual(floor_features(self.image,None,()).shape,(0,4))
