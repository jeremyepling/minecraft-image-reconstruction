import unittest

import numpy as np

from engine.secondary_plane_depth import adjust, supported_face, wrap_selector
from engine.visible_placement import _sample_face


class SecondaryDepthTests(unittest.TestCase):
    def test_new_cell_requires_full_mask_support_at_raster_samples(self):
        P=np.array([[80.,0,64,0],[0,80,64,0],[0,0,1,0]])
        mask=np.zeros((32,32),bool)
        eye=np.zeros(3)
        cell=(0,0,4)
        box,pixels,*_=_sample_face(P,cell,2,-1,mask.shape)
        self.assertFalse(supported_face(cell,2,4,eye,P,mask))
        mask[box]|=pixels
        self.assertTrue(supported_face(cell,2,4,eye,P,mask))
        y,x=np.argwhere(pixels)[0]
        mask[box][y,x]=False
        self.assertFalse(supported_face(cell,2,4,eye,P,mask))
        self.assertFalse(supported_face((0,-1,4),2,4,eye,P,np.ones_like(mask)))

    def test_missing_atlas_and_nonfinite_geometry_keep_original_cells(self):
        choose=object()
        self.assertEqual(wrap_selector(choose,None,None),(choose,None))
        cells={(1,2,3)}
        self.assertEqual(adjust(cells,None,np.full((3,4),np.nan),np.zeros(3),None,2,3,None,object()),(cells,0))
