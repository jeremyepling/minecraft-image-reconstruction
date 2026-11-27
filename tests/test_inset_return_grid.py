import unittest

import numpy as np

from engine.inset_returns import _brightness, _fit, _labels


class InsetReturnGridTest(unittest.TestCase):
    def setUp(self):
        u,v = np.meshgrid(np.arange(-1.,5.01,.045),np.arange(-1.,4.01,.045))
        self.front=np.stack((u,v,np.zeros_like(u)),axis=-1)
        self.eye=np.array([-5.,7.,-12.])
        self.edges=(0,4,0,3)
        self.depth=2
        self.labels=_labels(self.front,self.eye,2,0,self.depth,self.edges)
        self.host=(self.labels==0)|(self.labels==3)
        self.side=self.labels==1
        self.sill=self.labels==2

    def test_grid_fits_two_observed_returns_at_the_same_depth(self):
        fit=_fit(self.front,self.eye,2,0,self.host,self.side,self.sill)
        self.assertIsNotNone(fit)
        self.assertEqual(fit['depth'], 2)
        self.assertEqual(fit['rim'],list(self.edges))
        self.assertEqual(fit['side_iou'],1.)
        self.assertEqual(fit['sill_iou'],1.)
        rgb=np.array([[150,150,150],[91,91,91],[78,78,78],[113,113,113]],np.float32)[self.labels]
        self.assertTrue(_brightness(rgb,self.front,self.eye,2,fit))
        self.assertFalse(_brightness(np.full_like(rgb,120.),self.front,self.eye,2,fit))

    def test_inset_requires_solid_visible_back_support_and_both_return_directions(self):
        self.assertIsNone(_fit(self.front,self.eye,2,0,self.labels==0,self.side,self.sill))
        self.assertIsNone(_fit(self.front,self.eye,2,0,self.host,self.side,np.zeros_like(self.sill)))

    def test_extruding_outward_is_a_competing_hypothesis(self):
        shape=_labels(self.front,self.eye,2,0,-2,self.edges)
        self.assertIsNone(_fit(self.front,self.eye,2,0,(shape==0)|(shape==3),shape==1,shape==2))
