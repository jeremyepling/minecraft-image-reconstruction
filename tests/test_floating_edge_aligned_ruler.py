import unittest
import numpy as np
from benchmark.floating_edge_aligned_ruler import aligned,vanishing,project


class EdgeAlignedRulerTest(unittest.TestCase):
    def test_distinct_parallel_lines_fit_an_affine_direction(self):
        rows=[dict(ends=[[0.,y],[150.,y]],weight=150.) for y in (20.,60.,100.,140.)]
        p,detail=vanishing(rows,np.array([75.,80.]),150.)
        self.assertTrue(detail["accepted"])
        self.assertLess(abs(p[2]),1e-6)
        self.assertLess(abs(p[1]),1e-6)

    def test_near_duplicate_lines_do_not_define_wall_perspective(self):
        rows=[dict(ends=[[0.,y],[150.,y]],weight=150.) for y in (20.,21.,22.,23.)]
        p,detail=vanishing(rows,np.array([75.,80.]),150.)
        self.assertIsNone(p)
        self.assertFalse(detail["accepted"])

    def test_alignment_keeps_anchor_and_applies_its_vanishing_points(self):
        h=np.array([[15.,2.,100.],[1.,-18.,50.],[.01,.02,1.]])
        u=np.array([2000.,100.,1.])
        v=np.array([0.,-1.,0.])
        center=np.array([.5,.5])
        new=aligned(h,[u,v],center)
        np.testing.assert_allclose(project(new,center),project(h,center))
        np.testing.assert_allclose(np.cross(new[:,0],u),0,atol=1e-8)
        np.testing.assert_allclose(np.cross(new[:,1],v),0,atol=1e-8)
