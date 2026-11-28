import unittest

import numpy as np

from benchmark.floating_peer_ruler_refit import line_score, perspective, project


class PeerRulerTest(unittest.TestCase):
    def test_pivot_and_local_jacobian_survive_perspective_search(self):
        h=np.array([[12.,1.,100.],[1.,-10.,200.],[.01,.02,1.]])
        pivot=np.array([7.,2.])
        eps=1e-6
        for strength in (0.,.5,1.,1.5):
            candidate=perspective(h,pivot,strength)
            np.testing.assert_allclose(project(candidate,np.zeros(2)),project(h,pivot))
            np.testing.assert_allclose(project(candidate,eps*np.eye(2)),project(h,pivot+eps*np.eye(2)))
        np.testing.assert_allclose(project(perspective(h,pivot,1.),np.array([[2.,4.],[-3.,5.]])),
                                   project(h,pivot+np.array([[2.,4.],[-3.,5.]])))

    def test_held_out_phase_can_disagree_with_fitting_partition(self):
        uv=np.arange(0.,20.,.125)
        signal=np.cos(2*np.pi*uv/3.)
        profile=(uv,[(4+signal,np.full(len(uv),40)),(4-signal,np.full(len(uv),40))])
        fit,_=line_score(profile,3.,np.array([0.]),0)
        far,_=line_score(profile,3.,np.array([0.]),1)
        self.assertGreater(fit[0],0.)
        self.assertLess(far[0],0.)
