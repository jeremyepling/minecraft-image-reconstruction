import unittest

import numpy as np

from benchmark.boundary_guard import seam_evidence


class BoundaryGuardTest(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((12,24))
        self.left = x<12
        self.right = ~self.left
        self.proposal = (y>=2) & (y<10) & (x>=6) & (x<18)
        self.flat = (x.astype(float)*.02+y*.01)
        self.normal = np.array([0.,0.,1.])

    def test_tilt_and_segment_ends_do_not_invent_a_step(self):
        self.assertIsNone(seam_evidence(self.left,self.right,self.proposal,self.normal,self.normal,self.flat))

    def test_parallel_setback_has_step(self):
        found = seam_evidence(self.left,self.right,self.proposal,self.normal,self.normal,
                              self.flat+np.where(self.right, .2, 0.))
        self.assertAlmostEqual(abs(found['depth_step']), .2)
        self.assertEqual(found['sign_agreement'], 1.)

    def test_corner_with_continuous_depth_stays_separate(self):
        found = seam_evidence(self.left,self.right,self.proposal,
                              self.normal,np.array([1.,0.,0.]),self.flat)
        self.assertAlmostEqual(found['angle_deg'],90.)
        self.assertIsNone(seam_evidence(self.left,self.right,self.proposal & self.left,
                                        self.normal,np.array([1.,0.,0.]),self.flat))
