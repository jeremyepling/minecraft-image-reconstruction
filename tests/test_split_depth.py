import unittest

import numpy as np

from benchmark.split_depth import relative_depth


class PointDepthTest(unittest.TestCase):
    def test_recovers_relative_depth_after_unknown_forward_translation(self):
        h, w = 60, 90
        y, x = np.indices((h,w))
        u, v = (2*x+1-w)/np.hypot(w,h), (2*y+1-h)/np.hypot(w,h)
        z = np.where(x < 45, 5., 10.) + v
        points = np.stack((u*z/1.3, v*z/1.3, z-4.1),axis=-1).astype(np.float32)
        mask = np.ones_like(z)
        mask[:8] = 0
        actual = relative_depth(points,mask)
        np.testing.assert_allclose(actual[8:],1/z[8:],rtol=1.e-5)
        self.assertFalse(actual[:8].any())
