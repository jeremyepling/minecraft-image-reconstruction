"""Analytical camera rays for the foreground conflict experiment."""
import unittest

import numpy as np

from engine.foreground_conflicts import first_hit


class ForegroundOwnershipTest(unittest.TestCase):
    def test_observed_tile_front_side_and_miss(self):
        eye = np.array([.5, .5, -3.])
        own = np.tile([.25, 0., 1.], (4, 1))
        probes = np.array([[.25, 0., 1.], [.15, 0., 1.], [-.10, 0., 1.]])
        hit = first_hit(own, probes, eye, 2, 0)
        np.testing.assert_allclose(hit[:2], [3., 10/3], atol=2e-6)
        self.assertTrue(np.isinf(hit[2]))
        self.assertTrue(np.isinf(first_hit(own[:3], probes, eye, 2, 0)).all())
