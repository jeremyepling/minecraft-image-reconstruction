import unittest

import numpy as np

from benchmark.boundary_block_seam import boundary_phase, predict


class BoundaryPhaseTest(unittest.TestCase):
    def test_parallel_edges_predict_fractional_origin(self):
        mask = np.zeros((40,100),bool)
        mask[4:36,5:85] = True
        self.assertAlmostEqual(boundary_phase(mask,5),.25)

    def test_crop_edge_or_single_edge_cannot_anchor_grid(self):
        mask = np.ones((40,100),bool)
        self.assertIsNone(boundary_phase(mask,5))
        mask[:,:5] = False
        self.assertIsNone(boundary_phase(mask,5))

    def test_preserves_accepted_phase_and_missing_pitch(self):
        sample = dict(mask=np.zeros((40,100),bool))
        base = dict(counts=[5,None],phase=[.25,None])
        self.assertEqual(predict(sample,base),base)
