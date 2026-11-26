import unittest

import numpy as np

from benchmark.strip_evidence import gate


class StripEvidenceTest(unittest.TestCase):
    def test_tail_must_resemble_neighbor_more_than_its_own_surface(self):
        rgb = np.full((32, 32, 3), 65, np.uint8)
        body = np.zeros(rgb.shape[:2], bool)
        body[:10, 3:29] = True
        rgb[body] = 115
        tail = np.zeros_like(body)
        tail[10:30, 15:17] = True
        owned = body | tail
        np.testing.assert_array_equal(gate(rgb, owned, tail), tail)
        rgb[tail] = 115
        self.assertFalse(gate(rgb, owned, tail).any())
        rgb[:] = 75
        self.assertFalse(gate(rgb, owned, tail).any())
