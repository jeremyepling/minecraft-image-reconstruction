import unittest
import numpy as np

from benchmark.strip_neighbor import select_receiver


class StripNeighborTests(unittest.TestCase):
    def setUp(self):
        self.tail = np.zeros((56,35), bool)
        self.tail[7:49,15:17] = True
        self.donor = self.tail.copy()
        self.donor[3:8,4:29] = True
        self.rgb = np.full((56,35,3), 80, np.uint8)
        self.wall = np.zeros_like(self.tail)
        self.wall[7:51,17:32] = True

    def select(self, receivers):
        return select_receiver(self.rgb, self.donor, self.tail, receivers, 3)

    def test_continuous_single_neighbor_preserves_ownership_distinction(self):
        choice = self.select([('wall', self.wall)])
        self.assertEqual(choice['receiver'], 'wall')
        self.assertEqual(choice['overlap_fraction'], 0)
        self.assertGreaterEqual(choice['near_fraction'], .90)

    def test_isolated_tail_and_duplicate_donor_are_kept(self):
        self.assertIsNone(self.select([]))
        self.assertIsNone(self.select([('copy', self.donor)]))

    def test_disconnected_votes_from_two_surfaces_cannot_combine(self):
        upper, lower = self.wall.copy(), self.wall.copy()
        upper[28:] = False
        lower[:28] = False
        self.assertIsNone(self.select([('a', upper), ('b', lower)]))

    def test_mismatching_wall_is_insufficient(self):
        self.rgb[self.wall] = 150
        self.assertIsNone(self.select([('wall', self.wall)]))
