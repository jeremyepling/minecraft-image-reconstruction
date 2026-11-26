import unittest
import numpy as np

from PIL import Image

from engine.plane_consensus import row_runs
from engine.strip_neighbor import clean_surfaces, select_receiver
from engine.surface_perimeter import _native_alpha


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

    def test_shipped_cleanup_records_removed_pixels_and_leaves_neighbor_alone(self):
        size = (120, 900)
        donor = np.zeros(size[::-1], bool)
        donor[20:80, 10:100] = True
        donor[80:132, 42:45] = True
        receiver = np.zeros_like(donor)
        receiver[80:150, 45:100] = True
        rgb = np.full((*donor.shape, 3), 60, np.uint8)
        rgb[20:80, 10:100] = 190
        parts = [dict(id=name, support=dict(size=list(size), runs=row_runs(mask)))
                 for name, mask in [('beam', donor), ('side', receiver)]]
        result, stats = clean_surfaces(Image.fromarray(rgb), parts)
        self.assertEqual(stats['changed_surfaces'], 1)
        self.assertEqual(stats['removed_pixels'], 3*52)
        self.assertEqual(stats['decisions'][0]['receiver'], 'side')
        self.assertEqual(stats['decisions'][0]['overlap_fraction'], 0.)
        self.assertIs(result[1], parts[1])
        alpha, box = _native_alpha(result[0]['support'], size)
        self.assertEqual(box, (10, 20, 100, 80))
        self.assertEqual(np.count_nonzero(alpha), 90*60)
