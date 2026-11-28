import unittest

import numpy as np

from benchmark.mobile_surface_maps import labels_by_pixel


class VisibleSurfaceMap(unittest.TestCase):
    def test_groups_first_hit_pixels_by_surface_instead_of_block_face(self):
        owner = np.array([[0, 1, 2], [3, 1, 2]])
        labels = {'faces': [
            {'id': 1, 'surface_id': 10, 'visible_pixels': 2},
            {'id': 2, 'surface_id': 10, 'visible_pixels': 2},
            {'id': 3, 'surface_id': 11, 'visible_pixels': 1},
        ]}
        np.testing.assert_array_equal(labels_by_pixel(labels, owner), [[0, 10, 10], [11, 10, 10]])
        labels['faces'][0]['visible_pixels'] = 1
        with self.assertRaises(ValueError):
            labels_by_pixel(labels, owner)
