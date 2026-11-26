import unittest
import numpy as np
from PIL import Image

from engine.neighbor_photo_faces import recover


class NeighborPhotoTests(unittest.TestCase):
    def setUp(self):
        # Looking along -z at the front faces of two adjacent cubes, each 20 px wide.
        self.projection = np.array([[80, 0, -60, 220], [0, -80, -45, 265], [0, 0, -1, 5]])
        self.base = {(0, 0, 0)}
        self.region = [{'support': {'size': [120, 90], 'runs': [[y, 40, 80] for y in range(35,55)]}}]
        self.rgb = np.zeros((90,120,3), np.uint8)
        patch = np.random.default_rng(349).integers(30, 210, size=(20,20,3), dtype=np.uint8)
        self.rgb[35:55,40:60] = patch
        self.rgb[35:55,60:80] = patch

    def test_neighbor_has_visible_repeated_phase(self):
        self.assertEqual(recover(Image.fromarray(self.rgb), self.region, self.projection, self.base),
                         {(1, 0, 0)})

    def test_same_color_and_occluded_texture_do_not_supply_a_phase_match(self):
        self.rgb[35:55,40:80] = 86
        self.assertEqual(recover(Image.fromarray(self.rgb), self.region, self.projection, self.base), set())

    def test_pixels_beyond_the_observed_mask_supply_no_target(self):
        self.region[0]['support']['runs'] = [[y, 40, 60] for y in range(35,55)]
        self.assertEqual(recover(Image.fromarray(self.rgb), self.region, self.projection, self.base), set())
