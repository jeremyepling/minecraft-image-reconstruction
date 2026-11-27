import unittest

import numpy as np
from PIL import Image

from engine.plane_consensus import row_runs
from engine.pillar_seams import native_mask
from engine.skyline_extent import refine


class SkylineExtentTests(unittest.TestCase):
    def setUp(self):
        self.size = (64, 48)
        self.sky = np.ones(self.size[::-1], bool)
        self.sky[10:32, 8:55] = False
        rgb = np.full((*self.sky.shape, 3), [152, 191, 255], dtype=np.uint8)
        rgb[~self.sky] = [102, 98, 100]
        self.image = Image.fromarray(rgb)
        self.mask = np.zeros_like(self.sky)
        self.mask[12:32, 8:55] = True

    def part(self, key, mask, normal=(.47, -.10, -.87)):
        return dict(id=key, normal_camera=list(normal), support=dict(size=list(self.size), runs=row_runs(mask)))

    def test_observed_wall_only_and_opening_preserved(self):
        self.mask[20:23, 17:21] = False
        part = self.part('wall', self.mask)
        result, events = refine(self.image, [part], self.sky)
        revised = native_mask(result[0], self.size)
        expected = self.mask.copy()
        expected[10:12, 8:55] = True
        np.testing.assert_array_equal(revised, expected)
        self.assertEqual(events, [dict(surface='wall', added_pixels=94)])
        np.testing.assert_array_equal(native_mask(part, self.size), self.mask)

    def test_conflicting_owner_blocks_extension(self):
        parts = [self.part('wall', self.mask), self.part('occluder', self.mask)]
        result, events = refine(self.image, parts, self.sky)
        self.assertIs(result, parts)
        self.assertFalse(events)

    def test_ambiguous_view_or_color_keeps_original(self):
        wall = self.part('wall', self.mask)
        for photo, part, sky in ((self.image, self.part('wall', self.mask, (.5,.1,-.85)), self.sky),
                                 (self.image, wall, np.zeros_like(self.sky)),
                                 (self.image.copy(), wall, self.sky)):
            if photo is not self.image:
                photo.paste((75, 75, 75), (8, 10, 55, 12))
            parts = [part]
            result, events = refine(photo, parts, sky)
            self.assertIs(result, parts)
            self.assertFalse(events)
