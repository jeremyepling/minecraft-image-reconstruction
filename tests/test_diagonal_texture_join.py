import unittest

import numpy as np
from PIL import Image

from engine.diagonal_texture_join import contact_pixels, diagonal_span, refine


class DiagonalTextureTest(unittest.TestCase):
    def test_missing_or_malformed_camera_keeps_original_parts(self):
        image = Image.new('RGB', (60, 60))
        parts = [dict(id='wall')]
        for camera in (None, {}, {'P': 'unavailable'}, {'P': [[0] * 4] * 3},
                       {'P': np.full((3, 4), np.nan)}):
            unchanged, groups = refine(image, parts, camera)
            self.assertIs(unchanged, parts)
            self.assertEqual(groups, [])

    def test_cropped_masks_with_gap_or_duplicate_area(self):
        mask = np.ones((40, 45), bool)
        a = (mask, (3, 3, 48, 43))
        self.assertIsNone(contact_pixels(a, (mask, (3, 48, 48, 88))))
        self.assertIsNone(contact_pixels(a, a))
        xy = contact_pixels(a, (mask, (3, 43, 48, 83)))
        self.assertIsNotNone(xy)

    def test_physical_grid_boundary_preserved(self):
        inv = np.eye(3)
        u = np.linspace(0, 3., 80)
        straight = np.stack((u, u * 0 + .1, np.ones_like(u)), axis=1)
        self.assertEqual(diagonal_span(straight, inv, axis=2), 0.)
        straight[:,1] += .4 * u
        self.assertGreater(diagonal_span(straight, inv, axis=2), 3.)


if __name__ == '__main__':
    unittest.main()
