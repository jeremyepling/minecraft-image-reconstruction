import unittest

import numpy as np
from PIL import Image

from engine.surface_perimeter import draw_outline, masked_crop, outline, with_outline


def support(mask):
    runs = []
    for y, row in enumerate(mask):
        cuts = np.flatnonzero(np.diff(row, prepend=False, append=False))
        runs.extend([y, int(left), int(right)] for left, right in cuts.reshape(-1, 2))
    return dict(size=list(mask.shape[::-1]), runs=runs)


def fill(perimeter):
    # Cross each vertical edge with a horizontal ray from the pixel center.
    w, h = perimeter['size']
    inside = np.zeros((h, w), bool)
    for ring in perimeter['rings']:
        points = ring['points']
        for (x, y), (u, v) in zip(points, points[1:] + points[:1]):
            if x == u:
                inside[min(y, v):max(y, v), :x] ^= True
            elif y != v:
                raise ValueError('edge crosses the mask diagonally')
    return inside


class SurfacePerimeterTests(unittest.TestCase):
    def test_all_three_by_three_masks_round_trip_with_simple_rings(self):
        for bits in range(512):
            mask = ((bits >> np.arange(9)) & 1).astype(bool).reshape(3, 3)
            perimeter = outline(support(mask))
            np.testing.assert_array_equal(fill(perimeter), mask)
            for ring in perimeter['rings']:
                self.assertEqual(len(ring['points']), len(set(map(tuple, ring['points']))))

    def test_holes_disconnected_parts_and_an_island_inside_a_hole(self):
        mask = np.zeros((17, 24), bool)
        mask[:15, :15] = mask[:15, 17:23] = True
        mask[2:13, 2:13] = False
        mask[4:6, 4:6] = True
        mask[4:6, 18:20] = False
        part = with_outline(dict(support=support(mask), quad=[[1, 1], [8, 1], [8, 8], [1, 8]]))
        perimeter = part['outline']
        self.assertEqual(sorted(r['kind'] for r in perimeter['rings']), ['hole'] * 2 + ['outer'] * 3)
        np.testing.assert_array_equal(fill(perimeter), mask)
        self.assertEqual(part['sampling_chart'], part['quad'])

    def test_crop_and_render_cover_steps_outside_sampling_chart(self):
        mask = np.zeros((6, 8), bool)
        mask[1:4, 1:5] = mask[3:5, 4:7] = True
        mask[2, 2] = False
        part = dict(support=support(mask), quad=[[8, 5], [17, 5], [17, 11], [8, 11]])
        w, h = 43, 31
        rgb = np.arange(w * h * 3, dtype=np.uint8).reshape(h, w, 3)
        image = Image.fromarray(rgb)
        crop, (l, t, r, b) = masked_crop(image, part)
        enlarged = mask[np.ix_(((2*np.arange(h)+1)*6)//(2*h), ((2*np.arange(w)+1)*8)//(2*w))]
        restored = np.zeros_like(enlarged)
        restored[t:b, l:r] = np.asarray(crop.getchannel('A')) > 0
        np.testing.assert_array_equal(restored, enlarged)
        np.testing.assert_array_equal(np.asarray(crop)[:, :, :3], rgb[t:b, l:r])
        self.assertEqual(crop.getchannel('A').getbbox(), (0, 0, r-l, b-t))
        self.assertGreater(r, max(x for x, y in part['quad']))
        draw_outline(image, part, 'cyan', width=1)
        self.assertEqual(image.getpixel((int(7*w/8), int(4*h/6))), (0, 255, 255))
        # A point on the sampling chart in the solid part of the mask stays untouched.
        self.assertEqual(image.getpixel((8, 8)), tuple(rgb[8, 8]))

    def test_empty_support(self):
        part = dict(support=dict(size=[4, 5], runs=[]))
        self.assertEqual(outline(part['support'])['rings'], [])
        self.assertIsNone(masked_crop(Image.new('RGB', (13, 7)), part))
