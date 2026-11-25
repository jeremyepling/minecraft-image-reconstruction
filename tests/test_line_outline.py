import unittest

import numpy as np
from PIL import Image, ImageDraw

from engine.line_outline import group_lines, silhouette


class LineOutlineTests(unittest.TestCase):
    def test_bright_wall_with_clear_opening(self):
        image = Image.new('RGB', (220, 160), '#404860')
        paint = ImageDraw.Draw(image)
        paint.rectangle((45, 30, 175, 135), fill='#ced0cf')
        paint.rectangle((85, 75, 130, 135), fill='#404860')
        lines = [dict(p=[45, 30], q=[45, 135], scale='coarse'),
                 dict(p=[45, 30], q=[175, 30], scale='coarse'),
                 dict(p=[85, 75], q=[85, 135], scale='coarse')]
        mask = silhouette(image, lines)
        self.assertEqual(mask.getpixel((100, 40)), 255)
        self.assertEqual(mask.getpixel((100, 115)), 0)
        self.assertEqual(mask.getpixel((13, 65)), 0)
        self.assertTrue(all(l['role']=='outline' for l in group_lines(lines, mask)))

    def test_alpha_records_visible_footprint(self):
        image = Image.new('RGBA', (18, 21))
        ImageDraw.Draw(image).rectangle((4, 7, 12, 14), fill='white')
        np.testing.assert_equal(np.asarray(silhouette(image, [])), np.asarray(image.getchannel('A')))
