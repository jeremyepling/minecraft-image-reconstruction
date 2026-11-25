import unittest

import numpy as np
from PIL import Image, ImageDraw

from engine.structural_lines import detect


def covers(lines, start, end, tolerance=3):
    a, b = np.asarray(start), np.asarray(end)
    direction = b - a
    length = np.linalg.norm(direction)
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    for line in lines:
        endpoints = np.array([line['p'], line['q']]) - a
        if np.max(np.abs(endpoints @ normal)) > tolerance:
            continue
        low, high = sorted(endpoints @ direction)
        if min(high, length) - max(low, 0) > .72 * length:
            return True
    return False


class WholeFrameLines(unittest.TestCase):
    def test_sloping_outline_includes_internal_diagonal(self):
        im = Image.new('RGB', (360, 240), (80, 108, 129))
        draw = ImageDraw.Draw(im)
        draw.polygon([(32, 38), (306, 68), (290, 202), (32, 184)], fill=(173, 176, 170))
        draw.polygon([(82, 92), (153, 100), (146, 159), (82, 152)], fill=(90, 85, 79))
        draw.polygon([(82, 92), (109, 109), (109, 150), (82, 152)], fill=(129, 135, 131))
        lines = detect(im)
        broad = [row for row in lines if row['scale'] == 'coarse']
        self.assertTrue(covers(broad, (32, 38), (306, 68)))
        self.assertTrue(covers(broad, (290, 202), (306, 68)))
        self.assertTrue(covers(lines, (82, 92), (109, 109)))
        self.assertEqual(lines, detect(im))

    def test_isoluminant_step_and_separate_collinear_spans(self):
        im = Image.new('RGB', (280, 180), (205, 43, 24))
        draw = ImageDraw.Draw(im)
        for x in (28, 160):
            draw.rectangle((x, 65, x + 74, 115), fill=(38, 123, 43))
        lines = detect(im)
        self.assertTrue(covers(lines, (28, 65), (102, 65)))
        self.assertTrue(covers(lines, (160, 65), (234, 65)))
        self.assertFalse(covers(lines, (28, 65), (234, 65)))

    def test_constant_or_tiny_image(self):
        for size in ((242, 148), (2, 10)):
            self.assertEqual(detect(Image.new('RGB', size, 'gray')), [])
