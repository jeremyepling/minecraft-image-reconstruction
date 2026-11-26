import unittest

import numpy as np
from PIL import Image

from benchmark.sky_gap_method import gap_mask


class SkyOpeningTest(unittest.TestCase):
    def test_measure_within_rows_and_require_gradient_or_horizon(self):
        image = np.full((100,160,3), [112,112,115], np.uint8)
        for y in range(60):
            image[y] = [130+min(y,35),170+min(y,35),255]
        original = image.copy()
        image[:60,40:120] = [71,69,65]
        image[12:26,48:60] = original[12:26,48:60]
        image[36:48,65:78] = original[36:48,65:78]   # identical flat blue away from horizon
        image[45:60,82:93] = original[45:60,82:93]   # clear gap reaching the ground transition
        image[61:75,96:112] = original[55,10]        # blue below the sky
        blue = np.zeros(image.shape[:2],bool)
        blue[:60,:40] = blue[:60,120:] = True
        added = gap_mask(Image.fromarray(image),blue,np.zeros_like(blue))
        expected = np.zeros_like(blue)
        expected[12:26,48:60] = expected[45:60,82:93] = True
        np.testing.assert_array_equal(added,expected)
        blue[:,120:] = False
        self.assertFalse(gap_mask(Image.fromarray(image),blue,np.zeros_like(blue)).any())
