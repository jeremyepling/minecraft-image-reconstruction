import unittest

import numpy as np
from PIL import Image

from engine import pavement_opening


class PavementOpeningTests(unittest.TestCase):
    size = (100, 100)

    def patch(self, name, x, y, right, bottom):
        return dict(id=name, normal_camera=[0., 0., 1.], support=dict(
            size=list(self.size), runs=[[row, x, right] for row in range(y, bottom)]))

    def test_missing_or_degenerate_camera_preserves_patches(self):
        image = Image.new('RGB', self.size)
        patch = self.patch('interior', 35, 35, 65, 65)
        for p in (None, [], np.full((3, 4), np.nan), np.ones((3, 4))):
            with self.subTest(projection=p):
                self.assertEqual([], pavement_opening.detect(image, [patch], dict(P=p)))

    def test_check_for_enclosure_before_estimating_camera(self):
        center = self.patch('interior', 35, 35, 65, 65)
        edges = [self.patch('left', 28, 35, 34, 65), self.patch('right', 66, 35, 72, 65),
                 self.patch('head', 35, 27, 65, 34), self.patch('sill', 35, 66, 65, 74)]
        self.assertTrue(pavement_opening.possible(self.size, [center] + edges))
        self.assertFalse(pavement_opening.possible(self.size, [center] + edges[:-1]))
        self.assertFalse(pavement_opening.possible(self.size, [center]))

