import unittest

import numpy as np
from PIL import Image

from engine import pavement_opening
from engine.pavement_obstructions import window_obstructions


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

    def test_only_visible_elevated_cells_can_be_removed(self):
        size = (101, 99)
        camera = dict(P=[[100, 0, 50, 400], [0, -100, 49, 692], [0, 0, 1, 8]])
        region = dict(support=dict(size=size, runs=[[y, 0, size[0]] for y in range(size[1])]))
        cells = {(0, -1, 0), (0, 0, 0), (0, 1, 0), (8, 1, 0)}
        self.assertEqual({(0, 1, 0)}, window_obstructions(size, camera, [region], cells))
