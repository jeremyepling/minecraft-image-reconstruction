"""Exercise a camera-free planar sheet with exactly known native full-face support."""
import unittest

import numpy as np
from PIL import Image

try:
    from benchmark.floating_build_local_grid import atlas, predict
except ImportError:
    predict = None


@unittest.skipIf(predict is None, "Local-grid experiment requires OpenCV")
class LocalGridTest(unittest.TestCase):
    def test_wall_exports_with_no_third_axis_and_excludes_opening(self):
        try:
            catalog = atlas()
        except ValueError:
            self.skipTest("Java 26.3 vanilla atlas is unavailable")
        tile = Image.fromarray(catalog.tiles[catalog.labels.index('bricks')])
        tile = np.asarray(tile.resize((32, 32), Image.Resampling.NEAREST))
        wall = np.tile(tile, (4, 5, 1))
        rgb = np.pad(wall, ((20, 20), (20, 20), (0, 0)), constant_values=170)
        rgb[52:84, 84:116] = 170
        height, width = rgb.shape[:2]
        runs = []
        for y in range(20, 148):
            spans = ((20, 84), (116, 180)) if 52 <= y < 84 else ((20, 180),)
            runs.extend([y, left, right] for left, right in spans)
        support = dict(size=[width, height], runs=runs)
        part = dict(id='wall', normal_camera=[0, 0, -1], support=support,
                    support_area=32*32*19)
        counts = dict(surfaces=[dict(id='wall', counts=[5, 4],
                                     quad=[[20, 20], [180, 20], [180, 148], [20, 148]])])
        hits, _, evidence = predict(Image.fromarray(rgb), [part], counts)
        self.assertEqual(evidence['status'], 'local_only_nonempty')
        self.assertFalse(evidence['global_projection'])
        centers = {tuple(np.mean(row['quad'], axis=0).round().astype(int)) for row in hits}
        expected = {(36+32*x, 36+32*y) for x in range(5) for y in range(4)}-{(100,68)}
        self.assertSetEqual(centers, expected)
        self.assertEqual({row['texture'] for row in hits}, {'bricks'})


if __name__ == '__main__':
    unittest.main()
