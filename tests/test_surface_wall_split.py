import unittest

import numpy as np
from PIL import Image

from engine.camera_lines import _runs
from engine.surface_wall_split import refine


class WallSplitTest(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((95, 124))
        self.lower = y >= 37
        self.mask = ((x > 7) & (x < 117) & (y > 6) & (y < 87)
                     & (~self.lower | (x < 34) | (x > 89)))
        # An independent through-hole in the upper beam.
        self.mask[17:24, 59:67] = False
        plane = (.1 + .0012 * x + .0007 * y).astype(np.float32)
        self.plane = plane
        self.depth = plane + self.lower * .14
        gray = np.where(self.lower, 167, 106).astype(np.uint8)
        self.image = Image.fromarray(gray).convert('RGB')
        runs, = _runs(self.mask)
        self.part = dict(id='wall', support=dict(size=list(self.image.size), runs=runs),
                         normal_camera=[0., 0., -1.], quad=[[1, 1], [120, 1], [120, 90], [1, 90]])
        self.rectangle = dict(id='upper', normal_camera=[0., 0., -1.],
                              quad=[[7, 7], [117, 7], [117, 37], [7, 37]])

    def test_two_separate_pillars_and_original_hole(self):
        found = refine(self.image, [self.part], self.depth, [self.rectangle])
        self.assertEqual(len(found), 3)
        self.assertEqual(len({p['id'] for p in found}), 3)
        cover = np.zeros(self.mask.shape, int)
        for p in found:
            self.assertEqual(p['parent_id'], 'wall')
            self.assertNotEqual(p.get('quad'), self.part['quad'])
            for y, start, end in p['support']['runs']:
                cover[y, start:end] += 1
        np.testing.assert_array_equal(cover, self.mask)

    def test_rejects_rectangle_across_continuous_depth_and_shadowless_depth_step(self):
        self.assertEqual(refine(self.image, [self.part], self.plane, [self.rectangle]), [self.part])
        blank = Image.new('RGB', self.image.size, 'gray')
        self.assertEqual(refine(blank, [self.part], self.depth, [self.rectangle]), [self.part])
