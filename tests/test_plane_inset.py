import unittest

import numpy as np
from PIL import Image

from benchmark.mask_metrics import footprint
from engine.camera_lines import _runs
from engine.plane_inset import predict


class PlaneInsetTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (160, 120), 'gray')
        self.wall = np.zeros((30, 40), bool)
        self.wall[2:28, 2:38] = True
        self.wall[19:22, 22:26] = False
        runs, = _runs(self.wall)
        self.parent = dict(id='wall', support=dict(size=[40,30], runs=runs),
                           normal_camera=[.6, 0., -.8])
        labels = self.wall.astype(np.int32)
        labels[8:18, 12:22] = 2
        self.fields = dict(relaxed=labels,
                           relaxed_normals=np.array([[0.,0.,0.], [.6,0.,-.8], [.6,0.,-.8]]),
                           relaxed_centers=np.array([[0.,0.,0.], [0.,0.,5.], [0.,0.,4.]]))

    def test_parallel_depth_step_preserves_every_parent_pixel_and_hole(self):
        parts, events = predict(self.image, [self.parent], self.fields, 'clear-step')
        self.assertEqual(len(events), 1)
        self.assertEqual(len(parts), 2)
        masks = [footprint(p, self.wall.shape, self.image.size, True) for p in parts]
        np.testing.assert_array_equal(masks[0] | masks[1], self.wall)
        self.assertFalse((masks[0] & masks[1]).any())
        np.testing.assert_array_equal(masks[0], self.fields['relaxed'] == 2)

    def test_depth_difference_along_the_same_sloping_plane_keeps_it_whole(self):
        self.fields['relaxed_centers'][2] = [.8, 0., 5.6]
        parts, events = predict(self.image, [self.parent], self.fields, 'clear-step')
        self.assertFalse(events)
        self.assertEqual(parts, [self.parent])
