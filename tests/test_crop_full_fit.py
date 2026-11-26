import unittest

import numpy as np
from PIL import Image

from benchmark.crop_full_fit import _project, expand_sample, sample
from engine.surface_sample import sample_surface


class FullFitTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (40, 40), 'white')
        runs = [(row, start, stop) for row in range(4, 34)
                for start, stop in ([(3, 14), (22, 35)] if 10 <= row < 26 else [(3, 35)])]
        self.part = dict(quad=[[7, 6], [32, 6], [32, 30], [7, 30]],
                         support=dict(size=(40, 40), runs=runs))

    def test_full_fit_covers_edges_but_preserves_hole(self):
        result = sample(self.image, self.part)
        np.testing.assert_allclose(result['quad'], [[3, 4], [35, 4], [35, 34], [3, 34]])
        self.assertFalse(result['mask'][15, 15])
        self.assertTrue(result['mask'][15, 5])
        self.assertFalse(result['mask'][0].any())
        self.assertFalse(result['rgb'][~result['mask']].any())

    def test_existing_window_and_unusable_chart(self):
        wide = dict(self.part, quad=[[1, 1], [38, 1], [38, 37], [1, 37]])
        np.testing.assert_equal(sample(self.image, wide)['quad'], wide['quad'])
        self.assertIsNone(sample(self.image, dict(self.part, quad=[[1, 1]] * 4)))

    def test_selector_receives_only_owned_centers(self):
        def selector(uv):
            self.assertEqual(len(uv), 32 * 30 - 8 * 16)
            return None
        result = expand_sample(self.image, self.part, selector)
        np.testing.assert_equal(result['mask'], sample_surface(self.image, self.part)['mask'])

    def test_pole_in_extended_window(self):
        mapping = np.array([[1., 0., 0.], [0., 1., 0.], [-.5, 0., 1.]])
        self.assertIsNone(_project(mapping, np.array([[0., 0.], [3., 1.]]), np.array([.5, .5])))
