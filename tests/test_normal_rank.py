import unittest

import numpy as np
from PIL import Image, ImageDraw

from engine.normal_rank import _evidence, detect, SETTINGS


class NormalRankTests(unittest.TestCase):
    def setUp(self):
        self.quad = np.array([(25, 20), (95, 20), (95, 80), (25, 80)], dtype=float)
        self.field = np.zeros((100, 130, 3), np.float32)
        self.field[:, :, 0] = 1.
        self.field[20:80, 25:95] = (0, 0, -1)

    def test_orientation_turn_promotes_the_same_color_edge(self):
        supported, evidence = _evidence(self.quad, .92, self.field, np.ones(2))
        self.assertGreater(supported, SETTINGS['min_confidence'])
        self.assertGreater(min(evidence['side_orientation_support']), .85)
        self.field[:] = (0, 0, -1)
        flat, evidence = _evidence(self.quad, .92, self.field, np.ones(2))
        self.assertLess(flat, SETTINGS['min_confidence'])
        self.assertEqual(evidence['side_orientation_support'], [0.] * 4)

    def test_missing_outside_measurements_cannot_support_a_bend(self):
        self.field[self.field[:, :, 0] == 1] = 0
        confidence, evidence = _evidence(self.quad, .92, self.field, np.ones(2))
        self.assertLess(confidence, SETTINGS['min_confidence'])
        self.assertEqual(evidence['side_orientation_support'], [0.] * 4)

    def test_normal_field_pixel_coordinates_scale_with_image(self):
        a = _evidence(self.quad, .92, self.field, np.ones(2))
        b = _evidence(self.quad, .92, np.repeat(self.field, 2, axis=1), [2, 1])
        self.assertAlmostEqual(a[0], b[0], places=4)

    def test_nonplanar_interior_and_malformed_input(self):
        self.field[20:80, 60:95] = (1, 0, 0)
        self.assertIsNone(_evidence(self.quad, .92, self.field, np.ones(2)))
        with self.assertRaises(ValueError):
            detect(Image.new('RGB', (130, 100)), self.field[:, :, 0])

    def test_detector_returns_native_pixels_and_normal_evidence(self):
        image = Image.new('RGB', (260, 200), (154, 192, 233))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 123, 260, 200), fill=(144, 141, 135))
        draw.rectangle((50, 40, 189, 159), fill=(140, 81, 53))
        result = detect(image, self.field)
        self.assertTrue(result)
        plane = max(result, key=lambda p:p['confidence'])
        xy = np.asarray(plane['quad'])
        np.testing.assert_allclose(xy.min(0), (50, 40), atol=5)
        np.testing.assert_allclose(xy.max(0), (190, 160), atol=5)
        self.assertIn('interior_normal_agreement', plane['evidence'])
