"""The budget-matched arm accepts an RGB and never opens the learned cache."""
import unittest
from unittest.mock import patch

from PIL import Image
from benchmark import camera_geocalib as method


class GeoCalibTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (1440, 900))

    def test_unlearned_search_uses_only_the_image_boundary(self):
        with patch.object(method, "boundary", return_value=400.) as boundary, patch.object(method, "peaks", return_value=[]) as peaks:
            result = method.solve(self.image)
        boundary.assert_called_once()
        self.assertEqual([c.args[1] for c in peaks.call_args_list], [400+offset*900 for offset in method.HORIZON_OFFSETS])
        self.assertIsNone(result["P"])

    def test_learned_search_uses_proposed_horizon(self):
        with patch.object(method, "boundary", side_effect=AssertionError), patch.object(method, "peaks", return_value=[]) as peaks:
            result = method.solve(self.image, {"horizon_center_px": 320., "focal_xy_px": [900.,900.]})
        self.assertEqual([c.args[1] for c in peaks.call_args_list], [320+offset*900 for offset in method.HORIZON_OFFSETS])
        self.assertIsNone(result["P"])

    def test_missing_boundary_gets_uniform_search(self):
        with patch.object(method, "boundary", return_value=None), patch.object(method, "peaks", return_value=[]) as peaks:
            method.solve(self.image)
        self.assertEqual(peaks.call_count, len(method.HORIZON_OFFSETS))


if __name__ == "__main__":
    unittest.main()
