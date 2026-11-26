import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import camera_metric


class MetricCameraSelectionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (320, 256), 'gray')
        self.camera = dict(P=np.eye(3, 4).tolist(), scale_status='supported', reason='measured')
        self.empty = dict(P=None, scale_status='uncertain', reason='no seams')

    def test_supported_seams_bypass_atlas(self):
        with patch.object(camera_metric.camera_seam, 'predict', return_value=self.camera), \
             patch.object(camera_metric.camera_atlas_floor, 'predict') as atlas:
            result = camera_metric.predict(self.image)
        self.assertEqual(result['P'], self.camera['P'])
        self.assertEqual(result['camera_source'], 'seam-fit')
        atlas.assert_not_called()

    def test_explicit_atlas_reaches_fallback(self):
        path = Path('specified-atlas')
        with patch.object(camera_metric.camera_seam, 'predict', return_value=self.empty), \
             patch.object(camera_metric.camera_atlas_floor, 'predict', return_value=self.camera) as atlas:
            result = camera_metric.predict(self.image, atlas_path=path)
        atlas.assert_called_once_with(self.image, '26.3', atlas_path=path)
        self.assertEqual(result['camera_source'], 'atlas-floor')
        self.assertEqual(result['P'], self.camera['P'])

    def test_abstention_records_both_reasons(self):
        missing = dict(self.empty, reason='calibrated atlas unavailable')
        with patch.object(camera_metric.camera_seam, 'predict', return_value=self.empty), \
             patch.object(camera_metric.camera_atlas_floor, 'predict', return_value=missing):
            result = camera_metric.predict(self.image)
        self.assertIsNone(result['P'])
        self.assertIsNone(result['camera_source'])
        self.assertEqual(result['reason'], 'no seams')
        self.assertEqual(result['atlas_reason'], missing['reason'])
