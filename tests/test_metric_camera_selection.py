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
             patch.object(camera_metric.camera_atlas_floor, 'predict') as atlas, \
             patch.object(camera_metric.camera_repeat_floor, 'predict') as repeat:
            result = camera_metric.predict(self.image)
        self.assertEqual(result['P'], self.camera['P'])
        self.assertEqual(result['camera_source'], 'seam-fit')
        atlas.assert_not_called()
        repeat.assert_not_called()

    def test_explicit_atlas_reaches_fallback(self):
        path = Path('specified-atlas')
        with patch.object(camera_metric.camera_seam, 'predict', return_value=self.empty), \
             patch.object(camera_metric.camera_atlas_floor, 'predict', return_value=self.camera) as atlas, \
             patch.object(camera_metric.camera_repeat_floor, 'predict') as repeat:
            result = camera_metric.predict(self.image, atlas_path=path)
        atlas.assert_called_once_with(self.image, '26.3', atlas_path=path)
        self.assertEqual(result['camera_source'], 'atlas-floor')
        self.assertEqual(result['P'], self.camera['P'])
        repeat.assert_not_called()

    def test_abstention_records_both_reasons(self):
        missing = dict(self.empty, reason='calibrated atlas unavailable')
        with patch.object(camera_metric.camera_seam, 'predict', return_value=self.empty), \
             patch.object(camera_metric.camera_atlas_floor, 'predict', return_value=missing), \
             patch.object(camera_metric.camera_repeat_floor, 'predict', return_value=missing):
            result = camera_metric.predict(self.image)
        self.assertIsNone(result['P'])
        self.assertIsNone(result['camera_source'])
        self.assertEqual(result['reason'], 'no seams')
        self.assertEqual(result['atlas_reason'], missing['reason'])
        self.assertEqual(result['repeat_reason'], missing['reason'])

    def test_repeat_fallback_preserves_earlier_failure_reasons_and_explicit_atlas(self):
        path = Path('specified-atlas')
        recovered = dict(self.camera, camera_source='horizon-repeat-floor')
        atlas_failure = dict(self.empty, reason='missing texel-frequency tracks')
        with patch.object(camera_metric.camera_seam, 'predict', return_value=self.empty), \
             patch.object(camera_metric.camera_atlas_floor, 'predict', return_value=atlas_failure), \
             patch.object(camera_metric.camera_repeat_floor, 'predict', return_value=recovered) as repeat:
            result = camera_metric.predict(self.image, atlas_path=path)
        repeat.assert_called_once_with(self.image, '26.3', atlas_path=path)
        self.assertEqual(result['P'], recovered['P'])
        self.assertEqual(result['camera_source'], 'horizon-repeat-floor')
        self.assertEqual(result['seam_reason'], self.empty['reason'])
        self.assertEqual(result['atlas_reason'], atlas_failure['reason'])
