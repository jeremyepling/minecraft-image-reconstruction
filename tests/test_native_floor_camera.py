from contextlib import ExitStack
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from engine import camera_metric, camera_native_floor


class NativeFloorCameraTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (320, 256), (125, 125, 115))
        self.empty = dict(P=None, reason='abstained', scale_status='uncertain')
        self.accepted = dict(P=[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
                             reason='tested', camera_source='native-floor-atlas', scale_status='supported')

    def test_source_order_and_explicit_atlas(self):
        path = Path('selected-atlas')
        with patch.object(camera_metric.camera_seam, 'predict', return_value=self.empty), \
             patch.object(camera_metric.camera_atlas_floor, 'predict', return_value=self.empty), \
             patch.object(camera_metric.camera_repeat_floor, 'predict', return_value=self.empty), \
             patch.object(camera_native_floor, 'predict', return_value=self.accepted) as native:
            result = camera_metric.predict(self.image, '26.3', atlas_path=path)
        native.assert_called_once_with(self.image, '26.3', atlas_path=path)
        self.assertEqual(result['P'], self.accepted['P'])
        self.assertEqual(result['repeat_reason'], self.empty['reason'])

    def test_existing_successes_bypass_native_fit(self):
        stages = (camera_metric.camera_seam, camera_metric.camera_atlas_floor, camera_metric.camera_repeat_floor)
        for index in range(len(stages)):
            with self.subTest(index=index), ExitStack() as patches:
                for slot, stage in enumerate(stages):
                    patches.enter_context(patch.object(stage, 'predict', return_value=self.accepted if slot == index else self.empty))
                native = patches.enter_context(patch.object(camera_native_floor, 'predict'))
                self.assertEqual(camera_metric.predict(self.image)['P'], self.accepted['P'])
                native.assert_not_called()

    def test_missing_filters_and_unsupported_image_abstain_before_loading_atlas(self):
        with patch.object(camera_native_floor, '_atlas', side_effect=AssertionError('unexpected atlas load')):
            for size in ((255, 320), (4097, 320)):
                result = camera_native_floor.predict(Image.new('RGB', size))
                self.assertIsNone(result['P'])
            self.assertIsNone(camera_native_floor.predict(self.image, '26.1')['P'])
            with patch.object(camera_native_floor, 'cv2', None), patch.dict(os.environ, MCIR_RECOVERY_PYTHON=''):
                self.assertEqual(camera_native_floor.predict(self.image)['reason'], 'native image filters unavailable')

    def test_broken_explicit_atlas_abstains(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'atlas.json').write_text(json.dumps({'atlas': 'absent.png'}))
            self.assertIsNone(camera_native_floor.predict(self.image, atlas_path=path)['P'])

    def test_uniform_floor_has_no_phase(self):
        with patch.object(camera_native_floor, '_atlas', return_value=object()):
            self.assertIsNone(camera_native_floor.predict(self.image)['P'])

    def test_configured_interpreter_receives_original_pixels_and_explicit_atlas(self):
        import sys
        from io import BytesIO
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / 'custom-python.exe'
            executable.touch()
            selected = Path(folder) / 'atlas'
            with patch.dict(os.environ, MCIR_RECOVERY_PYTHON=str(executable)), \
                 patch.object(camera_native_floor, 'cv2', None), \
                 patch.object(camera_native_floor.subprocess, 'run',
                              return_value=SimpleNamespace(stdout=json.dumps(self.accepted).encode())) as spawn:
                actual = camera_native_floor.predict(self.image, atlas_path=selected)
            self.assertEqual(actual, self.accepted)
            self.assertEqual(spawn.call_args.args[0], [str(executable.resolve()), '-m', 'engine.camera_native_floor',
                                                      '26.3', str(selected.resolve())])
            with Image.open(BytesIO(spawn.call_args.kwargs['input'])) as sent:
                self.assertEqual(sent.tobytes(), self.image.tobytes())
            with patch.dict(os.environ, MCIR_RECOVERY_PYTHON=sys.executable), patch.object(camera_native_floor, 'cv2', None):
                self.assertIsNone(camera_native_floor.predict(self.image)['P'])
