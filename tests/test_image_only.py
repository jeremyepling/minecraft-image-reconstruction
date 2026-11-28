"""A saved-image run must never fall back to a Minecraft client or registry launch."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import reconstruct
from benchmark import camera_fixture, score_saved_planes
from calibration.verify_build import verify
from runtime import capture
from runtime.mode import IMAGE_ONLY, image_only, mark_image_only


ROOT = Path(__file__).resolve().parents[1]


class ImageOnlyTests(unittest.TestCase):
    def test_saved_comparison_rejects_unexpected_fixture_capture_before_staging(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ), patch.object(sys, 'argv', ['score']):
            os.environ.pop(IMAGE_ONLY, None)
            out = Path(tmp) / 'unexpected-world'
            # Simulate a future data-loading helper trying to replace a missing screenshot.
            with patch.object(score_saved_planes, 'read', side_effect=lambda _: camera_fixture.capture(out, 'smooth_stone')), \
                 patch.object(camera_fixture, 'discover', side_effect=AssertionError('looked for installed Minecraft')):
                with self.assertRaisesRegex(RuntimeError, 'Image-only run'):
                    score_saved_planes.main()
            self.assertFalse(out.exists())
            self.assertNotIn(IMAGE_ONLY, os.environ)

    def test_ordinary_build_rejects_unexpected_verification(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ):
            os.environ.pop(IMAGE_ONLY, None)
            folder = Path(tmp)
            with patch.object(sys, 'argv', ['reconstruct', 'build', 'picture.png', '--atlas', str(folder),
                                            '--out', str(folder / 'out')]), \
                 patch('engine.floating.build', side_effect=lambda *a, **k: verify(folder, None, folder, [])):
                with self.assertRaisesRegex(RuntimeError, 'Image-only run'):
                    reconstruct.main()
            self.assertFalse((folder / 'out').exists())

    def test_worker_and_its_subprocess_cannot_reach_runtime_setup(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ):
            os.environ.pop(IMAGE_ONLY, None)
            # Exercise spawn even on systems where fork would inherit the parent's Python state.
            with ProcessPoolExecutor(1, mp_context=get_context('spawn'), initializer=mark_image_only) as pool:
                for action, flag in [('capture', '--run'), ('registry', '--out')]:
                    args = [sys.executable, '-m', 'runtime.capture', action, '--version', 'missing-fixture-version',
                            '--manifest', str(Path(tmp) / 'missing.json'), flag, tmp]
                    result = pool.submit(subprocess.run, args, cwd=ROOT, capture_output=True, text=True).result()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Image-only run cannot start Minecraft', result.stderr)
                    self.assertNotIn('Missing version manifest', result.stderr)
            self.assertNotIn(IMAGE_ONLY, os.environ)

    def test_explicit_capture_reaches_runtime_setup_outside_image_only_command(self):
        with patch.dict(os.environ), patch.object(sys, 'argv', ['capture', 'capture', '--version', '26.3', '--run', '.']):
            os.environ.pop(IMAGE_ONLY, None)
            with patch.object(capture, 'classpath', side_effect=FileNotFoundError('explicit runtime path')) as setup:
                with image_only(), self.assertRaisesRegex(RuntimeError, 'Image-only run'):
                    capture.main()
                setup.assert_not_called()
                with self.assertRaisesRegex(FileNotFoundError, 'explicit runtime path'):
                    capture.main()
                setup.assert_called_once()
