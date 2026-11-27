"""Reject unverified texture scales and preserve normal camera fallbacks."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import camera_repeat_floor as repeat


class RepeatFloorCameraTests(unittest.TestCase):
    def test_missing_or_explicitly_broken_atlas_abstains_before_search(self):
        image = Image.new('RGB', (512, 384), 'gray')
        path = Path('missing-atlas')
        with patch.object(repeat, '_atlas', return_value=None) as atlas, \
                patch.object(repeat, '_solve', side_effect=AssertionError('must not search')):
            result = repeat.predict(image, atlas_path=path)
        atlas.assert_called_once_with('26.3', path)
        self.assertIsNone(result['P'])
        self.assertEqual(result['reason'], 'calibrated atlas unavailable')

    def test_unsupported_inputs_do_not_load_assets(self):
        with patch.object(repeat, '_atlas', side_effect=AssertionError('must not load')):
            for size, version in [((512, 384), 'unknown'), ((64, 48), '26.3'), ((4097, 256), '26.3')]:
                with self.subTest(size=size, version=version):
                    self.assertIsNone(repeat.predict(Image.new('RGB', size), version)['P'])

    def test_unregistered_repeats_cannot_establish_a_block_grid(self):
        y, x = np.mgrid[:384, :512]
        rng = np.random.default_rng(41)
        patterns = [np.full(x.shape, 127), 126 + x % 8,
                    126 + ((x//6+y//6) % 2), rng.integers(124, 129, x.shape)]
        with patch.object(repeat, '_atlas', return_value=object()), \
                patch.object(repeat, 'register', return_value=(None, {'reason': 'no independent atlas match'})):
            for pattern in patterns:
                with self.subTest(std=float(pattern.std())):
                    image = Image.fromarray(pattern.astype(np.uint8)).convert('RGB')
                    self.assertIsNone(repeat.predict(image)['P'])

    def test_singular_fit_abstains(self):
        with patch.object(repeat, '_atlas', return_value=object()), \
                patch.object(repeat, '_solve', side_effect=np.linalg.LinAlgError):
            result = repeat.predict(Image.new('RGB', (512, 384)))
        self.assertIsNone(result['P'])
        self.assertEqual(result['reason'], 'degenerate tile translation fit')
