import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from benchmark import compare_surface_methods as runner, normal_fields, surface_metrics
from benchmark.compare_surface_methods import validate_comparison
from engine import surface_model


class SurfaceJobTests(unittest.TestCase):
    def test_checks_same_inputs_and_one_stage_across_every_challenger(self):
        base = dict(schema=2, comparison={'screenshots':'aaa'}, methods={'outlines':'edges', 'pitch':'repeat'},
                    settings={'outlines':{}, 'pitch':{}}, lift={}, implementations={'outlines':'a', 'pitch':'b'})
        alternative = copy.deepcopy(base)
        alternative['methods']['outlines'] = 'normal-regions'
        validate_comparison([base, alternative])
        third = copy.deepcopy(base)
        third['settings']['pitch']['phase_bins'] = 80
        with self.assertRaisesRegex(ValueError, 'one configured stage'):
            validate_comparison([base, alternative, third])
        third['comparison']['screenshots'] = 'bbb'
        with self.assertRaisesRegex(ValueError, 'identical splits'):
            validate_comparison([base, third])
        unchanged_names = copy.deepcopy(base)
        unchanged_names['implementations'].update(outlines='new outline code', pitch='new pitch code')
        with self.assertRaisesRegex(ValueError, 'one configured stage'):
            validate_comparison([base, unchanged_names])

    def test_fingerprints_separate_algorithm_edits_from_evaluation(self):
        methods = dict(outlines='edges+shade', pitch='repeat', order='geometry')
        evaluator = runner.evaluator_signature()
        original = runner.stage_implementations(methods)
        with patch.object(surface_model,'lens',surface_model.intrinsic):
            altered = runner.stage_implementations(methods)
            self.assertEqual([k for k in original if original[k] != altered[k]], ['lift'])
            self.assertEqual(evaluator, runner.evaluator_signature())
        with patch.object(surface_metrics,'IOU_THRESHOLD', 0.1):
            self.assertNotEqual(evaluator, runner.evaluator_signature())

    def test_saved_input_recipe_survives_windows_checkout_line_endings(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for path, code in [('engine/surface_fields.py', b'a\nb\n'), ('benchmark/depth_probe.py', b'c\r\nd\r\n')]:
                file = root / path
                file.parent.mkdir()
                file.write_bytes(code)
            with patch.object(normal_fields, 'ROOT', root):
                normal_fields.inference_signatures.cache_clear()
                try:
                    signatures = normal_fields.inference_signatures()
                    normal = np.ones((2,2,3))
                    np.savez(root/'frame.npz', normal=normal, signature='frame:' + signatures[1])
                    (root/'engine/surface_fields.py').write_bytes(b'a\r\nb\r\n')
                    normal_fields.inference_signatures.cache_clear()
                    np.testing.assert_equal(normal_fields.cached_fields('frame', root, ['normal'])['normal'], normal)
                    (root/'engine/surface_fields.py').write_bytes(b'changed\n')
                    normal_fields.inference_signatures.cache_clear()
                    self.assertIsNone(normal_fields.cached_fields('frame', root, ['normal']))
                finally:
                    normal_fields.inference_signatures.cache_clear()
