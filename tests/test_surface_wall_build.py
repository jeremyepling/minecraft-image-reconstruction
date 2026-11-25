import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import surface_model


class WallBuildTest(unittest.TestCase):
    def test_rectangles_share_normal_inference_and_use_lifted_orientation(self):
        normal = np.ones((6, 6, 3), np.float32)
        depth = np.ones((6, 6), np.float32)
        wall = dict(id='mask', support=dict(size=[6, 6], runs=[[1, 1, 4]]))
        rectangle = dict(id='rect', normal_camera=[1, 0, 0], quad=[[7, 6], [50, 6], [50, 32], [7, 32]])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'source.png'
            Image.new('RGB', (64, 64)).save(path)
            with patch('engine.surface_fields.infer', side_effect=[normal, depth]) as infer, \
                    patch('engine.mask_first.regions', return_value=[wall]) as regions, \
                    patch('engine.normal_regions.detect', return_value=[rectangle]) as hints, \
                    patch('engine.surface_wall_split.refine', return_value=[wall]) as split, \
                    patch.object(surface_model, 'pitch') as pitch:
                surface_model.build(path, '26.3', root, boundary='wall-depth-anything')
            manifest = json.loads((root/'build-manifest.json').read_text())
        self.assertEqual([c.args[1] for c in infer.call_args_list], ['normal', 'depth'])
        self.assertIs(regions.call_args.args[1], normal)
        self.assertIs(hints.call_args.args[1], normal)
        self.assertIs(split.call_args.args[2], depth)
        np.testing.assert_allclose(split.call_args.args[3][0]['normal_camera'], [0, 0, -1], atol=1.e-8)
        pitch.assert_not_called()
        self.assertEqual(manifest['stages']['split_hints'], 'rectangle-finder')
        self.assertEqual(manifest['stages']['initial_boundary'], 'color-edges')
        self.assertEqual(manifest['surface_finder']['relative_depth'], 'Depth Anything V2')
        self.assertIn('wall_split', manifest['surface_finder']['boundary_settings'])

    def test_old_and_new_rectangle_names_agree(self):
        normal = np.ones((6, 6, 3), np.float32)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'source.png'
            Image.new('RGB', (64, 64)).save(path)
            models = []
            for name in ('normal-regions', 'rectangle-finder'):
                out = root/name
                with patch('engine.surface_fields.infer', return_value=normal) as infer, \
                        patch('engine.normal_regions.detect', return_value=[]):
                    surface_model.build(path, '26.3', out, outlines=name)
                self.assertEqual([c.args[1] for c in infer.call_args_list], ['normal'])
                manifest = json.loads((out/'build-manifest.json').read_text())
                self.assertEqual(manifest['stages']['outlines'], name)
                self.assertEqual(manifest['surface_finder']['implementation'], 'normal-regions')
                models.append(json.loads((out/'surfaces.json').read_text()))
        self.assertEqual(*models)
