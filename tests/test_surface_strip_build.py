import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import mlsd_tiny, surface_model


class StripBuildTest(unittest.TestCase):
    def test_strip_receives_one_image_and_the_initial_contrast_masks(self):
        normal = np.ones((5, 5, 3), np.float32)
        depth = np.ones((5, 5), np.float32)
        aligned = [{'id':'aligned'}]
        lines = dict(image_size=[32, 32], junctions=[], segments=[[2., 10., 30., 10.]])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'independent.png'
            Image.new('RGB', (32,32), 'orange').save(source)
            with patch('engine.surface_fields.infer', side_effect=[normal, depth]) as infer, \
                    patch('engine.mask_first.regions', return_value=[]), \
                    patch('engine.surface_boundary_refine.refine', return_value=aligned), \
                    patch('engine.mlsd_tiny.detect', return_value=lines) as detect, \
                    patch('engine.surface_strip_split.refine', return_value=[]) as split:
                surface_model.build(source, '26.3', root/'out', boundary='strip-mlsd')
            manifest = json.loads((root/'out/build-manifest.json').read_text())
        self.assertEqual([c.args[1] for c in infer.call_args_list], ['normal', 'depth'])
        self.assertIs(detect.call_args.args[0], split.call_args.args[0])
        self.assertIs(split.call_args.args[1], aligned)
        self.assertIs(split.call_args.args[2]['mlsd'], lines)
        self.assertIs(split.call_args.args[3], normal)
        self.assertIs(split.call_args.args[4], depth)
        self.assertEqual(manifest['model_sha256']['mlsd'], mlsd_tiny.MODEL[2])
        self.assertEqual(manifest['stages']['initial_boundary'], 'color-edges')
        self.assertEqual(manifest['surface_finder']['boundary_settings']['detected_segments'], 1)
