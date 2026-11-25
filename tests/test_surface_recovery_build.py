import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import surface_model


class WallRecoveryEntryTest(unittest.TestCase):
    def test_prompts_precede_junctions_and_mobile_partitions_completed_undersides(self):
        walls, junctions, undersides = ([{'id': name}] for name in ('wall', 'junction', 'underside'))
        normals = np.ones((8, 8, 3), np.float32)
        depth = np.ones((8, 8), np.float32)
        masks = np.ones((1, 8, 8), bool)
        extra = dict(lines={'segments': []}, records=[{'area': 32}], selected=[0],
                     provenance={name: {'checkpoint_sha256': name} for name in ('scalelsd', 'mobilesam')})
        with TemporaryDirectory() as temp:
            source = Path(temp)/'arbitrary.png'
            Image.new('RGB', (32, 32)).save(source)
            with patch('engine.surface_fields.infer', side_effect=[normals, depth]), \
                    patch('engine.mask_first.regions', return_value=[]), \
                    patch('engine.normal_regions.detect', return_value=[]), \
                    patch('engine.surface_wall_split.refine', return_value=walls), \
                    patch('engine.recovery_evidence.infer', return_value=(extra, masks)) as prompt, \
                    patch('engine.surface_junction_split.refine', return_value=junctions) as join, \
                    patch('engine.mlsd_tiny.detect', return_value={'segments': []}), \
                    patch('engine.surface_strip_split.refine', return_value=undersides) as underside, \
                    patch('benchmark.mobile_wall_split.refine', return_value=([], [])) as mobile:
                surface_model.build(source, '26.3', Path(temp)/'out', boundary='wall-recovery')
            manifest = json.loads((Path(temp)/'out/build-manifest.json').read_text())
        self.assertIs(prompt.call_args.args[1], walls)
        self.assertIs(join.call_args.args[1], walls)
        self.assertIs(underside.call_args.args[1], junctions)
        self.assertIs(mobile.call_args.args[1], undersides)
        self.assertIs(mobile.call_args.args[2], masks)
        self.assertEqual(mobile.call_args.args[3:], (extra['records'], extra['selected']))
        self.assertEqual(set(manifest['model_sha256']), {'normal', 'depth', 'mlsd', 'scalelsd', 'mobilesam'})
        self.assertEqual(manifest['stages']['boundary_sequence'],
                         ['color-edges', 'wall-depth-anything', 'junction-cuts', 'strip-mlsd', 'mobile-wall-split'])
