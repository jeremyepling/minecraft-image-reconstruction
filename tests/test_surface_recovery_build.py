import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import surface_model


class WallRecoveryEntryTest(unittest.TestCase):
    def test_single_outline_uses_selected_last_depth_field(self):
        from engine.single_wall_gate import apply
        image_size=(48,32)
        first_field=np.ones((8,8),np.float32)
        normal=np.ones((8,8,3),np.float32)
        apple_field=np.full((8,12),3.,np.float32)
        recovery=dict(lines={'segments':[]},records=[],selected=[],
                      provenance=dict(device='cuda',scalelsd={'checkpoint_sha256':'lines'},
                                      mobilesam={'checkpoint_sha256':'shapes'}))
        with TemporaryDirectory() as temp:
            source=Path(temp)/'new.png'
            Image.new('RGB',image_size).save(source)
            for name,model in surface_model.SINGLE_WALL_DEPTH.items():
                dest=Path(temp)/name
                defaults=Path(temp)/'defaults.json'
                defaults.write_text(json.dumps({'outlines':'mask-first','boundary':name}))
                with self.subTest(name=name), patch.object(surface_model,'DEFAULTS',defaults), \
                        patch('engine.surface_fields.infer',side_effect=[normal,first_field]), \
                        patch('engine.mask_first.regions',return_value=[]), \
                        patch('engine.normal_regions.detect',return_value=[]), \
                        patch('engine.recovery_evidence.infer',return_value=(recovery,np.empty((0,8,12),bool))), \
                        patch('engine.mlsd_tiny.detect',return_value={'segments':[]}), \
                        patch('engine.recovery_evidence.depth_pro',return_value=(apple_field,{'checkpoint_sha256':'apple'})) as apple, \
                        patch('engine.single_wall_gate.apply',wraps=apply) as gate:
                    surface_model.build(source,'26.3',dest)
                    manifest=json.loads((dest/'build-manifest.json').read_text())
                    self.assertEqual(manifest['stages']['single_wall_depth'],model)
                    self.assertEqual(manifest['stages']['boundary_sequence'][-1],'single-wall-split')
                    if model=='depth-pro':
                        apple.assert_called_once_with(source,'cuda')
                        self.assertIs(gate.call_args.args[-1],apple_field)
                        self.assertEqual(manifest['model_sha256']['depth_pro'],'apple')
                    else:
                        apple.assert_not_called()
                        np.testing.assert_array_equal(gate.call_args.args[-1],np.ones((8,12)))

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
