"""The exterior build must publish its current result when a reused directory held a build."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.mask_blocks import decoded_pack
from engine.export import JAVA_26_3_FORMAT, write_pack
from engine.surface_model import build


class SupportedPlacementEntryTest(unittest.TestCase):
    def test_export_after_geometry_survives_missing_material_atlas(self):
        # A missing floor atlas can also make the camera uncertain. Freeze a
        # successful placer here so the missing *material* atlas is exercised.
        cells = {(1,0,2),(1,1,2)}
        status = dict(camera_status='metric', placed_blocks=len(cells),
                      P=[[80,0,40,0],[0,-80,25,0],[0,0,1,6]])
        with TemporaryDirectory() as temp:
            out = Path(temp)/'build'
            image = Path(temp)/'source.png'
            Image.new('RGB',(80,50),'gray').save(image)
            with patch('engine.surface_fields.infer',return_value=np.ones((5,8))), \
                    patch('engine.surface_model.proposals_for',return_value=[]), \
                    patch('engine.visible_placement.place',return_value=(cells,status)), \
                    patch('engine.camera_atlas_floor._atlas',return_value=None):
                build(image,'26.3',out,outlines='mask-first',boundary='regions-first',through='placement')
            self.assertEqual(decoded_pack(out/'reconstruction.zip'),dict.fromkeys(cells,'minecraft:smooth_stone'))
            assigned = json.loads((out/'placements.json').read_text())['blocks']
            self.assertEqual({(r['x'],r['y'],r['z']) for r in assigned},cells)
            self.assertTrue(all(r['material_source']=='fallback' for r in assigned))
            summary = json.loads((out/'build-manifest.json').read_text())['placement']['material_assignment']
            self.assertEqual(summary['fallback_reason'],'verified atlas unavailable')
            self.assertEqual(summary['sources'], {'fallback':2})

    def test_uncertain_camera_replaces_prior_nonempty_build(self):
        with TemporaryDirectory() as temp:
            out = Path(temp) / 'build'
            out.mkdir()
            image = Path(temp) / 'flat.png'
            Image.new('RGB', (160, 100), 'gray').save(image)
            write_pack(out / 'reconstruction.zip', {(0, 0, 0): 'stone'}, JAVA_26_3_FORMAT)
            (out / 'placements.json').write_text('{"blocks":[{"x":0,"y":0,"z":0}]}')
            self.assertEqual(len(decoded_pack(out / 'reconstruction.zip')), 1)

            # Isolate learned inference. The camera, counter, placement and pack writer run normally.
            with patch('engine.surface_fields.infer', return_value=np.ones((10, 16))), \
                    patch('engine.surface_model.proposals_for', return_value=[]):
                build(image, '26.3', out, outlines='mask-first', boundary='regions-first', through='placement')

            manifest = json.loads((out / 'build-manifest.json').read_text())
            self.assertEqual(manifest['artifact'], 'reconstruction.zip')
            self.assertEqual(manifest['placement']['camera_status'], 'uncertain')
            self.assertEqual(manifest['placement']['placed_blocks'], 0)
            self.assertEqual(manifest['placement']['final_refinement'],
                             dict(before=0, added=0, removed=0, after=0))
            for key in ('duplicate_side_cells', 'textured_side_cells', 'grounded_side_cells', 'shared_side_cells'):
                self.assertEqual(manifest['placement'][key], 0)
            outputs = manifest['artifacts']
            self.assertEqual(decoded_pack(out / outputs['datapack']), {})
            placements = json.loads((out / outputs['placements']).read_text())
            self.assertEqual(placements['blocks'], [])
            self.assertEqual(placements['camera_status'], 'uncertain')
            for stage in ('surfaces', 'counts'):
                self.assertTrue((out / outputs[stage]).is_file())
