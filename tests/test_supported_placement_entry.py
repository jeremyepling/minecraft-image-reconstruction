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
            outputs = manifest['artifacts']
            self.assertEqual(decoded_pack(out / outputs['datapack']), {})
            placements = json.loads((out / outputs['placements']).read_text())
            self.assertEqual(placements['blocks'], [])
            self.assertEqual(placements['camera_status'], 'uncertain')
            for stage in ('surfaces', 'counts'):
                self.assertTrue((out / outputs[stage]).is_file())
