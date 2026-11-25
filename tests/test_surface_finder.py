"""Exercise ordinary surface builds, preserved masks and B's standalone spacing pass."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.score_saved_planes import ROUND, read, score_case
from engine import legacy_count, surface_model
from engine.surface_order import order


class SurfaceFinderTests(unittest.TestCase):
    def test_default_keeps_a_chartless_wall_and_its_opening_without_counting(self):
        normal = np.ones((6, 6, 3), dtype=np.float32)
        depth = np.ones((6, 6), dtype=np.float32)
        wall = dict(id='mask-0', support=dict(size=[6, 6], runs=[[0, 0, 6], [1, 0, 1], [1, 5, 6]]),
                    geometry_status='mask only')
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'original.png'
            Image.new('RGB', (64, 64)).save(path)
            with patch('engine.surface_fields.infer', side_effect=[normal, depth]) as infer, \
                    patch('engine.mask_first.regions', return_value=[wall]) as regions, \
                    patch.object(surface_model, 'pitch') as pitch:
                surface_model.build(path, '26.3', Path(temp))
            self.assertEqual([call.args[1] for call in infer.call_args_list], ['normal', 'depth'])
            self.assertIs(regions.call_args.args[1], normal)
            self.assertIs(regions.call_args.args[2], depth)
            pitch.assert_not_called()
            model = json.loads((Path(temp)/'surfaces.json').read_text())
            manifest = json.loads((Path(temp)/'build-manifest.json').read_text())
            plane, = model['planes']
            with Image.open(Path(temp)/plane['source_crop']['path']) as crop:
                self.assertEqual(crop.mode, 'RGBA')
                self.assertEqual(crop.getpixel((30, 15))[3], 0)
            context = plane['context_crop']
            with Image.open(Path(temp)/context['path']) as crop:
                self.assertEqual((crop.mode, crop.size), ('RGB', (64, 29)))
            self.assertEqual(context['mask_xyxy_in_crop'], [0, 0, 64, 21])
            self.assertEqual(context['margins_ltrb_px'], [0, 0, 0, 8])
            self.assertEqual(context['requested_margin_px'], 8)
        self.assertEqual(model['footprint'], 'support')
        self.assertEqual({k: plane[k] for k in wall}, wall)
        self.assertIsNone(plane['sampling_chart'])
        self.assertEqual([r['kind'] for r in plane['outline']['rings']], ['outer'])
        self.assertEqual(plane['source_crop']['xyxy'], [0, 0, 64, 21])
        self.assertEqual(manifest['artifacts']['source_crops'], 'surface-crops/')
        self.assertEqual(manifest['stages'], {'outlines': 'mask-first','surface_boundary':'color-edges'})
        self.assertEqual(set(manifest['model_sha256']), {'normal', 'depth'})
        dev = next(r for r in read(ROUND/'cases.json') if r['id'].startswith('dev-'))
        self.assertEqual(score_case(ROUND, dev, model['planes'], use_masks=True)['proposed'], 1)

    def test_depth_order_can_use_masks_without_four_corners(self):
        shape = (48, 48)
        def wall(name, start, stop):
            return dict(id=name, support=dict(size=list(shape), runs=[[y, start, stop] for y in range(48)]))
        depth = np.ones(shape, dtype=np.float32)
        depth[:, :24] = 4.
        result = order([wall('front', 0, 24), wall('return', 24, 48)], shape, depth, use_masks=True)
        self.assertEqual([(p['near'], p['far']) for p in result], [('front', 'return')])

    def test_explicit_historical_outline_keeps_its_chart_processing(self):
        part = dict(id='wall', quad=[[2, 2], [30, 2], [30, 30], [2, 30]])
        with TemporaryDirectory() as temp:
            path = Path(temp)/'original.png'
            Image.new('RGB', (32, 32)).save(path)
            with patch.object(surface_model, 'proposals_for', return_value=[part]), \
                    patch('engine.surface_fields.infer') as infer:
                surface_model.build(path, '26.3', Path(temp), outlines='edges+shade')
            infer.assert_not_called()
            row, = json.loads((Path(temp)/'surfaces.json').read_text())['planes']
            stages = json.loads((Path(temp)/'build-manifest.json').read_text())['stages']
        self.assertEqual(stages['pitch'], 'repeat')
        self.assertEqual(stages['order'], 'geometry')
        self.assertEqual(row['quad'], part['quad'])
        self.assertIn('pitch', row)

    def test_legacy_counter_preserves_partial_phase_and_empty_result(self):
        evidence = dict(phase_by_axis=[.2, None])
        sample = dict(source_image=Image.new('RGB', (32, 32)), quad=[[0, 0], [32, 0], [32, 32], [0, 32]])
        original = dict(counts=[2.5, 3.], evidence=evidence, grid=None, confidence=.8, reason='pitch only')
        with patch('engine.legacy_count.fit', return_value=original) as fit:
            result = legacy_count.count(sample)
        fit.assert_called_once_with(sample['source_image'], sample['quad'], '26.3')
        self.assertEqual(result['counts'], original['counts'])
        self.assertEqual(result['phase'], [.2, None])
        self.assertEqual(legacy_count.count(None)['counts'], [None, None])
