import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import surface_model
from engine.atlas_count import _prepare, _verify_material_index
from engine.atlas_phase import ATLAS_SHA256, register_phase


def example_atlas():
    tile = np.random.default_rng(142).integers(40, 220, (16, 16, 3), dtype=np.uint8)
    # Synthetic pixels exercise the matcher; the production loader checks actual file digests.
    provenance = dict(version='26.3', atlas_sha256=ATLAS_SHA256)
    return _prepare(tile[None], ['stone'], [['minecraft:stone']], provenance)


class RestrictedPhaseTests(unittest.TestCase):
    def setUp(self):
        self.atlas = example_atlas()
        rgb = np.roll(np.tile(self.atlas.tiles[0], (3, 5, 1)), (4, 7), axis=(0, 1))
        mask = np.ones(rgb.shape[:2], bool)
        mask[12:28, 12:35] = False
        rgb[~mask] = [255, 0, 255]
        self.sample = dict(rgb=rgb, mask=mask)

    def test_recovers_only_missing_offsets_and_preserves_counter_evidence(self):
        for old_phase, expected in (([None, None], [7/16, 4/16]), ([.2, None], [.2, 4/16]),
                                    ([None, .6], [7/16, .6]), ([.2, .6], [.2, .6])):
            with self.subTest(phase=old_phase):
                original = dict(counts=[5, 3], phase=old_phase.copy(), evidence={'source': 'counter'})
                result = register_phase(self.sample, original, self.atlas)
                self.assertEqual(result['phase'], expected)
                self.assertEqual(result['counts'], [5, 3])
                self.assertEqual(result['evidence'], original['evidence'])
                self.assertEqual(original['phase'], old_phase)

    def test_unmeasured_spacing_is_never_inferred_even_with_a_matching_sprite(self):
        for bad in (None, 0, -1, float('nan'), float('inf')):
            for counts in ([5, bad], [bad, 3]):
                with self.subTest(counts=counts):
                    original = dict(counts=counts, phase=[None, None])
                    result = register_phase(self.sample, original, self.atlas)
                    self.assertEqual(result['phase'], [None, None])
                    np.testing.assert_equal(result['counts'], counts)

    def test_flat_or_insufficient_visible_evidence_leaves_grid_unresolved(self):
        original = dict(counts=[5, 3], phase=[None, None])
        sparse = np.zeros_like(self.sample['mask'])
        sparse[4:8, 4:8] = True
        for sample in (dict(self.sample, rgb=np.full_like(self.sample['rgb'], 123)),
                       dict(self.sample, mask=sparse)):
            result = register_phase(sample, original, self.atlas)
            self.assertEqual(result['phase'], [None, None])

    def test_texture_with_indistinguishable_half_block_shifts_cannot_locate_a_seam(self):
        tile = np.tile(self.atlas.tiles[0, :, :8], (1, 2, 1))
        atlas = _prepare(tile[None], ['stripe'], [['minecraft:stripe']], self.atlas.provenance)
        sample = dict(self.sample, rgb=np.tile(tile, (3, 5, 1)))
        result = register_phase(sample, dict(counts=[5, 3], phase=[None, None]), atlas)
        self.assertEqual(result['phase'], [None, 0])


class AtlasFallbackBuildTests(unittest.TestCase):
    def test_regenerated_catalog_can_omit_blocks_but_cannot_relabel_them(self):
        header = dict(version='26.3', client_sha256='client', atlas_sha256='art')
        calibrated = dict(header, choices={'south': {'stone': ['stone'], 'bricks': ['bricks']}})
        local = dict(header, choices={'south': {'bricks': ['bricks']}})
        _verify_material_index(local, calibrated)
        local['choices']['south']['bricks'] = ['stone']
        with self.assertRaisesRegex(ValueError, 'Unexpected block mapping'):
            _verify_material_index(local, calibrated)

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / 'example.png'
        Image.new('RGB', (96, 64), 'gray').save(self.image)

    def run_build(self, name, phases=((.25, None), (0, 0)), **options):
        planes = []
        for index, left in enumerate((0, 48)):
            right = left + 48
            planes.append(dict(id=f'wall-{index}', quad=[[left, 0], [right, 0], [right, 64], [left, 64]],
                               support=dict(size=[96, 64], runs=[[y, left, right] for y in range(64)])))
        measured = [dict(counts=[3, 4], phase=list(phase)) for phase in phases]
        out = self.root / name
        # Replace upstream mask and spacing estimates; use real sampling, grouping and artifact writers.
        with patch('engine.surface_fields.infer'), patch.object(surface_model, 'proposals_for', return_value=planes), \
                patch('engine.texture_guided_count.count', side_effect=measured):
            surface_model.build(self.image, '26.3', out, through='counts', boundary='regions-first', **options)
        return tuple(json.loads((out / filename).read_text())
                     for filename in ('surfaces.json', 'counts.json', 'build-manifest.json'))

    def test_missing_corrupt_or_incompatible_atlas_preserves_geometry(self):
        expected_model, expected_counts, _ = self.run_build('control')
        for name, metadata in (('missing', None), ('corrupt', '{'), ('wrong-version', '{"version":"1.0"}')):
            with self.subTest(atlas=name):
                atlas_path = self.root / f'{name}-atlas'
                if metadata is not None:
                    atlas_path.mkdir()
                    (atlas_path / 'atlas.json').write_text(metadata)
                model, counts, manifest = self.run_build(name, material_match='shape', atlas_path=atlas_path)
                self.assertEqual(model, expected_model)
                for row, expected in zip(counts['surfaces'], expected_counts['surfaces'], strict=True):
                    for key in ('counts', 'phase', 'dimensions'):
                        self.assertEqual(row[key], expected[key])
                    self.assertEqual([cell['quad'] for cell in row['visible_cells']],
                                     [cell['quad'] for cell in expected['visible_cells']])
                self.assertTrue(counts['block_groups'])
                self.assertFalse(any((group.get('material') or {}).get('block_id') for group in counts['block_groups']))
                self.assertEqual(manifest['atlas_phase']['status'], 'unavailable')

    def test_shape_without_a_prepared_atlas_preserves_the_ordinary_result(self):
        with patch('engine.atlas_count.prepared_atlas_paths', return_value=[]), \
                patch('engine.atlas_count.make_atlas') as regenerate:
            _, counts, manifest = self.run_build('no-prepared-atlas', material_match='shape')
        regenerate.assert_not_called()
        self.assertEqual(manifest['atlas_phase']['status'], 'unavailable')
        self.assertEqual(counts['surfaces'][0]['phase'], [.25, None])
        self.assertTrue(counts['surfaces'][1]['visible_cells'])
        self.assertFalse(any((group.get('material') or {}).get('block_id') for group in counts['block_groups']))

    def test_material_run_automatically_recovers_offset_and_ranks_newly_visible_crops(self):
        atlas = example_atlas()
        pixels = np.roll(np.tile(atlas.tiles[0], (4, 6, 1)), (4, 7), axis=(0, 1))
        Image.fromarray(pixels).save(self.image)
        with patch('engine.atlas_count.load_atlas', return_value=atlas):
            _, counts, manifest = self.run_build('registered', material_match='atlas', atlas_path=self.root)
        first, second = counts['surfaces']
        self.assertEqual(first['counts'], [3, 4])
        self.assertEqual(first['phase'], [.25, 4/16])
        self.assertEqual(second['phase'], [0, 0])
        self.assertTrue(first['visible_cells'])
        self.assertTrue(all(cell['material']['block_id'] == 'minecraft:stone' for cell in first['visible_cells']))
        self.assertEqual(manifest['atlas_phase']['offsets_added'], 1)

    def test_discovery_skips_damaged_copy_and_uses_next_prepared_atlas(self):
        dirs = [self.root/'incomplete', self.root/'calibrated']
        with patch('engine.atlas_count.prepared_atlas_paths', return_value=dirs), \
                patch('engine.atlas_count.load_atlas', side_effect=[ValueError('damaged atlas'), example_atlas()]):
            _, counts, manifest = self.run_build('search', material_match='atlas')
        self.assertEqual(manifest['atlas']['directory'], str(dirs[1]))
        self.assertEqual(manifest['atlas_search_errors'][0]['path'], str(dirs[0]))
        self.assertEqual(counts['atlas_phase']['status'], 'enabled')

    def test_required_counter_tries_all_discovered_copies_but_respects_explicit_path(self):
        dirs = [self.root/'damaged', self.root/'valid']
        atlas = example_atlas()
        with patch('engine.atlas_count.prepared_atlas_paths', return_value=dirs), \
                patch('engine.atlas_count.load_atlas', side_effect=[ValueError('bad checksum'), atlas]):
            found, _, path, failures = surface_model._material_assets('26.3', 'atlas-repeat', 'atlas', None)
        self.assertIs(found, atlas)
        self.assertEqual(path, dirs[1])
        self.assertEqual(failures[0]['path'], str(dirs[0]))
        with patch('engine.atlas_count.prepared_atlas_paths') as search, \
                patch('engine.atlas_count.load_atlas', side_effect=ValueError('bad checksum')):
            with self.assertRaisesRegex(ValueError, 'bad checksum'):
                surface_model._material_assets('26.3', 'atlas-repeat', 'atlas', dirs[0])
            search.assert_not_called()

    def test_registration_exception_preserves_counter_and_working_material_matcher(self):
        atlas = example_atlas()
        pixels = np.roll(np.tile(atlas.tiles[0], (4, 6, 1)), (4, 7), axis=(0, 1))
        Image.fromarray(pixels).save(self.image)

        def fail_on_left_wall(sample, result, atlas):
            if sample['quad'][0][0] == 0:
                raise RuntimeError('broken registration')
            return register_phase(sample, result, atlas)

        with patch('engine.atlas_count.load_atlas', return_value=atlas), \
                patch('engine.blocker_shape.ShapeMatcher') as matcher_type, \
                patch('engine.atlas_phase.register_phase', side_effect=fail_on_left_wall):
            matcher = matcher_type.return_value
            matcher.mapping = {'stone': ['stone']}
            matcher.rank_patch.side_effect = lambda patch: dict(block_id='stone', texture='stone', ranked_ids=['stone'])
            _, counts, manifest = self.run_build('phase-error', phases=[[.25, None], [None, 0]],
                                                 material_match='shape', atlas_path=self.root)
        self.assertEqual(manifest['atlas_phase']['status'], 'partial')
        self.assertEqual(manifest['atlas_phase']['surface_errors'], 1)
        self.assertEqual(manifest['atlas_phase']['offsets_added'], 1)
        first, second = counts['surfaces']
        self.assertEqual(first['counts'], [3, 4])
        self.assertEqual(first['phase'], [.25, None])
        self.assertEqual(first['visible_cells'], [])
        self.assertEqual(second['phase'], [7/16, 0])
        self.assertTrue(second['visible_cells'])
        self.assertTrue(all(cell['material']['block_id'] == 'minecraft:stone' for cell in second['visible_cells']))
