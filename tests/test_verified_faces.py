import unittest

import numpy as np

from engine.atlas_count import _prepare
from engine.atlas_plane_depth import _patch_scores, _texel_samples
from engine.partial_atlas_face import verify
from engine.visible_placement import _frontmost_cells
from engine.visible_ray_filter import fully_occluded_cells


class VerifiedFacesTests(unittest.TestCase):
    def test_trailing_image_pixels_do_not_supply_missing_mask_evidence(self):
        tiles = np.random.default_rng(21).integers(0, 256, (30, 16, 16, 3), dtype=np.uint8)
        atlas = _prepare(tiles, list(map(str, range(30))), [], {})
        projection = np.array([[32., 0, 49, 0], [0, 32., 49, 0], [0, 0, 1., 0]])
        cell = (0, 0, 2)
        for size in (65, 66):
            with self.subTest(size=size):
                rgb = np.zeros((size, size, 3), dtype=np.uint8)
                rgb[49:65, 49:65] = tiles[0]
                mask = np.ones((16, 16), dtype=bool)
                _texel_samples(mask, rgb, np.linalg.inv(projection[:, :3]), 2, [0, 1])
                self.assertFalse(_patch_scores(mask, np.array([[0, 0]]), 2, [0, 1], 2,
                                               projection, rgb, atlas))
                self.assertTrue(verify(rgb, atlas, mask, projection, cell, 2, -1))
                mask[:] = False
                self.assertFalse(verify(rgb, atlas, mask, projection, cell, 2, -1))

    def test_front_axis_tracks_owner_and_ray_check_keeps_visible_cell(self):
        k = np.array([[75., 0, 48], [0, 75., 48], [0, 0, 1.]])
        orientation = np.diag([1., -1., 1.])
        eye = np.array([.5, .5, -5])
        projection = k @ np.column_stack((orientation, -orientation @ eye))
        cells = [(0, 0, 0), (0, 0, 1)]
        owner, axes = _frontmost_cells(cells, projection, eye, (24, 24), face_axes=True)
        self.assertTrue((owner == 0).any())
        self.assertTrue((axes[owner == 0] == 2).all())
        self.assertFalse((owner == 1).any())
        self.assertEqual(fully_occluded_cells((96, 96), projection, cells), {(0, 0, 1)})

    def test_coarse_sample_outside_odd_width_cannot_keep_offscreen_cell(self):
        k = np.array([[2., 0, 65], [0, 2., 30], [0, 0, 1.]])
        eye = np.array([.5, .5, -1.])
        projection = np.column_stack((k, -k @ eye))
        self.assertEqual(fully_occluded_cells((65, 65), projection, {(1, 0, 0)}), {(1, 0, 0)})
