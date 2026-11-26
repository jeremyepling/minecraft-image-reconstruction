import unittest

import numpy as np
from PIL import Image

from engine.geometry import UNIT, project
from engine.surface_pitch import _axis, fit


def repeating_surface(two_courses=False):
    rng = np.random.default_rng(489)
    tile = rng.integers(132, 165, (16, 16, 3), dtype=np.uint8)
    tile[:, [0, -1]] = 55
    tile[[0, -1]] = 55
    if two_courses:
        tile[[7, 8]] = 55
    field = np.tile(tile, (7, 8, 1)).repeat(4, axis=0).repeat(4, axis=1)
    return Image.fromarray(field)


class SurfacePitchTest(unittest.TestCase):
    def test_periodic_stripes_without_off_grid_samples_are_unresolved(self):
        line = np.tile([0., 0., 0., 200., 200., 200.], 8)
        rgb = np.tile(line[None, :, None], (10, 1, 3)).astype(np.float32)
        result = _axis(rgb, np.ones(rgb.shape[:2], dtype=bool), len(line))
        self.assertIsNone(result['count'])

    def test_fractional_spans_and_seams_of_an_off_phase_crop(self):
        image = repeating_surface()
        start, end = np.array([25., 46.]), np.array([379., 365.])
        quad = start + UNIT * (end - start)
        result = fit(image, quad)
        np.testing.assert_allclose(result["counts"], (end - start)/64, atol=.03)
        grid = result["grid"]
        self.assertIsNotNone(grid)
        expected_phase = (-start/64) % 1.
        np.testing.assert_allclose(grid["phase_blocks"], expected_phase, atol=.04)
        first_corner = project(np.array(grid["grid_to_image"]), [[0, 0]])[0]
        np.testing.assert_allclose(first_corner, [64, 64], atol=2.6)
        for axis, key in enumerate(("u_lines", "v_lines")):
            pixels = start[axis] + (end[axis] - start[axis]) * np.array(grid[key])
            np.testing.assert_allclose(pixels, np.round(pixels/64)*64, atol=2.6)

    def test_metered_pitch_can_survive_ambiguous_half_course_phase(self):
        image = repeating_surface(two_courses=True)
        result = fit(image, np.array([25., 46.]) + UNIT * [354., 319.])
        np.testing.assert_allclose(result["counts"], [354/64, 319/64], atol=.03)
        self.assertIsNone(result["grid"])

    def test_untextured_or_sparse_stripes_do_not_establish_a_block(self):
        blank = Image.new("RGB", (352, 256), (171, 172, 173))
        sparse = np.full((256, 352, 3), 165, np.uint8)
        sparse[::16] = 58
        y, x = np.indices(sparse.shape[:2])
        mortar = (y % 32 < 4) | ((x + (y // 32 % 2)*32) % 64 < 4)
        bricks = Image.fromarray(np.uint8(np.where(mortar, 58, 165)))
        for image in (blank, Image.fromarray(sparse), bricks):
            result = fit(image, UNIT * image.size)
            self.assertIsNone(result["counts"])
            self.assertIsNone(result["grid"])

    def test_minified_unobservable_patch_abstains(self):
        result = fit(repeating_surface(), np.array([60, 60]) + UNIT * [7, 12])
        self.assertIsNone(result["grid"])
        self.assertIsNone(result["counts"])

    def test_surrounding_image_cannot_supply_a_phase_or_pitch(self):
        wall = np.array(repeating_surface())
        blank = np.full_like(wall, 168)
        wall[80:270, 95:321] = blank[80:270, 95:321]
        quad = [95, 80] + UNIT * [226, 190]
        self.assertEqual(fit(Image.fromarray(wall), quad), fit(Image.fromarray(blank), quad))
