"""Check that region evidence requires mortar in both directions on the same surface."""
import unittest

import numpy as np

from engine.courses import course_evidence


def sample(pattern, omitted=(), depth_fixture=False):
    # Each grid cell spans a full vertical atlas repeat. The staged material
    # detector must also work when the cropped cell straddles a mortar course.
    cols, rows, size = 9, 8, 12
    width, height = cols * size, rows * size
    yy, xx = np.mgrid[:height, :width]
    rng = np.random.default_rng(40)
    tone = 150 + rng.normal(0, 1.1, (height, width)) + 17 * xx / width
    horizontal = ((yy + 2) % 8) == 0
    width_px = 20
    stagger = (((yy + 2) // 8) % 2) * (width_px // 2)
    if pattern == 'stripe':
        dark = horizontal
    elif pattern == 'grid':
        dark = horizontal | ((xx + 4) % width_px == 0)
    elif pattern in ('brick', 'mixed'):
        dark = horizontal | ((xx + stagger + 4) % width_px == 0)
        if pattern == 'mixed':
            dark &= xx < size * 4
    elif pattern == 'noise':
        tone += rng.normal(0, 18, tone.shape)
        dark = False
    else:
        dark = False
    shade = tone - 68 * dark
    image = np.stack((shade + 18, shade + 3, shade - 4), axis=-1).clip(0, 255).astype('uint8')
    plane = dict(columns=cols, rows=rows, quad=[[0,0],[width,0],[width,height],[0,height]],
                 grid_to_image=[[size,0,0],[0,size,0],[0,0,1]])
    cells = [dict(ij=(i,j)) for j in range(rows) for i in range(cols) if (i,j) not in omitted]
    if depth_fixture:
        plane.update(origin=[0,0,5], facing='south')
        for cell in cells:
            i,j = cell['ij']
            cell['xyz'] = [i, rows-1-j, 4 if i == 4 else 5]
        # A visible return can duplicate the coordinates and ij of a facade cell
        # while sampling a different quad; it must never replace that cell's ID.
        cells.append(dict(ij=(1,3), xyz=[1,4,5], facing='east', depth_surface='visible return'))
        cells.append(dict(ij=(3,3), xyz=[3,4,5], facing='south', depth_surface='recess back'))
    return cells, course_evidence(image, plane, cells)


class CoursesTest(unittest.TestCase):
    def test_staggered_ridges_under_warm_light(self):
        cells, evidence = sample('brick')
        self.assertGreaterEqual(len(evidence), int(len(cells) * .8))
        central = next(k for k,c in enumerate(cells) if c['ij'] == (4,4))
        self.assertAlmostEqual(evidence[central]['course_height_cells'], 8/12, delta=.09)
        self.assertAlmostEqual(evidence[central]['brick_width_cells'], 20/12, delta=.2)

    def test_negative_controls(self):
        for kind in ('stripe', 'grid', 'flat', 'noise'):
            with self.subTest(kind=kind):
                _, evidence = sample(kind)
                self.assertFalse(evidence)

    def test_adjacent_surface_and_missing_cells(self):
        holes = {(2,4),(3,4)}
        cells, evidence = sample('mixed', holes)
        self.assertTrue(evidence)
        for k, match in evidence.items():
            i,j = cells[k]['ij']
            self.assertLess(i, 4)
            left, top, right, bottom = match['patch_uv']
            self.assertTrue(left <= i < right and top <= j < bottom)
            self.assertFalse(any(left <= x < right and top <= y < bottom for x,y in holes))

    def test_parallel_setback_and_duplicate_returns(self):
        cells, evidence = sample('brick', depth_fixture=True)
        self.assertTrue(evidence)
        self.assertNotIn(-1, evidence)
        for k, match in evidence.items():
            self.assertEqual(cells[k]['xyz'][2], 5)
            self.assertNotIn('depth_surface', cells[k])
            left, _, right, _ = match['patch_uv']
            self.assertFalse(left <= 4 < right)
        expected = next(k for k, cell in enumerate(cells) if cell['ij'] == (1,3))
        self.assertIn(expected, evidence)
