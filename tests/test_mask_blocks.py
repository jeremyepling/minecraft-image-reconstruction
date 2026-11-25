import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark.mask_blocks import (_blocks, _cell_map, _footprints, _plane_from_chart,
                                   _write_read, registered)


class MaskBuilding(unittest.TestCase):
    def test_used_cells_follow_support_including_holes(self):
        P = np.array([[8., 0., 0., 0.], [0., 8., 0., 0.], [0., 0., 1., 0.]])
        # At z=1 each block face projects to 2x2 samples. The positive-going ray
        # hits the negative-z face of the block at z=1, leaving z=0 empty.
        plane = (2, 1, -1)
        quad = [[0, 0], [24, 0], [24, 24], [0, 24]]
        all_cells = _footprints([dict(quad=quad)], (24, 24), False)[0]
        mask = all_cells.copy()
        mask[2:4, 2:4] = False
        mask[0, 0] = False  # Even one unsupported sample vetoes this block.
        carriers = [(all_cells, plane, 'supported')]
        baseline, _, _ = _blocks([all_cells], carriers, P, {})
        candidate, counts, _ = _blocks([mask], carriers, P, {})
        self.assertEqual(len(baseline), 9)
        self.assertEqual(candidate, baseline-{(1, 1, 1), (0, 0, 1)})
        self.assertEqual(counts, {'emitted': 1})
        self.assertEqual(_cell_map(P, mask.shape, (2, 1, 1)).occupied(mask),
                         {(x, y, 0) for x, y, z in candidate})

    def test_uncertain_camera_scale_and_depth_stay_unknown(self):
        mask = np.ones((2, 2), bool)
        self.assertEqual(_blocks([mask], [], None, {})[:2], (set(), {'unresolved_camera': 1}))
        E = np.column_stack((np.eye(3), np.zeros(3)))
        self.assertEqual(_plane_from_chart(dict(units='relative'), E), (None, 'unresolved_scale'))
        chart = dict(units='blocks', normal_camera=[0, 0, -1],
                     corners_camera=[[0, 0, 4], [1, 0, 4], [1, 1, 6], [0, 1, 6]])
        self.assertEqual(_plane_from_chart(chart, E), (None, 'unresolved_depth'))

    def test_registration_and_pack_readback(self):
        T = np.diag([-1., 1., 1., 1.])
        T[0, 3] = 7
        self.assertEqual(registered({(2, 3, 4)}, T), {(4, 3, 4)})
        with tempfile.TemporaryDirectory() as tmp:
            for cells in (set(), {(-65, -1, -3), (34, 0, 4), (35, 3, 5)}):
                self.assertEqual(_write_read(Path(tmp)/'pack.zip', cells), cells)

    def test_region_without_outline_keeps_its_support(self):
        region = dict(support=dict(size=[2, 2], runs=[[0, 0, 2], [1, 0, 1]]))
        mask, = _footprints([region], (8, 8), True)
        np.testing.assert_array_equal(mask, [[1, 1], [1, 0]])


if __name__ == '__main__':
    unittest.main()
