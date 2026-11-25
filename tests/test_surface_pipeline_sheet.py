import unittest

import numpy as np

from benchmark.surface_pipeline_sheet import _straight_photo


class PipelineSheetGridTest(unittest.TestCase):
    def test_only_located_axis_is_drawn_and_openings_stay_empty(self):
        rgb = np.full((20,40,3),72,np.uint8)
        mask = np.ones(rgb.shape[:2],bool)
        mask[6:15,5:36] = False
        clean,grid,lines = _straight_photo(dict(rgb=rgb,mask=mask),
                                           dict(counts=[4.,3.],phase=[.25,None]))
        a,b = np.asarray(clean),np.asarray(grid)
        drawn = np.any(a!=b,axis=2)
        self.assertTrue(drawn.any())
        self.assertFalse(drawn[~mask].any())
        self.assertEqual(lines,[[.0625,.3125,.5625,.8125],[]])
        self.assertEqual(np.count_nonzero(drawn[0]),4)
        self.assertFalse(np.any(drawn[:,7]))

    def test_known_spacing_does_not_create_grid_lines_without_an_origin(self):
        rgb = np.full((12,23,3),110,np.uint8)
        a,b,lines = _straight_photo(dict(rgb=rgb,mask=np.ones(rgb.shape[:2],bool)),
                                    dict(counts=[5.,2.],phase=[None,None]))
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        self.assertEqual(lines,[[],[]])
