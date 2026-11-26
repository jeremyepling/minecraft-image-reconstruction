import unittest

import numpy as np

from benchmark.surface_topology import junctions, opening_mask, opening_coverage, summarize


def inverse_depth_chart(ax, ay, offset):
    return np.linalg.inv([[1.,0.,0.], [0.,1.,0.], [ax,ay,offset]])


class SurfaceTopologyTest(unittest.TestCase):
    def test_open_to_floor_hole_and_background_exclusions(self):
        ids = np.zeros((12,14), dtype=int)
        ids[2:4, 2:12] = 1
        ids[4:10, 3:5] = 2
        ids[4:10, 9:11] = 3
        expected = np.zeros_like(ids, dtype=bool)
        expected[4:10, 5:9] = True
        actual = opening_mask(ids)
        np.testing.assert_array_equal(actual, expected)
        left_half = {"quad":[[20,16],[28,16],[28,40],[20,40]]}
        s = opening_coverage(actual, [left_half,left_half])
        self.assertEqual(s, dict(opening_pixels=24,filled_pixels=12,filled_fraction=.5))
        # An entirely occupied rear surface or an unroofed slot is not visible background under a roof.
        ids[expected] = 4
        self.assertFalse(opening_mask(ids).any())
        ids[expected] = 0
        ids[2:4] = 0
        self.assertFalse(opening_mask(ids).any())

    def test_fold_junction_requires_depth_agreement_at_both_endpoint_rays(self):
        left = dict(id="left", quad=[[10,20],[40,20],[40,80],[10,80]])
        right = dict(id="right", quad=[[42,20],[70,20],[70,80],[42,80]])
        model = dict(planes=[left,right],shared_edges=[dict(planes=["left","right"],corners=[[1,2],[0,3]])])
        matches = [dict(proposal_index=0,patch_id=5), dict(proposal_index=1,patch_id=7)]
        t = dict(charts={5:inverse_depth_chart(0,0,.1), 7:inverse_depth_chart(.001,0,.059)},
                 patches={5:dict(normal=[0,0,1]),7:dict(normal=[1,0,0])})
        fold = junctions(t, model, matches)
        self.assertEqual((fold["joins"],fold["matched_distinct"],fold["measured"],fold["supported"]),(1,1,1,1))
        edge = fold["edges"][0]
        self.assertEqual(edge["endpoint_xy"], [[41.,20.],[41.,80.]])
        np.testing.assert_allclose(edge["endpoint_delta_blocks"],[0.,0.],atol=1e-10)
        self.assertTrue(edge["nonparallel"])

        # Both segments coincide in the image. Truth depth differs by two blocks
        # at the lower endpoint, although it agrees exactly at the upper one.
        t["charts"][7] = inverse_depth_chart(0.,-1/3600.,.1+20/3600.)
        occlusion = junctions(t, model, matches)
        self.assertEqual((occlusion["matched_distinct"],occlusion["measured"],occlusion["supported"]),(1,1,0))
        np.testing.assert_allclose(occlusion["edges"][0]["endpoint_delta_blocks"],[0.,2.],atol=1e-10)
        missing = junctions(t, model, matches[:1])
        self.assertEqual((missing["joins"],missing["matched_distinct"],missing["measured"]),(1,0,0))
        self.assertIsNone(missing["edges"][0]["supported"])
        totals = summarize([dict(openings={},junctions={"repeat":x}) for x in [fold,occlusion,missing]])
        self.assertEqual(totals["junctions"]["repeat"]["supported_per_measured"],.5)
        self.assertAlmostEqual(totals["junctions"]["repeat"]["supported_per_join"],1/3)
