from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from engine import visible_roof_rectangle as roof


class VisibleRoofRectangleTests(TestCase):
    def test_contiguous_requires_both_shared_patch_and_new_pixels(self):
        with patch.object(roof,'recover',return_value=(set(),{})) as recover:
            roof.contiguous(None,[],None,{},set(),atlas_path='custom')
            self.assertEqual(recover.call_args.kwargs,dict(texture=False,unclaimed=True,shared_patch=True,atlas_path='custom'))

    def test_only_measured_adjacent_arms_bound_probe_cells(self):
        corner=(0,3,0)
        across={corner,(1,3,0),(2,3,0),(4,3,0)}
        away={corner,(0,3,1),(0,3,2)}
        actual, count=roof.rects({0:away,2:across},(-2,5,-2),across|away)
        self.assertEqual(count,1)
        self.assertEqual(actual,{(1,3,1),(1,3,2),(2,3,1),(2,3,2)})
        result, count=roof.rects({0:away-{corner},2:across},(-2,5,-2),across|away)
        self.assertEqual((result,count),(set(),0))

    def test_unclaimed_pixels_and_horizontal_patch_must_support_each_new_cell(self):
        c=(1,3,1)
        shape=(3,3)
        pixel_grid=np.ones(shape,bool)
        ctx=SimpleNamespace(native=None,shape=shape,depth=np.zeros(shape),
                            masks={1:pixel_grid.copy()})
        joint=(np.zeros(shape,int),np.ones(shape,int),{c:0})
        region=(slice(0,3),slice(0,3))
        hit=(region,pixel_grid,np.ones(shape))
        with patch.object(roof,'_sample_face',return_value=hit):
            self.assertIsNotNone(roof.top_witness(ctx,c,joint,True,({c:{2}},np.full(shape,2))))
            self.assertIsNone(roof.top_witness(ctx,c,joint,True,({c:{2}},np.full(shape,3))))
            ctx.depth[0,:]=.5
            self.assertIsNone(roof.top_witness(ctx,c,joint,True))
            ctx.depth[:]=0
            joint[1][:]=0
            self.assertIsNone(roof.top_witness(ctx,c,joint,True))

    def test_missing_camera_preserves_build(self):
        cells={(0,0,0)}
        result,status=roof.contiguous(None,[],{},dict(P=None),cells)
        self.assertEqual(result,cells)
        self.assertEqual(status['accepted'],0)
