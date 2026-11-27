import unittest

import numpy as np
from PIL import Image

from engine.guarded_wall_outlines import _remap_junctions, refine
from engine.plane_consensus import row_runs, run_mask


def part(name, mask, parent='wall', normal=(0.,0.,-1.)):
    return dict(id=name, parent_id=parent, normal_camera=list(normal),
                support=dict(size=list(mask.shape[::-1]), runs=row_runs(mask)))


class GuardedWallOutlinesTest(unittest.TestCase):
    def setUp(self):
        y,x = np.indices((180,240))
        self.mask = (x>8)&(x<230)&(y>9)&(y<170)
        self.mask[100:115,180:191] = False
        self.left = x < y+20
        self.image = Image.new('RGB', (240,180), (110,110,110))

    def test_union_preserves_unknown_openings_and_records_real_native_area(self):
        parts = [part('a', self.mask & self.left), part('b', self.mask & ~self.left)]
        after, records = refine(self.image, parts)
        self.assertEqual(len(after),1)
        np.testing.assert_array_equal(run_mask(after[0]['support']), self.mask)
        self.assertEqual(records[0]['output_area_px'], int(self.mask.sum()))
        self.assertGreaterEqual(records[0]['border_pixels'], 96)
        self.assertEqual(after[0]['diagonal_join']['normal_depth_source'], 'b')

    def test_keep_material_step_axis_edge_different_parents_and_different_planes(self):
        first = part('a',self.mask & self.left)
        other = part('b',self.mask & ~self.left)
        step = np.where(self.left[:,:,None], 92, 172) * np.ones((1,1,3))
        self.assertFalse(refine(Image.fromarray(step.astype('uint8')), [first, other])[1])
        self.assertFalse(refine(self.image, [first,dict(other,parent_id='another')])[1])
        self.assertFalse(refine(self.image, [first,dict(other,normal_camera=[1.,0.,0.])])[1])
        axis = np.indices(self.mask.shape)[1]<130
        self.assertFalse(refine(self.image, [part('a',self.mask & axis),part('b',self.mask & ~axis)])[1])

    def test_preserve_only_chart_junctions_whose_corners_survive(self):
        a = dict(id='a',quad=[[0,0],[3,0],[3,3],[0,3]],corner_junctions=[
            dict(corner=0,neighbor='c',neighbor_corner=0,distance_px=0.),
            dict(corner=1,neighbor='d',neighbor_corner=0,distance_px=0.),
            dict(corner=2,neighbor='b',neighbor_corner=0,distance_px=0.)])
        b = dict(id='b',quad=[[3,3],[7,0],[7,7],[0,7]])
        c = dict(id='c',quad=[[0,0]],corner_junctions=[dict(corner=0,neighbor='a',neighbor_corner=0,distance_px=0.)])
        d = dict(id='d',quad=[[3,0]],corner_junctions=[dict(corner=0,neighbor='a',neighbor_corner=1,distance_px=0.)])
        merged = dict(id='joined',quad=[[0,0],[7,0],[7,7],[0,7]])
        output = _remap_junctions([a,b,c,d], [c,d,merged], {'a':'joined','b':'joined'})
        self.assertEqual([j['neighbor'] for j in output[2]['corner_junctions']], ['c'])
        self.assertEqual(output[0]['corner_junctions'][0]['neighbor'], 'joined')
        self.assertEqual(output[1]['corner_junctions'],[])
