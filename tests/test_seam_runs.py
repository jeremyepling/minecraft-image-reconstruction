import unittest

import numpy as np
from PIL import Image

from engine.seam_runs import remove_duplicate_runs
from engine.visible_placement import _sample_face


class VerticalSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eye = np.array([11.,4.2,12.])
        forward = np.array([1.,1.5,.4]) - cls.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward,[0.,1.,0.])
        right /= np.linalg.norm(right)
        rotation = np.stack([right,np.cross(forward,right),forward])
        width,height = 480,360
        intrinsic = np.array([[620.,0.,width/2],[0.,620.,height/2],[0.,0.,1.]])
        cls.projection = intrinsic @ np.column_stack([rotation,-rotation@cls.eye])
        cls.wall = {(0,y,0) for y in range(5)}
        cls.extension = {(1,y,0) for y in range(5)}
        ys,xs=np.indices((height,width),dtype=float)
        rays=np.stack([xs+.5,ys+.5,np.ones_like(xs)],axis=2) @ np.linalg.inv(cls.projection[:,:3]).T
        cls.images={}
        cls.masks={}
        for key,cells in [('shared',cls.wall),('distinct',cls.wall | cls.extension)]:
            depth = np.full((height,width),np.inf)
            rgb=np.full((height,width,3),164,dtype=np.uint8)
            for coord in cells:
                near=(np.array(coord)-cls.eye)/rays
                far=(np.array(coord)+1-cls.eye)/rays
                front=np.minimum(near,far).max(axis=2)
                hit=(front>0) & (front<np.maximum(near,far).min(axis=2)) & (front<depth)
                depth[hit]=front[hit]
                rgb[hit]=[92,70,51] if coord[0] else [49,80,70]
            cls.images[key]=Image.fromarray(rgb)
            cls.masks[key]=[np.isfinite(depth[2::4,2::4])]

    def removals(self,key,cells):
        return remove_duplicate_runs(self.images[key],cells,set(),self.masks[key],self.projection,
                                     self.eye,_sample_face)

    def test_extra_column_replaced_by_exposed_neighbor_side(self):
        self.assertEqual(self.extension,self.removals('shared',self.wall | self.extension))

    def test_true_second_column_with_distinct_front_is_retained(self):
        self.assertEqual(set(),self.removals('distinct',self.wall | self.extension))

    def test_elevated_strip_stays_even_when_floor_below_is_visible(self):
        raised = {p for p in self.extension if p[1]>=2}
        self.assertEqual(set(),self.removals('shared',self.wall | raised))
