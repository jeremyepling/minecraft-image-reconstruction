import unittest

import numpy as np
from PIL import Image

from engine.seam_runs import grounded_side_owners
from engine.visible_placement import _frontmost_cells, _sample_face


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
            with np.errstate(divide='ignore',invalid='ignore'):
                ground = cls.eye - rays*(cls.eye[1]/rays[:,:,1,None])
            # A visible checker on the floor is repeated at unit spacing.
            pattern = (np.floor(4*ground[:,:,0])+np.floor(4*ground[:,:,2])) % 2
            rgb=np.repeat((145+35*pattern)[:,:,None],3,axis=2).astype(np.uint8)
            for coord in cells:
                near=(np.array(coord)-cls.eye)/rays
                far=(np.array(coord)+1-cls.eye)/rays
                front=np.minimum(near,far).max(axis=2)
                hit=(front>0) & (front<np.maximum(near,far).min(axis=2)) & (front<depth)
                depth[hit]=front[hit]
                rgb[hit]=[92,70,51] if coord[0] else [49,80,70]
            cls.images[key]=Image.fromarray(rgb)
            cls.masks[key]=[np.isfinite(depth[2::4,2::4])]

    def removals(self,key,cells,*,photo=None,support=None):
        seeds = {(1,2,0):{(0,2,0)}}
        owners = grounded_side_owners(self.images[key] if photo is None else photo, cells, set(),
                                      self.masks[key] if support is None else support, self.projection,
                                      self.eye, _sample_face, _frontmost_cells, seeds)
        return set(owners)

    def test_extra_column_replaced_by_exposed_neighbor_side(self):
        self.assertEqual(self.extension,self.removals('shared',self.wall | self.extension))

    def test_true_second_column_with_distinct_front_is_retained(self):
        self.assertEqual(set(),self.removals('distinct',self.wall | self.extension))

    def test_elevated_strip_stays_even_when_floor_below_is_visible(self):
        raised = {p for p in self.extension if p[1]>=2}
        self.assertEqual(set(),self.removals('shared',self.wall | raised))

    def test_flat_matching_floor_and_wall_colors_do_not_establish_a_seam(self):
        same_color=Image.new('RGB',self.images['shared'].size,(85,85,85))
        self.assertEqual(set(),self.removals('shared',self.wall|self.extension,photo=same_color))

    def test_alternative_must_preserve_observed_coverage(self):
        self.assertEqual(set(),self.removals('shared',self.wall|self.extension,support=self.masks['distinct']))
