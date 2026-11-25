import unittest

import numpy as np
from PIL import Image

from benchmark.mask_metrics import footprint
from engine import single_wall_gate as gate
from engine.camera_lines import _runs


class SingleWallGateTest(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(160,80))
        self.parent=np.ones((20,40),bool)
        self.parent[7:12,9:11]=False
        runs=[run for group in _runs(self.parent) for run in group]
        self.regions=[dict(id='wall',support=dict(size=(40,20),runs=runs),normal_camera=[0,0,-1])]
        self.shape=np.indices(self.parent.shape)[1]<20
        self.masks=[self.shape]
        self.records=[dict(quality=.98,area=int(self.shape.sum()))]
        self.inv=gate.prepare(self.image,self.regions,self.masks,self.records,[0])

    def process(self,z):
        return gate.apply(self.image,self.regions,self.masks,self.records,self.inv,z)

    def test_affine_depth_gradient_alone_keeps_wall(self):
        y,x=np.indices(self.parent.shape,dtype=float)
        result,events=self.process(2+x/8+y/16)
        self.assertEqual(result,self.regions)
        self.assertEqual(len(events),1)
        self.assertFalse(events[0]['accepted'])
        self.assertAlmostEqual(events[0]['median_step'],0.)

    def test_single_step_retains_hole_and_is_unit_independent(self):
        y,x=np.indices(self.parent.shape,dtype=float)
        z=5+x/8+y/16+self.shape*2
        result,events=self.process(z)
        transformed,other=self.process(z*.125-2)
        self.assertTrue(events[0]['accepted'])
        self.assertGreater(events[0]['median_step'],0.)
        self.assertEqual(result,transformed)
        self.assertAlmostEqual(events[0]['median_step'],other[0]['median_step'])
        masks=[footprint(p,self.parent.shape,self.image.size,True) for p in result]
        np.testing.assert_array_equal(np.sum(masks,axis=0),self.parent)
        self.assertEqual({p['single_wall_side'] for p in result},{'inside','remainder'})

    def test_native_map_sampling_on_odd_images(self):
        original=np.arange(17*21,dtype=np.float32).reshape(17,21)
        sampled=gate.depth_grid(original,(21,17))
        np.testing.assert_array_equal(sampled,original[2:16:4,2:20:4])


if __name__=='__main__':
    unittest.main()
