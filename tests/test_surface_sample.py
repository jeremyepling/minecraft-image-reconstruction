import unittest

import numpy as np
from PIL import Image

from engine.geometry import UNIT, project, transform
from engine.surface_sample import sample_surface


class SurfaceSampleTest(unittest.TestCase):
    def test_inverse_projective_warp_straightens_tilted_gradient(self):
        quad = [[50,25],[195,52],[168,160],[15,128]]
        y,x=np.indices((185,215))
        uv=project(np.linalg.inv(transform(quad)),np.stack([x+.5,y+.5],axis=-1))
        field=np.dstack((uv,uv[:,:,0]*0))
        image=Image.fromarray(np.rint(np.clip(field,0,1)*255).astype(np.uint8))
        s=sample_surface(image,{'quad':quad})
        h,w=s['mask'].shape
        y,x=np.indices((h,w))
        ideal=np.stack(((x+.5)/w,(y+.5)/h),axis=-1)*255
        np.testing.assert_allclose(s['rgb'][s['mask'],:2],ideal[s['mask']],atol=1.1)

    def test_mask_keeps_holes_and_excludes_interpolation_across_them(self):
        base=np.full((80,100,3),94,np.uint8)
        # Half-resolution RLE with a central hole.
        runs=[]
        for y in range(4,36):
            runs.extend([[y,3,14],[y,18,47]] if 10<=y<31 else [[y,3,47]])
        support=np.zeros((40,50),bool)
        for y,a,b in runs:
            support[y,a:b]=True
        changed=base.copy()
        changed[~support.repeat(2,0).repeat(2,1)]=[255,0,255]
        region=dict(quad=[[3.6,2.1],[94,5],[98,76],[5.2,72]],
                    support=dict(size=[50,40],runs=runs))
        a,b=[sample_surface(Image.fromarray(rgb),region) for rgb in (base,changed)]
        np.testing.assert_array_equal(a['rgb'],b['rgb'])
        self.assertTrue((a['rgb'][a['mask']]==94).all())
        self.assertFalse(b['mask'][b['mask'].shape[0]//2,b['mask'].shape[1]//3])

    def test_chartless_wall_uses_its_camera_normal_but_roof_yaw_is_unknown(self):
        image=Image.new('RGB',(120,80),'gray')
        mask={'size':image.size,'runs':[[y,10,90] for y in range(9,59)]}
        s=sample_surface(image,dict(normal_camera=[0,0,-1],support=mask))
        np.testing.assert_allclose(s['quad'],[10,9]+UNIT*[80,50],atol=.001)
        self.assertIn('wall normal',s['chart_source'])
        for normal in (None,[0,1,0],[1,0,.01]):
            self.assertIsNone(sample_surface(image,dict(normal_camera=normal,support=mask)))
