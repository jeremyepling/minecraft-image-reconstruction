import unittest

import numpy as np

from benchmark.camera_moge_fields import intrinsics
from benchmark.camera_moge_surface import geometry


class MogeSurfaceTest(unittest.TestCase):
    def test_focal_uses_native_pixels_and_recovers_z_shift(self):
        h,w=50,80
        y,x=np.mgrid[:h,:w]
        size=np.array([1600,1000])
        uv=(np.stack(((x+.5)/w,(y+.5)/h),axis=2)-.5)*size
        depth=2+np.sin(x/14)+.25*np.cos(y/7)
        points=np.dstack((uv*depth[:,:,None]/850,depth-.72))
        mask=x>5
        np.testing.assert_allclose(intrinsics(points,mask,size),[850,.72],atol=1e-8)

    def test_no_manhattan_fit_from_floor_alone(self):
        h,w=60,90
        y,x=np.mgrid[:h,:w]
        pts=np.stack([x/25,np.ones_like(x),(h-y)/20+1],axis=2)
        n=np.zeros_like(pts)
        n[:,:,1]=-1
        field=dict(mask=np.ones((h,w),bool),points=pts,normal=n,z_shift=0,
                   focal_px=100,native_size=np.array([w,h]))
        self.assertIsNone(geometry(field,[]))
