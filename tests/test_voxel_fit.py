import unittest

import numpy as np

from benchmark.voxel_eval import camera, registration
from calibration.world import GROUND
from engine.voxel_models import predict
from engine.voxel_space import render, silhouette_volume


class VoxelFitTest(unittest.TestCase):
    def test_registration_preserves_reflected_cell_coordinates(self):
        pose=dict(eye_world=[50.,78.,29.],yaw_pitch_deg=[-162.,15.],vertical_fov_deg=60)
        transform=np.diag([1.,1.,-1.,1.])
        transform[:3,3]=[48,GROUND+1,10]
        P=camera(pose,(320,200))@transform
        np.testing.assert_array_equal(registration(P,pose,(320,200)),transform)

    def test_hull_requires_front_of_camera(self):
        # All cells behind the camera must be rejected even if pixels lie on the mask.
        P=np.array([[8.,0,8,0],[0,8,8,0],[0,0,1,-20]])
        with self.assertRaises(ValueError):
            silhouette_volume(P,(3,3,3),np.ones((16,16),bool))

    def test_symmetric_model_retains_centered_opening(self):
        hull=np.ones((5,4,4),bool)
        hull[2,0:3,:]=False
        self.assertTrue(np.array_equal(hull,predict(dict(hull=hull),'symmetric')))
        P=camera(dict(eye_world=[2.5,2.,14.],yaw_pitch_deg=[180.,0.],vertical_fov_deg=40),(160,160))
        mask=np.asarray(render(hull,P,(160,160)))
        self.assertEqual(mask[80,80],0)
        self.assertTrue(mask.any())
