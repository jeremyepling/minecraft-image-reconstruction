import unittest

import numpy as np

from benchmark.camera_grid_joint import intrinsic, rotations, patches, solve
from benchmark.camera_grid_joint_replay import matches, raw_grid_levels


class JointFit(unittest.TestCase):
    def test_rotations_preserve_right_handed_camera_axes(self):
        R=np.diag([1.,-1.,1.])
        K=intrinsic([750]*2,[1440,900])
        geo=dict(up_cam=[.03,-.95,.1],focal_xy_px=[800,800])
        for name,L,result in rotations(K@R,750,geo,[1440,900]):
            np.testing.assert_allclose(result.T@result,np.eye(3),atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(result),-1.)
            if name=='geocalib':
                np.testing.assert_allclose(L,intrinsic([800]*2,[1440,900])@result)

    def test_overlap_excluded_across_surface_ids(self):
        a=dict(axis=2,corr=.8,quad=[[10,10],[20,10],[20,20],[10,20]],surface='a')
        self.assertEqual(len(patches([a,dict(a,surface='b')])),1)

    def test_mask_must_match_after_resegmentation(self):
        a=dict(support=dict(size=[4,4],runs=[[0,0,4]]))
        b=dict(support=dict(size=[4,4],runs=[[3,0,4]]))
        self.assertTrue(matches(a,a))
        self.assertFalse(matches(a,b))

    def test_raw_integer_gauge_preserves_native_projection(self):
        L=intrinsic([700]*2,[1440,900]) @ np.diag([1.,-1.,1.])
        raw_eye=np.array([-49.73,28.59,-76.61])
        joint_eye=np.array([.29,28.52,.20])
        camera=lambda e: dict(P=np.column_stack((L,-L@e)))
        levels,shift=raw_grid_levels(camera(raw_eye),camera(joint_eye),dict(x=[0,52],y=[1,4],z=[2,95]))
        self.assertEqual(shift,[-50,0,-77])
        self.assertEqual(levels,dict(x=[0,2],y=[1,4],z=[2,18]))
        joint_grid_points=np.array([[52,4,95],[53,5,95]])
        raw_grid_points=joint_grid_points+shift
        def project(P,points):
            homogeneous=np.column_stack((points,np.ones(len(points)))) @ P['P'].T
            return homogeneous[:,:2]/homogeneous[:,2,None]
        np.testing.assert_allclose(project(camera(raw_eye),raw_grid_points),
                                   project(camera(raw_eye-shift),joint_grid_points))
        self.assertFalse(np.allclose(raw_eye-shift,joint_eye))

    def test_exact_disjoint_wall_and_floor(self):
        L=intrinsic([650]*2,[1440,900])@np.diag([1.,-1.,1.])
        eye=np.array([0.,2.,0.])
        anchors=[]
        for key,depth,count in [('front',8,3),('rear',10,2)]:
            for j in range(count):
                xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])+[j*2,1,depth]-eye
                pix=xyz@L.T
                anchors.append(dict(surface=key,axis=2,quad=(pix[:,:2]/pix[:,2,None]).tolist(),xyz=xyz.tolist()))
        contacts=[[720+i,612.5] for i in range(5)]
        walls=dict(front=(2,4,contacts),rear=(2,5,[]))
        candidates=solve(L,1.,walls,anchors,dict(front=[2.]))
        exact=min(candidates,key=lambda c:c['rank'])
        self.assertTrue(exact['accepted'])
        self.assertEqual(exact['overrides'],dict(front=[2,8],rear=[2,10]))
        self.assertLess(exact['held_p90_px'],1e-8)
