import unittest

import numpy as np

from engine.cube_axes import fit_axes


def scene(pitch, seed=824):
    rng = np.random.default_rng(seed)
    y,p = np.radians([34.,pitch])
    forward = [-np.sin(y)*np.cos(p),-np.sin(p),np.cos(y)*np.cos(p)]
    right = [-np.cos(y),0.,-np.sin(y)]
    up = np.cross(right,forward)
    basis = np.stack([right,-up,forward]).T
    size = (900,640)
    vp = basis.copy()
    vp[:,:2] = 640*basis[:,:2] + np.asarray(size)*.5*basis[:,2,None]
    groups = []
    for point in vp:
        mids = rng.uniform([110,85],[790,550],size=(14,2))
        direction = point[:2]-mids*point[2]
        delta = direction/np.linalg.norm(direction,axis=1,keepdims=True)*36
        groups.append([dict(p=(m-d).tolist(),q=(m+d).tolist()) for m,d in zip(mids,delta)])
    return size, vp, groups


class CubeAxisTest(unittest.TestCase):
    def check_frame(self, pitch):
        size, expected, groups = scene(pitch)
        fitted = np.array(fit_axes(sum(groups, []),size))
        self.assertEqual(fitted.shape,(3,3))
        # Compare edge directions at two separated image points. A far-away
        # vanishing point can have large coordinates yet subpixel line error.
        errors = np.zeros((3,3))
        for xy in [[240,170],[680,490]]:
            a = fitted[:,:2]-xy*fitted[:,2,None]
            b = expected[:,:2]-xy*expected[:,2,None]
            a /= np.linalg.norm(a,axis=1)[:,None]
            b /= np.linalg.norm(b,axis=1)[:,None]
            errors = np.maximum(errors, np.degrees(np.arccos(abs(a@b.T).clip(0,1))))
        self.assertEqual(len(set(errors.argmin(0))),3)
        self.assertLess(errors.min(0).max(), .15)

    def test_oblique_perspective(self):
        self.check_frame(19)

    def test_level_camera_with_vertical_axis_at_infinity(self):
        self.check_frame(0)

    def test_missing_axis_abstains(self):
        size, _, groups = scene(19)
        self.assertEqual(fit_axes(groups[0]+groups[1],size), [])
        self.assertEqual(fit_axes([],size), [])
