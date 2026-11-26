import unittest

import numpy as np
from PIL import Image

from engine.visible_horizontal_faces import _keep_prior_visible, recover
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def camera(size, eye, target):
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross([0,1,0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    rotation = np.array([right, down, forward])
    intrinsics = np.array([[75, 0, size[0]/2], [0, 75, size[1]/2], [0, 0, 1]])
    basis = intrinsics @ rotation
    return np.column_stack((basis, -basis @ eye)), rotation


def mask_runs(mask):
    rows = []
    for y, row in enumerate(mask):
        boundaries = np.flatnonzero(np.diff(np.pad(row, (1,1)).astype(int)))
        rows.extend([y, int(left), int(right)] for left,right in boundaries.reshape(-1,2))
    return dict(size=list(mask.shape[::-1]), runs=rows)


class VisibleHorizontalFacesTest(unittest.TestCase):
    def test_recovery_requires_horizontal_image_evidence(self):
        size = (95,73)
        image = Image.new('RGB',size)
        eye = np.array([2.5,7.,-5.])
        projection, rotation = camera(size,eye,[1.5,1.,.5])
        expected = {(0,0,0), (1,0,0), (2,0,0)}
        prior = expected-{(1,0,0)}
        native = projection.copy()
        native[:2] *= SAMPLE_STRIDE
        _, axes = _frontmost_cells(sorted(expected),native,eye,size[::-1],face_axes=True)
        parts = [dict(support=mask_runs(axes==axis), normal_camera=rotation[:,axis]) for axis in range(3)]
        self.assertEqual(recover(image,parts,{'P':projection},prior),expected)
        self.assertEqual(recover(image,[parts[0],parts[2]],{'P':projection},prior),prior)
        self.assertEqual(recover(image,parts,{'P':None},prior),prior)

    def test_multiple_front_blockers_cannot_cover_the_last_prior_pixel(self):
        size = (109,75)
        eye = np.array([.5,.5,-5.])
        projection,_ = camera(size,eye,[.5,.5,3.])
        prior = {(0,0,3)}
        additions = {(0,0,1),(0,0,0),(2,0,0)}
        self.assertEqual(_keep_prior_visible(size,projection,eye,prior,additions),prior|{(2,0,0)})
