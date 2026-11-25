import unittest

import numpy as np
from PIL import Image

from benchmark.camera_metrics import truth_camera, registration
from calibration.world import GROUND
from engine.camera_lines import predict
from engine.geometry import project


class TileCameraTest(unittest.TestCase):
    def test_unmarked_or_unidirectional_floor_is_uncertain(self):
        for pixels in (np.full((400, 640), 127), (np.indices((400, 640))[1] % 30 < 2)*150):
            proposal = predict(Image.fromarray(pixels.astype('uint8')).convert('RGB'), '26.3')
            self.assertIsNone(proposal['P'])
            self.assertEqual(proposal['scale_status'], 'uncertain')

    def test_indexed_tiles_constrain_elevated_intersections(self):
        size = (800, 500)
        pose = dict(eye_world=[.12, GROUND+1+4.5, -.72], yaw_pitch_deg=[155, 10],
                    image_size=size, vertical_fov_deg=64)
        W = truth_camera(pose)
        y, x = np.indices(size[::-1])+.5
        rays = np.dstack([x, y, np.ones_like(x)]) @ np.linalg.inv(W[:, :3]).T
        floor = pose['eye_world']-4.5*rays/rays[:, :, 1, None]
        uv = floor[:, :, [0, 2]] % 1
        inside = np.all((uv > .055) & (uv < .945), axis=2)
        gray = np.where(rays[:, :, 1] < 0, np.where(inside, 170, 122), 200)
        proposal = predict(Image.fromarray(gray.astype('uint8')).convert('RGB'), '26.3')
        self.assertEqual(proposal['scale_status'], 'supported')
        P = np.array(proposal['P'])
        true = W @ registration(P, pose)
        lattice = [[x, y, z] for x in (-2, 0, 2) for y in (0, 2, 4) for z in (-2, 0, 2)]
        distance = np.linalg.norm(project(P, lattice)-project(true, lattice), axis=1)
        self.assertLess(np.median(distance), 2.)
