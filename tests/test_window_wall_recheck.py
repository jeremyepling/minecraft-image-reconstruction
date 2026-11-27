import unittest

import numpy as np
from PIL import Image

from engine.plane_consensus import row_runs
from engine.surface_model import intrinsic
from engine.visible_placement import _sample_face
from engine.visible_raster import SAMPLE_STRIDE
from engine.window_wall_recheck import recover


class WindowWallRecheckTests(unittest.TestCase):
    def setUp(self):
        # An eye at y=4 sees a short masonry wall behind a temporary elevated shelf.
        size = (336, 240)
        self.image = Image.new('RGB', size)
        eye = np.array([.5, 4., -8.])
        rotation = np.diag([1., -1., 1.])
        projection = intrinsic(size, 200) @ np.column_stack((rotation, -rotation @ eye))
        self.status = dict(P=projection, plane_offsets=[
            dict(id='wall', axis=2, level=4), dict(id='pavement', axis=1, level=3)])
        self.target = (0, 1, 4)
        self.placed = {(x, 0, 4) for x in range(-1, 2)} | {(-1, 1, 4), (1, 1, 4)}
        self.parts = []
        shape = (size[1]//SAMPLE_STRIDE, size[0]//SAMPLE_STRIDE)
        for name, axis, sign, points in [
            ('wall',2,-1,self.placed | {self.target}),
            ('pavement',1,1,{(x,2,z) for x in range(-1,2) for z in range(-4,2)})]:
            mask = np.zeros(shape, bool)
            for cell in points:
                hit = _sample_face(projection, cell, axis, sign, shape)
                if hit is not None:
                    box, pixels, *_ = hit
                    mask[box] |= pixels
            n = np.zeros(3)
            n[axis] = sign
            self.parts.append(dict(id=name, normal_camera=rotation @ n, support=dict(
                size=shape[::-1], runs=row_runs(mask))))

    def test_rechecks_only_wall_supported_on_final_projection(self):
        def check(placed, openings):
            return recover(self.image, self.parts, self.status, openings, placed)
        self.assertEqual({self.target}, check(self.placed, {'pavement'}))
        self.assertEqual(set(), check(self.placed, set()))
        self.assertEqual(set(), check(self.placed - {(0,0,4)}, {'pavement'}))
        self.assertEqual(set(), check(self.placed | {(0,1,3)}, {'pavement'}))
