"""Exercise the actual candidate enumerator on a visible overhead sheet with a gap."""
import unittest
import numpy as np
from PIL import Image

from engine import floating_grid_policy as floating
from engine.surface_model import intrinsic
from engine.visible_placement import _estimate_plane_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE


class FloatingGridTest(unittest.TestCase):
    def test_signed_observed_underside_and_gap(self):
        size = (96,96)
        image = Image.new('RGB',size)
        # Look straight up. The photograph contains two disjoint underside faces.
        rotation = np.array([[1,0,0],[0,0,1],[0,1,0]])
        matrix = intrinsic(size,90) @ rotation
        expected = {(-1,-1,0),(1,-1,0)}
        shape = tuple(s//SAMPLE_STRIDE for s in size)
        for shift in (0,7,-9):
            eye = np.array([.5,-5+shift,.5])
            projection = np.column_stack([matrix,-matrix@eye])
            mask = np.zeros(shape, bool)
            moved = {(x,y+shift,z) for x,y,z in expected}
            for cell in moved:
                box,pixels,*_ = _sample_face(projection,cell,1,-1,shape)
                mask[box] |= pixels
            part = dict(normal_camera=rotation@np.array([0,-1,0]),relative_inverse_depth=1)
            token = floating.scene_distance.set(4)
            try:
                candidates, = _estimate_plane_cells(image,[part],[mask],projection,eye)
                self.assertEqual(candidates,moved)
            finally:
                floating.scene_distance.reset(token)
            if shift==0:
                ordinary, = _estimate_plane_cells(image,[part],[mask],projection,eye,
                        lambda *args:-1)
                self.assertFalse(ordinary)


if __name__=='__main__':
    unittest.main()
