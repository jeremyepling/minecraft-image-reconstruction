"""Regression controls for texel scale, cavity truth, and the unknown-depth audit."""
import unittest
import numpy as np
from benchmark.fixtures import Building,canopy
from benchmark.geometry_metrics import covers_point
from engine.texel_grid import texel_grid


class GeometryExperimentTests(unittest.TestCase):
    def test_constant_material_pitch_from_texel_edges(self):
        sprite=np.random.default_rng(31).integers(0,256,(16,16,3),dtype=np.uint8)
        rgb=np.tile(sprite,(3,5,1)).repeat(4,axis=0).repeat(4,axis=1)
        q=np.array([[0.,0],[320,0],[320,192],[0,192]])
        self.assertEqual(texel_grid(rgb,q),[5,3])

    def test_canopy_gap_and_projection_are_independent(self):
        for gap in (2,4,6):
            for over in (1,2,4):
                b=Building(0,'train','canopy',{})
                canopy(b,gap,over,0)
                for x in range(1,gap+1):
                    self.assertNotIn((x,2,0),b.cells)
                self.assertIn((0,2,0),b.cells)
                self.assertIn((gap+1,2,0),b.cells)
                self.assertIn((0,5,over),b.cells)
                self.assertNotIn((0,5,over+1),b.cells)

    def test_gap_probe_near_boundary_uses_subpixel_polygon(self):
        p=np.array([[[0.,0],[2,0],[2,2],[0,2]]])
        self.assertTrue(covers_point(p,[1.99,1]))
        self.assertFalse(covers_point(p,[2.01,1]))


if __name__=='__main__': unittest.main()
