import unittest

import numpy as np
from PIL import Image

from engine.mask_line_refine import refine


class RefineTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB',(96,80))
        self.normals = np.zeros((40,48,3),np.float32)
        self.normals[:,:24,0] = 1
        self.normals[:,24:,2] = 1
        self.depth = np.ones((40,48),np.float32)
        self.parts = [dict(id=str(i),support=dict(size=[48,40],runs=[[y,start,end] for y in range(40) if y!=25]))
                      for i,(start,end) in enumerate(((0,22),(22,48)))]
        self.line = dict(segments=[[48,2,48,78]],junctions=[[48,2],[48,78]])

    def test_recovers_native_boundary_preserves_gap_and_source(self):
        got = refine(self.image,self.parts,self.normals,self.depth,self.line)
        self.assertIn([15,0,24],got[0]['support']['runs'])
        self.assertFalse(any(y==25 for p in got for y,a,b in p['support']['runs']))
        self.assertIn([15,0,22],self.parts[0]['support']['runs'])

    def test_texture_without_geometry_and_missing_lines_leave_mask_alone(self):
        self.normals[:] = [0,0,1]
        self.assertEqual(self.parts,refine(self.image,self.parts,self.normals,self.depth,self.line))
        self.assertEqual(self.parts,refine(self.image,self.parts,self.normals,self.depth,dict(segments=[])))
