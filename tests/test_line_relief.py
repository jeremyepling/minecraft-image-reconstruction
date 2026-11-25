import unittest
import numpy as np
from PIL import Image

from engine.line_relief import _faces, _stripe, _junctions, _flat_repeat, analyze


class LineReliefTest(unittest.TestCase):
    def test_stripe_rejects_smooth_gradient_but_keeps_both_edge_polarities(self):
        yy,xx=np.indices((100,140))
        a=np.array([30.,45.]);b=np.array([100.,45.])
        ramp=(yy/140.)[:,:,None]
        step=(yy>=45).astype(float)[:,:,None]
        self.assertLess(float(_stripe(ramp,a,b,2)),.001)
        for sign in (-1,1):
            self.assertGreater(float(_stripe(ramp+sign*step,a,b,2)),.95)

    def test_equal_luminance_geometry_does_not_require_dark_interior(self):
        quad=np.array([[32,30],[98,40],[100,95],[36,85]],float)
        lines=[dict(p=a.tolist(),q=b.tolist(),strength=10.)
               for a,b in zip(quad,np.roll(quad,-1,axis=0))]
        field=np.zeros((125,140,1))
        found=_faces(lines,np.ones((125,140),bool),field)
        self.assertTrue(any(np.max(abs(q-quad))<.1 for _,q in found))
        self.assertFalse(_flat_repeat(field,quad))
        self.assertTrue(all(t['weight']==0 for t in _junctions(field,quad,[])))

    def test_empty_outline_stops_before_local_line_detection(self):
        im=Image.new('RGB',(128,96),'#c0c0c0')
        self.assertEqual(analyze(im,[],Image.new('L',im.size)),[])
