"""A corner ray should end on a visible back rim before depth can be measured."""
import unittest

import numpy as np
from PIL import Image, ImageDraw

from engine.corner_edges import _segment, predict


class EdgeLocatorTests(unittest.TestCase):
    def test_flat_patch_has_no_back_corner(self):
        q=np.array([[40,60],[160,60],[160,160],[40,160]],float)
        rgb=np.full((210,240,3),140,dtype=np.uint8)
        rgb[60:160,40:160]=65
        result=predict(rgb,q,np.array([0.,12.]),16.,36.,[1.,1.])
        self.assertIsNone(result['depth'])

    def test_clipped_edge_excludes_wall_outside_rim(self):
        field=np.zeros((120,120,1),float)
        field[60:,65:]=200
        measured=_segment(field,np.array([5.,60.]),np.array([115.,60.]),
                          np.array([10.,10.,55.,90.]),0.)
        self.assertEqual(measured,0.)

    def test_angular_return_and_terminating_back_rim(self):
        q=np.array([[55,60],[165,60],[165,168],[55,168]],float)
        vp=np.array([3.,5.])
        back=vp+(q-vp)/(1+2/12)
        tile=Image.new('L',(240,210),140)
        draw=ImageDraw.Draw(tile)
        draw.rectangle((55,60,165,168),fill=55)
        draw.polygon(list(map(tuple,[q[1],q[2],back[2],back[1]])),fill=20)
        draw.polygon(list(map(tuple,[q[2],q[3],back[3],back[2]])),fill=110)
        # Clip the reveal panels to the aperture.
        canvas=Image.new('L',tile.size,140)
        canvas.paste(tile.crop((55,60,166,169)),(55,60))
        result=predict(np.asarray(canvas.convert('RGB')),q,vp,12.,36.,[1.,1.])
        self.assertEqual(result['depth'],2)
