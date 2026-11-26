import unittest

import numpy as np
from PIL import Image, ImageDraw

from engine.surface_edges import detect_planes


def canopy():
    image = Image.new('RGB',(400,270),(156,194,232))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0,137,400,270),fill=(142,145,131))
    # Two supports and a thin side return expose the open sky and ground.
    draw.polygon([(84,85),(110,89),(110,217),(84,208)],fill=(130,95,71))
    draw.polygon([(274,104),(295,107),(295,234),(274,232)],fill=(130,95,71))
    draw.polygon([(295,107),(310,99),(310,220),(295,234)],fill=(84,64,58))
    draw.polygon([(66,56),(311,76),(311,111),(66,86)],fill=(166,100,91))
    draw.polygon([(311,76),(329,65),(329,99),(311,111)],fill=(112,70,65))
    return image


def covers(quad, point):
    q=np.asarray(quad)
    v=np.roll(q,-1,axis=0)-q
    d=np.asarray(point)-q
    return np.all(v[:,0]*d[:,1]-v[:,1]*d[:,0]>=0)


class SurfaceEdgeTests(unittest.TestCase):
    def test_detects_visible_patches_and_preserves_air_between_posts(self):
        patches=detect_planes(canopy())
        self.assertTrue(any(covers(p['quad'],(150,78)) for p in patches))
        self.assertTrue(any(covers(p['quad'],(96,149)) for p in patches))
        for point in ((146,120),(207,148),(250,190)):
            self.assertFalse(any(covers(p['quad'],point) for p in patches))

    def test_shading_scores_the_same_geometry(self):
        plain={p['id']:p for p in detect_planes(canopy())}
        lit={p['id']:p for p in detect_planes(canopy(),shading=True)}
        shared=plain.keys() & lit.keys()
        self.assertTrue(shared)
        for key in shared:
            self.assertEqual(plain[key]['quad'],lit[key]['quad'])
            self.assertFalse(plain[key]['evidence']['shading_used'])
            self.assertIn('median_luminance_step',lit[key]['evidence'])

    def test_featureless_crop_provides_no_four_observed_sides(self):
        self.assertEqual(detect_planes(Image.new('RGB',(240,180),'gray')),[])
