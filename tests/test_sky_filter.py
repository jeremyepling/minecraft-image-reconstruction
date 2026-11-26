import unittest

import numpy as np
from PIL import Image

from engine.plane_consensus import run_mask
from engine.sky_filter import filter_surface, sky_masks


class SkyFilterTest(unittest.TestCase):
    def test_vanilla_clouds_require_exterior_gradient(self):
        a = np.full((80, 100, 3), 112, dtype=np.uint8)
        for y in range(40):
            red = 138+y
            a[y, :] = (red, round(.69*red+84), 255)
        a[10:16, 15:35] = (159, 177, 211)
        a[12:38, 55:60] = 240  # Pale pillar within the sky silhouette.
        a[60:65, 15:35] = (159, 177, 211)  # Same color on an enclosed wall.
        blue, cloud = sky_masks(Image.fromarray(a))
        self.assertTrue(blue[5, 5])
        self.assertTrue(cloud[12, 18])
        self.assertFalse((blue | cloud)[12:38, 55:60].any())
        self.assertFalse((blue | cloud)[60:65, 15:35].any())
        self.assertFalse(any(m.any() for m in sky_masks(Image.new('RGB',(100,80),(138,179,255)))))

    def test_native_pixel_subtraction_and_empty_cloud_proposal(self):
        image = Image.new('RGB',(8,8))
        blue = np.zeros((8,8),bool)
        cloud = blue.copy()
        cloud[:4,:4] = True
        blue[5,2] = True
        parent = dict(id='roof',support=dict(size=[4,4],runs=[[y,0,4] for y in range(4)]),
                      quad=[[0,0],[8,0],[8,8],[0,8]])
        other = dict(id='cloud',support=dict(size=[4,4],runs=[[0,0,2],[1,0,2]]))
        kept, log = filter_surface(image,[parent,other],(blue,cloud))
        self.assertEqual(log['trimmed'],['roof'])
        self.assertEqual(log['removed'],['cloud'])
        self.assertNotIn('quad',kept[0])
        np.testing.assert_array_equal(run_mask(kept[0]['support']),~(blue | cloud))
        self.assertEqual(filter_surface(image,kept,(blue,cloud))[0],kept)
