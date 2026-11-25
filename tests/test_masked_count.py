import unittest

import numpy as np
from PIL import Image

from engine.geometry import UNIT
from engine.masked_count import count
from engine.surface_sample import sample_surface


class MaskedCountTest(unittest.TestCase):
    def test_repeating_tile_with_two_mortar_courses_and_a_doorway(self):
        tile=np.random.default_rng(15).integers(55,210,(16,16,3),dtype=np.uint8)
        tile[:,[0,15]]=50
        tile[[0,7,8,15]]=50
        pixels=np.tile(tile,(5,7,1)).repeat(2,0).repeat(2,1)
        h,w=pixels.shape[:2]
        quad=[8.,5.]+UNIT*[w-22.,h-15.]
        mask=dict(size=[w,h],runs=[])
        for y in range(h):
            mask['runs'].extend([[y,0,45],[y,160,w]] if y>=68 else [[y,0,w]])
        source=Image.fromarray(pixels)
        sample=sample_surface(source,{'quad':quad,'support':mask})
        original=count(sample)
        np.testing.assert_allclose(original['counts'],(quad[2]-quad[0])/32,atol=.05)
        # An equally dark stripe halfway down each block makes V phase ambiguous.
        self.assertIsNone(original['phase'][1])
        pixels[68:,45:160]=np.random.default_rng(9).integers(0,256,(h-68,115,3),dtype=np.uint8)
        other=count(sample_surface(Image.fromarray(pixels),{'quad':quad,'support':mask}))
        self.assertEqual(original,other)

    def test_no_counts_from_blank_crop_or_missing_chart(self):
        blank=Image.new('RGB',(96,91),(171,171,171))
        for crop in (sample_surface(blank,{'quad':UNIT*blank.size}),None):
            self.assertEqual(count(crop)['counts'],[None,None])
            self.assertEqual(count(crop)['phase'],[None,None])

    def test_single_direction_does_not_force_second_when_aspect_is_unknown(self):
        rng=np.random.default_rng(22)
        tile=rng.integers(0,256,(43,16,3),dtype=np.uint8)
        im=Image.fromarray(np.tile(tile,(1,6,1)).repeat(2,1))
        s=sample_surface(im,{'quad':UNIT*im.size})
        # The chart mapping alone cannot recover a physical aspect on this
        # generic uncalibrated non-axis-aligned crop.
        s['quad']=[[5,3],[196,3],[178,49],[10,49]]
        out=count(s)
        self.assertAlmostEqual(out['counts'][0],6,delta=.05)
        self.assertIsNone(out['counts'][1])
