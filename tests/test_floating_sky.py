import unittest

import numpy as np
from PIL import Image

from engine.floating_sky import filter_proposals


class FloatingSkyTests(unittest.TestCase):
    def test_whole_cloud_rejected_mixed_and_enclosed_pixels_preserved(self):
        pixels = np.full((12, 16, 3), (110, 160, 250), dtype=np.uint8)
        pixels[2, 3:7] = (225, 235, 251)
        pixels[5:9, 6:10] = (120, 90, 80)
        pixels[6:8, 7:9] = (110, 160, 250)
        masks = [
            dict(id=name, support=dict(size=[16, 12], runs=[run]))
            for name, run in [('cloud', [2, 3, 7]), ('mixed', [5, 4, 7]),
                              ('enclosed', [6, 7, 9])]
        ]
        kept, receipt = filter_proposals(Image.fromarray(pixels), masks)
        self.assertEqual(receipt['rejected_ids'], ['cloud'])
        self.assertEqual(kept, masks[1:])

    def test_disagreeing_borders_keep_the_surface(self):
        image = Image.new('RGB', (16, 4), (110, 160, 250))
        image.putpixel((15, 2), (40, 50, 60))
        surface = dict(id='wall', support=dict(size=[16, 4], runs=[[2, 3, 7]]))
        kept, receipt = filter_proposals(image, [surface])
        self.assertEqual(kept, [surface])
        self.assertEqual(receipt['rejected_ids'], [])
