"""A matching atlas fragment can provide evidence with an incorrectly sized cell."""
import unittest

import numpy as np

from engine.motifs import LUMA, normalize, priors, proposals


class MotifTest(unittest.TestCase):
    def test_fragment_at_unknown_phase(self):
        rng = np.random.default_rng(813)
        tiles = rng.integers(55,145,(3,16,16,1)).repeat(3,axis=-1).astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(tiles@LUMA,(7,7),axis=(1,2))
        lib = (np.array(['first','second','third']),tiles.mean((1,2)),normalize(windows.reshape(3,100,49)))
        image = np.tile(tiles[1],(3,3,1)).repeat(2,axis=0).repeat(2,axis=1)
        cells = [dict(quad=[[45,45],[51,45],[51,51],[45,51]])]
        label,strength,margin = proposals(image,cells,lib)[0]
        self.assertEqual(label,'second')
        self.assertGreater(strength,.86)
        self.assertGreater(margin,.05)

    def test_missing_native_sprites(self):
        self.assertEqual(priors(None,[],None),{})
