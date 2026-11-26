import unittest

import numpy as np

from benchmark.template_block_seam import register
from engine.atlas_count import _prepare


class TemplateBlockSeamTests(unittest.TestCase):
    def setUp(self):
        gen = np.random.default_rng(142)
        self.tile = gen.integers(40, 220, (16,16,3), dtype=np.uint8)
        self.atlas = _prepare(self.tile[None], ['synthetic'], [['minecraft:synthetic']], {})
        self.rgb = np.roll(np.tile(self.tile,(3,5,1)),(4,7),(0,1))
        self.mask = np.ones(self.rgb.shape[:2],bool)
        self.mask[12:28,12:35] = False
        self.rgb[~self.mask] = [255,0,255]

    def run_registration(self, counts, phase):
        baseline = dict(counts=counts, phase=phase, evidence=dict(aspect=5/3, alternatives=[[],[]]))
        return register(dict(rgb=self.rgb, mask=self.mask), baseline, self.atlas)

    def test_new_pitches_and_offsets_from_masked_rgb(self):
        result = self.run_registration([None,None],[None,None])
        np.testing.assert_allclose(result['counts'],[5,3],atol=.1)
        np.testing.assert_allclose(result['phase'],[7/16,4/16],atol=.07)

    def test_retains_located_axis(self):
        result = self.run_registration([5,None],[7/16,None])
        np.testing.assert_allclose(result['counts'],[5,3],atol=.1)
        np.testing.assert_allclose(result['phase'],[7/16,4/16],atol=.07)

    def test_uniform_or_unsupported_crops_abstain(self):
        self.rgb[:] = 100
        self.assertEqual(self.run_registration([None,None],[None,None])['counts'],[None,None])
        self.mask[:] = False
        self.assertEqual(self.run_registration([5,None],[None,None])['phase'],[None,None])
