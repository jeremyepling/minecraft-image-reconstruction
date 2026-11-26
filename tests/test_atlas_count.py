import unittest

import numpy as np

from engine.atlas_count import _prepare, count, rank_tile


class AtlasCountTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(385)
        textures=rng.integers(50,210,size=(3,16,16,3),dtype=np.uint8)
        self.tile=textures[1]
        self.atlas=_prepare(textures,['a','b','c'],[['minecraft:a'],['minecraft:b','minecraft:b_alias'],
                                                     ['minecraft:c']],dict(version='26.3'))

    def test_full_texture_repeat_with_shift_and_a_hole(self):
        rgb=np.roll(np.tile(self.tile,(4,7,1)),(5,3),(0,1))
        mask=np.ones(rgb.shape[:2],bool)
        mask[13:41,18:59]=False
        rgb[~mask]=[255,255,255]
        result=count(dict(rgb=rgb,mask=mask),self.atlas)
        self.assertEqual(result['block_id'],'minecraft:b')
        self.assertEqual(result['same_texture_ids'],['minecraft:b','minecraft:b_alias'])
        np.testing.assert_allclose(result['counts'],[7,4],atol=.15)
        np.testing.assert_allclose(result['phase'],[3/16,5/16],atol=.07)

    def test_rank_tile_accepts_shading_rotation_and_irregular_support(self):
        rgb=(np.rot90(self.tile)*.57).astype(np.uint8)
        mask=np.ones((16,16),bool)
        mask[:6,:9]=False
        rgb[~mask]=200
        result=rank_tile(dict(rgb=rgb,mask=mask),self.atlas)
        self.assertEqual(result['block_id'],'minecraft:b')
        self.assertGreater(result['evidence']['correlation'],.99)

    def test_uniform_surface_and_empty_mask_cannot_supply_counts(self):
        rgb=np.full((90,100,3),128,np.uint8)
        mask=np.ones(rgb.shape[:2],bool)
        self.assertEqual(count(dict(rgb=rgb,mask=mask),self.atlas)['counts'],[None,None])
        mask[:]=False
        self.assertIsNone(rank_tile(dict(rgb=rgb,mask=mask),self.atlas)['block_id'])
        self.assertIsNone(count(None,self.atlas)['block_id'])
