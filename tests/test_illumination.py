"""Control soft and hard lighting boundaries and hue/texture changes independently."""
import unittest
import numpy as np

from engine.illumination import MATERIAL_THRESHOLD, edge_evidence


class IlluminationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(442)
        self.texture=np.repeat(rng.uniform(75,165,(1,16,16,1)),3,axis=3)
        self.dark_field=np.linspace(.45,1,16)[None,None,:,None]

    def test_soft_and_step_multipliers_preserve_texture(self):
        for fields,model in [(self.dark_field,'smooth'),(np.where(self.dark_field<.72,.55,1.),'step')]:
            sample=self.texture*fields
            result=edge_evidence(self.texture,sample,return_gain=True)
            self.assertEqual(result['shading_model'][0],model)
            self.assertLess(result['material_score'][0],MATERIAL_THRESHOLD)
            self.assertTrue(result['observable'][0])
            corrected=self.texture*result['gain_map'][...,None]
            self.assertLess(abs(corrected-sample).mean(),5)

    def test_different_irregular_pattern_is_a_material_seam(self):
        other=self.texture[:,::-1,:,:]
        self.assertGreater(edge_evidence(self.texture,other)['material_score'][0],MATERIAL_THRESHOLD)

    def test_untextured_gray_shadow_is_ambiguous(self):
        flat=np.full_like(self.texture,115)
        self.assertFalse(edge_evidence(flat,flat*.55)['observable'][0])

    def test_chromatic_change_with_preserved_markings(self):
        tinted=self.texture*np.array([1.35,.9,.55])
        self.assertGreater(edge_evidence(self.texture,tinted)['material_score'][0],MATERIAL_THRESHOLD)
