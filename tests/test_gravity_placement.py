"""Regression for airborne templates containing a material that falls when placed."""
import unittest
import numpy as np
from engine.reconstruction import ranked_stable_block, stable_alternative, stabilize_placements


class PlacementTests(unittest.TestCase):
    def test_falling_column_keeps_hole_and_supported_upper_block(self):
        blocks={(0,3,0):'suspicious_gravel',(0,4,0):'sand',(1,2,0):'tuff',(1,3,0):'white_concrete_powder'}
        alternates={(0,3,0):'coal_block',(0,4,0):'stone',(1,3,0):'white_concrete'}
        changes=stabilize_placements(blocks,alternates)
        self.assertEqual([(b['predicted'],b['placed']) for b in changes],[('suspicious_gravel','coal_block')])
        self.assertEqual(blocks[(0,4,0)],'sand')
        self.assertEqual(blocks[(1,3,0)],'white_concrete_powder')
        self.assertNotIn((0,2,0),blocks)

    def test_stable_texture_alias_precedes_other_texture(self):
        mapping={'gray': ['gravel','gray_wool'], 'coal_block':['coal_block']}
        self.assertEqual(stable_alternative(['gray','coal_block'],mapping),'gray_wool')

    def test_top_five_all_fall(self):
        falling=['sand','red_sand','gravel','suspicious_sand','white_concrete_powder']
        mapping={name:[name] for name in falling+['stone']}
        winner=ranked_stable_block(falling,mapping,None,(),np.array(list(mapping)),'multiscale',True)
        self.assertEqual(winner,'stone')


if __name__=='__main__':
    unittest.main()
