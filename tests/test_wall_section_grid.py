import unittest

import numpy as np

from engine import wall_section_exchange as exchange, wall_section_grid as grid


class WallSectionTest(unittest.TestCase):
    def test_sections_cannot_offer_the_same_cell_twice(self):
        a,b,c=(0,1,2),(0,2,2),(0,3,2)
        group=grid.disjoint([[dict(cells=(a,b))],[dict(cells=(b,c)),dict(cells=(c,))]])
        self.assertEqual([[o['cells'] for o in g] for g in group],[[(a,b)],[(c,)]])

    def test_matching_cannot_spend_a_source_or_destination_twice(self):
        # The strongest single edge would strand source 1. The augmenting path moves source 0.
        edges=np.array([[0,2,0],[0,3,2],[1,2,2],[1,2,0]])
        choices=exchange.matching(edges,np.array([.95,.3,.5,.1]))
        self.assertEqual(set(choices),{(0,3,2),(1,2,2)})

    def test_no_camera_keeps_the_control(self):
        prior={(0,0,0)}
        camera={'P':None}
        self.assertEqual(exchange.recover(None,[],[],camera,prior)[0],prior)
        self.assertEqual(grid.recover(None,[],[],camera,prior)[0],prior)
