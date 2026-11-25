import unittest

import numpy as np

from engine.routing import trusted_cells
from engine.appearance import descriptors, cost_matrix, palette_cost, prototypes


class AppearanceTests(unittest.TestCase):
    def test_tiny_consistent_planes_require_some_spatial_agreement(self):
        scores = np.full(100,.25)
        scores[:5] = .9
        self.assertTrue(trusted_cells(scores,np.full(100,7)).all())
        self.assertEqual(trusted_cells(scores,np.full(100,18)).sum(),5)
        self.assertFalse(trusted_cells(np.full(100,.4),np.full(100,7)).any())

    def test_equal_means_different_chroma(self):
        gray = np.full((16,16,3),120,dtype=np.uint8)
        colored = gray.copy()
        colored[::2] = [160,80,120]
        colored[1::2] = [80,160,120]
        references = descriptors([gray,colored]).astype(np.float32)
        actual = cost_matrix([gray,colored],references).argmin(axis=1)
        self.assertEqual(actual.tolist(),[0,1])

    def test_shared_palette_preserves_distinctive_accent(self):
        costs = np.array([[0.,.001,.4], [.001,0.,.4], [.001,0.,.4], [.5,.5,0.]])
        shared = palette_cost(costs,True).argmin(axis=1)
        self.assertEqual(shared.tolist(),[1,1,1,2])
        self.assertEqual(palette_cost(costs,False).argmin(axis=1).tolist(),[0,1,1,2])

    def test_bad_exemplar_cannot_turn_gray_atlas_green(self):
        tiles = np.array([np.full((16,16,3),x,np.uint8) for x in (30,80,130,180,230)])
        labels = np.array(list('abcde'))
        rendered = tiles.copy()
        rendered[2] = [20,200,20]
        refs = prototypes(tiles,labels,rendered,labels)
        self.assertEqual(cost_matrix([tiles[2]],refs).argmin(),2)
        self.assertTrue(np.allclose(refs[2,1:3],0))


if __name__ == '__main__':
    unittest.main()
