import unittest

import numpy as np

from benchmark.within_wall_grid import predict_surface


class WithinWallGridTest(unittest.TestCase):
    def test_quiet_material_requires_alignment_at_its_own_visible_edge(self):
        # The left six tiles have high contrast and an observable seam. The right
        # six have no texture at all. All pixels belong to one synthetic surface.
        rng=np.random.default_rng(128)
        tile=rng.integers(100,195,(48,16,3),dtype=np.uint8)
        tile[:,:2]=15
        colors=np.full((48,192,3),181,dtype=np.uint8)
        colors[:,:96]=np.tile(tile,(1,6,1))
        sample=dict(rgb=colors,mask=np.ones(colors.shape[:2],bool),
                    quad=[[0,0],[192,0],[192,48],[0,48]],normal_aspect=4)
        empty=dict(counts=[None,None],phase=[None,None])
        good=predict_surface(None,None,sample,empty)
        self.assertEqual(np.rint(good['counts']).tolist(),[12,3])
        self.assertEqual(good['evidence']['patch']['reason'],'donor seam phase reaches quiet far boundary')

        sample['mask'][:,-7:]=False
        cut=predict_surface(None,None,sample,empty)
        self.assertEqual(cut['counts'],[None,None])
        self.assertEqual(cut['evidence']['patch']['reason'],'far end does not confirm pitch or phase')
