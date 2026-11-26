import unittest
import numpy as np
from PIL import Image
from engine.camera_lines import _runs
from engine.surface_depth_split import split


class ParallelPlaneSplits(unittest.TestCase):
    def setUp(self):
        y,x=np.indices((80,110))
        self.mask=(x>=5)&(x<105)&(y>=4)&(y<76)
        self.mask[36:48,47:62]=False
        self.side=y > 30+x*.1
        self.depth=(.1+.0006*x+.0004*y).astype(np.float32)
        color=np.where(self.side,168,107).astype(np.uint8)
        self.image=Image.fromarray(np.repeat(color[:,:,None],3,axis=2))
        runs,=_runs(self.mask)
        self.part=dict(id='surface',support=dict(size=[110,80],runs=runs),
                       support_area=int(self.mask.sum()),normal_camera=[0.,0.,-1.],quad=[[0,0],[110,0],[110,80],[0,80]])

    def test_separates_parallel_offsets_and_preserves_opening(self):
        parts=split(self.image,[self.part],self.depth+self.side*.13)
        self.assertEqual(len(parts),2)
        votes=np.zeros(self.mask.shape,int)
        for p in parts:
            self.assertEqual(p['parent_id'],'surface')
            self.assertNotEqual(p.get('quad'),self.part['quad'])
            for y,x0,x1 in p['support']['runs']:
                votes[y,x0:x1]+=1
        np.testing.assert_array_equal(votes,self.mask)

    def test_rejects_palette_change_on_continuous_plane_and_depth_only_seam(self):
        self.assertEqual(split(self.image,[self.part],self.depth),[self.part])
        blank=Image.new('RGB',self.image.size,'gray')
        self.assertEqual(split(blank,[self.part],self.depth+self.side*.13),[self.part])
