import unittest
import numpy as np

from benchmark.mask_metrics import measure, corners


def region(mask):
    runs = []
    for y,row in enumerate(mask):
        xs = np.flatnonzero(np.diff(np.r_[False,row,False]))
        runs.extend([[y,int(a),int(b)] for a,b in zip(xs[::2],xs[1::2])])
    return dict(support=dict(size=list(mask.shape[::-1]),runs=runs))


class MaskMetricTests(unittest.TestCase):
    def test_mask_only_hole_and_concave_corners(self):
        ids = np.zeros((32,45),int)
        ids[4:28,4:36] = 1
        ids[11:21,13:27] = 0
        ref = dict(id_map=ids,image_size=[180,128],patches={1:dict(eligible=True)},charts={})
        m,_ = measure([region(ids==1)],ref,use_mask=True)
        self.assertEqual((m['matched'],m['corner_true'],m['opening_filled'],m['corner_distance_sum']), (1,8,0,0))
        self.assertEqual(m['correct_pixels'],np.count_nonzero(ids))
        envelope = region(np.pad(np.ones((24,32),bool),((4,4),(4,9))))
        wrong,_ = measure([envelope],ref,use_mask=True)
        self.assertEqual((wrong['gaps_half_filled'],wrong['opening_filled']), (1,140))

    def test_abutting_in_projection_can_be_separate_in_depth(self):
        ids = np.zeros((24,32),int)
        ids[2:22,2:16],ids[2:22,16:30] = 1,2
        ref = dict(id_map=ids,image_size=[128,96],patches={i:dict(eligible=True) for i in (1,2)},
                   charts={1:np.eye(3),2:np.diag([1.,1.,3.])})
        parts = [region(ids==i) for i in (1,2)]
        m,_ = measure(parts,ref,use_mask=True)
        self.assertEqual((m['matched'],m['real_connections'],m['proposed_connections'],m['correct_connections']), (2,0,1,0))
        ref['charts'][2] = ref['charts'][1]
        m,_ = measure(parts,ref,use_mask=True)
        self.assertEqual((m['real_connections'],m['correct_connections']), (1,1))

    def test_screen_crop_has_no_artificial_corners(self):
        self.assertEqual(len(corners(np.ones((20,20),bool))),0)
