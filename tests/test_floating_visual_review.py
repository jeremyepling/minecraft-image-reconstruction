import unittest
from benchmark.floating_visual_review import select, strict_delta


class ReviewQueueTest(unittest.TestCase):
    def test_material_replacement_is_distinct_from_geometry_change(self):
        truth={(0,0,0):'A',(1,0,0):'B',(2,0,0):'C'}
        before={(0,0,0):'A',(1,0,0):'A',(3,0,0):'X'}
        after={(1,0,0):'B',(2,0,0):'C',(4,0,0):'X'}
        gain,loss,counts=strict_delta(truth,before,after)
        self.assertEqual(gain,{(2,0,0)})
        self.assertEqual(loss,{(0,0,0)})
        self.assertEqual(counts,dict(gain=1,loss=1,remove=1,add=1,exact_gain=2,exact_loss=1))

    def test_disjoint_swap_and_gross_loss_select_independent_views(self):
        def frame(name, gain, loss, disjoint):
            return dict(dataset='v1',panel=None,id=name,changes={'method':dict(disjoint=disjoint,
                        dropped=['mask-1'] if disjoint else [],paired=dict(gain=gain,loss=loss))})
        cases=select([frame('arcade',154,28,True),frame('sawtooth',61,31,False)],'method')
        self.assertEqual(cases[('v1',None,'arcade')],['Disjoint component selection','Most new correct positions'])
        self.assertEqual(cases[('v1',None,'sawtooth')],['Most previously correct positions lost'])
