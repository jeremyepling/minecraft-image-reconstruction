import unittest
from unittest.mock import Mock, patch

import numpy as np

from benchmark import single_wall_metrics as metrics


class WallVoteAuditTests(unittest.TestCase):
    def test_same_surface_material_seam_and_signed_depth_step(self):
        # Four screen samples across each row. The top row has a nearer parallel
        # wall on the left. The bottom row is flat but changes material in the middle.
        reference=dict(id_map=np.array([[1,1,2,2],[1,1,1,1]]),
                       charts={1:np.eye(3),2:np.diag([1.,1.,2.])},
                       patches={i:dict(normal=[0,0,1],offset=i) for i in (1,2)})
        cells={1:dict(faces={(6,6):{'block_id':'a'},(10,6):{'block_id':'b'}})}
        def candidate(y,i):
            return dict(parent='wall',proposal=i,samples=[[y,0,y,1,y,2,y,3]])
        with patch.object(metrics,'recorded',return_value=(None,reference)),patch.object(metrics,'read',return_value={'faces':[]}),patch.object(metrics,'reference_planes',return_value=cells):
            truth=metrics.boundary_truth(Mock(),[candidate(0,0),candidate(1,1)])
        self.assertEqual([(r['kind'],r['sign'],r['material_seam']) for r in truth],
                         [('parallel step',1,False),('flush wall',0,True)])
        decisions=[dict(parent='wall',proposal=i,accepted=True,median_step=-1.) for i in (0,1)]
        score=metrics.grade_votes(truth,decisions)
        self.assertEqual((score['order_candidates'],score['order_correct'],score['flush_material_cuts']),(1,0,1))
