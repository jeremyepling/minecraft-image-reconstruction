import unittest
from unittest.mock import patch

import numpy as np
from benchmark import floating_expanded_wall_attachment as replay


class DetachedWallTest(unittest.TestCase):
    def test_filtered_child_and_parent_uv_cannot_reenter_output(self):
        parent=dict(id='wall',cells=[dict(u=u,v=v,block_id='stone') for u in range(3) for v in (0,1)],
                    axes=[1,2],normal_axis=3,H=np.eye(3).tolist())
        parent['export_uv']=[[u,v] for u in range(3) for v in (0,1) if (u,v)!=(2,1)]
        child=dict(parent,id='patch',cells=[dict(u=u,v=v,block_id='stone') for u in (0,1,2) for v in (0,1)])
        child['export_uv']=[[u,v] for u in (0,1,2) for v in (0,1) if (u,v)!=(2,0)]
        fit=dict(p90=.1,contacts=4,shift=[1,0])
        with patch.object(replay,'candidate',return_value=fit),patch.object(replay,'texture_depth',
                          return_value=(True,[0]*5,[])):
            additions,audit=replay.options(dict(charts=[parent,child]),{'wall':[0,0,0]},np.zeros((10,10,3)),{})
        self.assertEqual(audit[0]['blocked_child'],1)
        self.assertEqual(audit[0]['blocked_parent_uv'],1)
        self.assertEqual(len(additions['tangent']),4)
        self.assertNotIn((2,1,-1),[r[0] for r in additions['tangent']])
        self.assertNotIn((3,0,-1),[r[0] for r in additions['tangent']])

    def test_preserve_split_uses_actual_partition_membership(self):
        charts=[dict(id=k,cells=[dict(u=u,v=v,block_id='stone') for u in (0,1) for v in (0,1)],
                     axes=[1,2],normal_axis=3,H=np.eye(3).tolist()) for k in ('left','right')]
        with patch.object(replay,'candidate',return_value=dict(shift=[2,0],p90=0.,contacts=4)),patch.object(
                replay,'texture_depth',return_value=(True,[0]*4,[])):
            additions,audit=replay.options(dict(charts=charts),{'left':[0,0,0]},np.zeros((10,10,3)),
                                           {'left':'same source','right':'same source'})
        self.assertTrue(audit[0]['same_partition'])
        self.assertEqual(len(additions['tangent']),4)
        self.assertFalse(additions['tangent-preserve-split'])
