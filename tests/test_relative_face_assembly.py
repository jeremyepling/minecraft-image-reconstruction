
from itertools import product
import unittest

import numpy as np

from benchmark.relative_face_assembly import assemble, solve_origins


IMAGE = np.array([[18.,3.,-10.], [5.,-20.,-4.], [.004,-.001,.003]])
IMAGE_ORIGIN = np.array([460.,380.,1.])


def sheet(name, axes, normal, level, extents, uv_origin):
    """Generate a local grid from known test geometry with a private UV origin."""
    corners = np.zeros(3,int)
    corners[abs(normal)-1]=level
    directions=np.eye(3,dtype=int)[:,np.asarray(axes)-1]
    translated=corners+directions@uv_origin
    homography=np.column_stack((IMAGE@directions,IMAGE@translated+IMAGE_ORIGIN))
    cells=[dict(u=u-uv_origin[0],v=v-uv_origin[1],block_id='minecraft:stone')
           for u,v in product(*(range(*limits) for limits in extents))]
    return dict(id=name, axes=axes, normal_axis=normal, cells=cells, H=homography)


class RelativeFaceAssemblyTest(unittest.TestCase):
    def test_join_wall_side_and_underside_and_hold_disconnected_sheet(self):
        charts=[
            sheet('front',[1,2],3,3,((0,2),(0,3)),(9,-4)),
            sheet('side',[3,2],1,2,((0,3),(0,3)),(-6,14)),
            sheet('underside',[1,3],-2,0,((0,2),(0,3)),(12,-7)),
            sheet('independent',[1,2],3,3,((18,20),(0,2)),(2,2)),
        ]
        result=assemble(charts)
        best,detached=result['groups']
        self.assertEqual(set(best['origins']),{'front','side','underside'})
        self.assertEqual(set(detached['origins']),{'independent'})
        expected=({(x,y,2) for x,y in product(range(2),range(3))}
                  | {(1,y,z) for y,z in product(range(3),range(3))}
                  | {(x,0,z) for x,z in product(range(2),range(3))})
        # The anchor chart's local corner is the only origin.
        actual={tuple(np.array(p)+[9,-4,3]) for p in best['blocks']}
        self.assertEqual(actual,expected)
        self.assertEqual(len(result['conflicts']),0)

    def test_export_mask_preserves_original_component_priority_and_joins(self):
        front=sheet('front',[1,2],3,3,((0,2),(0,3)),(0,0))
        side=sheet('side',[3,2],1,2,((0,3),(0,3)),(0,0))
        independent=sheet('other',[1,2],3,3,((18,20),(0,2)),(0,0))
        ordinary=assemble([front,side,independent])
        front['export_uv']=[[0,0]]
        side['export_uv']=[]
        gated=assemble([front,side,independent])
        self.assertEqual(gated['joins'],ordinary['joins'])
        self.assertEqual(gated['groups'][0]['origins'],ordinary['groups'][0]['origins'])
        self.assertEqual(len(gated['groups'][0]['blocks']),1)
        self.assertEqual(gated['groups'][1]['blocks'],ordinary['groups'][1]['blocks'])

    def test_require_visible_evidence_on_both_sides(self):
        front=sheet('front',[1,2],3,3,((0,2),(0,3)),(0,0))
        side=sheet('side',[3,2],1,2,((0,3),(0,3)),(0,0))
        for cell in side['cells']:
            cell['visible_edges']=[]
        result=assemble([front,side])
        self.assertEqual(len(result['groups']),2)
        self.assertFalse(result['joins'])

    def test_assignment_follows_the_face_that_supplied_the_exported_id(self):
        front=sheet('front',[1,2],3,3,((0,2),(0,3)),(0,0))
        side=sheet('side',[3,2],1,2,((0,3),(0,3)),(0,0))
        front['confidence']=1.0
        for cell in front['cells']:
            cell.update(block_id='minecraft:smooth_stone',fallback=True)
        group=assemble([front,side])['groups'][0]
        fallback={p for p, block_id in group['blocks'].items() if block_id=='minecraft:smooth_stone'}
        self.assertEqual(len(fallback),6)
        self.assertEqual(group['assigned'],group['blocks'].keys()-fallback)

    def test_reject_incompatible_cycle(self):
        def edge(a,b,dx):
            return dict(i=a,j=b,shift=(dx,0,0))
        a,b,c=edge(0,1,2),edge(1,2,3),edge(0,2,4)
        rows,used,rejected=solve_origins(3,[a,b,c])
        self.assertEqual(rows[0][2].tolist(),[5,0,0])
        self.assertEqual(used,[a,b])
        self.assertEqual(rejected,[c])



    def test_coplanar_charts_join_across_a_visible_gap(self):
        front=sheet('front',[1,2],3,3,((0,2),(0,3)),(0,0))
        detached=sheet('detached',[1,2],3,3,((4,6),(1,3)),(7,-8))
        parallel=sheet('parallel',[1,2],3,13,((2,4),(0,3)),(-6,5))
        result=assemble([front,detached,parallel],edge_pixels=.01,coplanar_pixels=.01)
        self.assertEqual(len(result['groups']),2)
        self.assertEqual(set(result['groups'][0]['origins']),{'front','detached'})
        self.assertEqual(len(result['groups'][0]['blocks']),10)
        self.assertNotIn((2,1,-1),result['groups'][0]['blocks'])

    def test_changing_local_sheet_origins_moves_entire_build_as_one(self):
        sizes=[('front',[1,2],3,3,((0,2),(0,3))),
               ('side',[3,2],1,2,((0,3),(0,3)))]
        a,b=(assemble([sheet(*spec,origin) for spec,origin in zip(sizes,origins,strict=True)])
             for origins in ([(9,-4),(-6,14)],[(25,12),(0,-16)]))
        x,y=(list(z['groups'][0]['blocks']) for z in (a,b))
        delta=np.array(min(x))-min(y)
        self.assertEqual(set(x),{tuple(np.array(p)+delta) for p in y})


if __name__ == '__main__':
    unittest.main()
