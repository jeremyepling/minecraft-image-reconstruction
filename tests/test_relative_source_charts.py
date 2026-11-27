import unittest

import numpy as np
from PIL import Image

from benchmark.relative_source_charts import grow, counter_chart
from benchmark.relative_chart_grade import sheet_score
from engine.geometry import transform, project, UNIT
from benchmark.full_block_lattice import atlas


class ChartTest(unittest.TestCase):
    def test_relative_chart_grows_only_observed_cells(self):
        catalog=atlas()
        tile=Image.fromarray(catalog.tiles[catalog.labels.index('bricks')])
        tile=np.asarray(tile.resize((32,32),Image.Resampling.NEAREST))
        image=np.pad(np.tile(tile,(3,4,1)),((16,16),(16,16),(0,0)))
        mask=np.zeros((32,40),bool)
        mask[3:29,3:37]=True
        mask[12:20,12:20]=False
        H=transform([[16,112],[48,112],[48,80],[16,80]])
        cells=grow(image,H,mask)
        expected={(u,v) for u in range(4) for v in range(3)}-{(1,1)}
        found={(c['u'],c['v']) for c in cells}
        self.assertTrue(found)
        self.assertFalse(found-expected)
        self.assertIn((3,2),found)
        self.assertTrue(all(c['block_id']=='bricks' for c in cells))

    def test_counter_homography_uses_both_measured_phases(self):
        image=np.zeros((128,128,3),'B')
        mask=np.ones((32,32),bool)
        q=np.array([[16,16],[112,16],[112,112],[16,112]])
        H,_=counter_chart(image,q,mask,dict(counts=[3,3],phase=[.25,.5]))
        np.testing.assert_allclose(project(H,[[0,0],[1,1]]),[[24,96],[56,64]])

    def test_evaluator_requires_common_lattice_offset_per_sheet(self):
        H=np.diag([8.,8.,1.])
        front=np.repeat(np.arange(1,4,dtype='H'),8)[None,:].repeat(8,0)
        chart=dict(H=H,cells=[dict(u=i,v=0,block_id='bricks') for i in range(3)])
        faces={i+1:dict(id=i+1,block=1,xyz_world=[10+i+int(i==2),0,0],
                        normal_world=[0,0,1],corners_px=project(H,UNIT+[i,0])) for i in range(3)}
        result=sheet_score(chart,front,faces,{1:dict(block_id='minecraft:bricks')})
        self.assertEqual(result['consistent_integer_offsets'],2)
        self.assertEqual(result['exact_center_material'],3)
        self.assertEqual(result['corners_within_10pct_edge'],3)


if __name__=='__main__':
    unittest.main()
