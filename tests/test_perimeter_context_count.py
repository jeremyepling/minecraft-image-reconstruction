import unittest
import numpy as np
from PIL import Image

from benchmark.perimeter_context_count import context_distance,extend,ownership,project_reading
from engine.surface_sample import sample_surface


class CountingContextTest(unittest.TestCase):
    def test_unassigned_margin_stops_at_neighbors_openings_and_color_edges(self):
        image=Image.new('RGB',(52,42),'gray')
        runs=[[y,a,b] for y in range(10,30) for a,b in ([(12,35)] if y<15 or y>=25 else [(12,18),(26,35)])]
        part=dict(support=dict(size=list(image.size),runs=runs))
        next_wall=dict(support=dict(size=list(image.size),runs=[[y,36,40] for y in range(10,30)]))
        labels,x,y=ownership(image,[part,next_wall])
        smooth=np.full((42,52,3),80,np.int16)
        smooth[:8]=200
        dist,(ox,oy)=context_distance(part,1,labels,x,y,smooth,8)
        for px,py in [(20,20),(37,20),(40,19),(15,7),(12,12)]:
            self.assertGreater(dist[py-oy,px-ox],8)
        self.assertEqual(dist[20-oy,11-ox],1)
        self.assertEqual(dist[20-oy,4-ox],8)

    def test_straightened_sample_keeps_source_pixels_and_returns_chart_phase(self):
        yy,xx=np.indices((60,100))
        rgb=np.dstack((xx,yy,xx+yy)).astype(np.uint8)
        image=Image.fromarray(rgb)
        part=dict(quad=[[20,16],[80,16],[80,46],[20,46]],support=dict(size=[100,60],runs=[[y,20,80] for y in range(16,46)]))
        sample=sample_surface(image,part)
        labels,x,y=ownership(image,[part])
        d,offset=context_distance(part,1,labels,x,y,rgb.astype(np.int16),8)
        padded,extra,lo,span=extend(sample,d,offset,8)
        self.assertFalse((padded['mask']&extra).any())
        h,w=sample['mask'].shape
        px,py=np.rint(-lo*[w,h]).astype(int)
        keep=np.s_[py:py+h,px:px+w]
        np.testing.assert_array_equal(padded['mask'][keep],sample['mask'])
        np.testing.assert_array_equal(padded['rgb'][keep],sample['rgb'])
        self.assertTrue(extra.any())
        test_counts=np.array([6.,3.])
        test_phase=np.array([.1,.8])
        reading=dict(counts=(test_counts*span).tolist(),phase=((test_phase-lo*test_counts)%1).tolist())
        returned=project_reading(reading,lo,span)
        np.testing.assert_allclose(returned['counts'],test_counts)
        np.testing.assert_allclose(returned['phase'],test_phase)
