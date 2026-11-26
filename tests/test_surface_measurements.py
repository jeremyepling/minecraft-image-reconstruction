import unittest

import numpy as np
from PIL import Image

from engine.surface_dimensions import dimensions
from engine.surface_tiles import visible_tiles, material_cells
from engine.face_evidence import effective_width


class SurfaceMeasurementsTest(unittest.TestCase):
    def test_whole_surface_extends_beyond_chart_and_keeps_unknown_axis(self):
        quad = [[20,10],[60,10],[60,50],[20,50]]
        # A hole in the middle does not shorten the outside span of the wall.
        runs = [[y,0,20] for y in range(5,25)] + [[y,25,40] for y in range(5,25)]
        region = dict(support=dict(runs=runs,size=[50,30]))
        result = dimensions(region,quad,(100,60),[4,None])
        self.assertEqual(result['blocks'],[8,None])
        self.assertAlmostEqual(result['spans'][0],8)

    def test_count_and_grid_origin_are_independent_and_holes_veto_cells(self):
        mask = np.ones((30,30),bool)
        mask[15,15] = False
        source = Image.new('RGB',(30,30),'gray')
        sample = dict(mask=mask,quad=[[0,0],[30,0],[30,30],[0,30]],source_image=source)
        self.assertEqual(list(visible_tiles(sample,dict(counts=[3,3],phase=[0,None]))),[])
        result = visible_tiles(sample,dict(counts=[3,3],phase=[0,0]))
        cells = {tuple(tile['cell']) for tile in result}
        self.assertEqual(cells,{(x,y) for y in range(3) for x in range(3)}-{(1,1)})

    def test_thin_complete_faces_reported_without_matching(self):
        from unittest.mock import Mock
        for width, eligible in [(2, False), (4.99, False), (5, True)]:
            # Long, oblique sides can both exceed five while the actual face is thin.
            quad = [[2,2], [22,2], [29,2+width], [9,2+width]]
            sample = dict(mask=np.ones((16,16),bool), quad=quad, source_image=Image.new('RGB',(40,20)))
            match = Mock(return_value={'block_id':'stone'})
            ranked, skipped = material_cells(sample,dict(counts=[1,1],phase=[0,0]),match)
            self.assertAlmostEqual(effective_width(quad), width)
            self.assertEqual((len(ranked), len(skipped)), (int(eligible), int(not eligible)))
            self.assertEqual(match.call_count, int(eligible))
