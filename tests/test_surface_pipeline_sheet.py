import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from benchmark.surface_pipeline_sheet import STAGES, _straight_photo, build


class PipelineSheetGridTest(unittest.TestCase):
    def test_only_located_axis_is_drawn_and_openings_stay_empty(self):
        rgb = np.full((20,40,3),72,np.uint8)
        mask = np.ones(rgb.shape[:2],bool)
        mask[6:15,5:36] = False
        clean,grid,lines = _straight_photo(dict(rgb=rgb,mask=mask),
                                           dict(counts=[4.,3.],phase=[.25,None]))
        a,b = np.asarray(clean),np.asarray(grid)
        drawn = np.any(a!=b,axis=2)
        self.assertTrue(drawn.any())
        self.assertFalse(drawn[~mask].any())
        self.assertEqual(lines,[[.0625,.3125,.5625,.8125],[]])
        self.assertEqual(np.count_nonzero(drawn[0]),4)
        self.assertFalse(np.any(drawn[:,7]))

    def test_known_spacing_does_not_create_grid_lines_without_an_origin(self):
        rgb = np.full((12,23,3),110,np.uint8)
        a,b,lines = _straight_photo(dict(rgb=rgb,mask=np.ones(rgb.shape[:2],bool)),
                                    dict(counts=[5.,2.],phase=[None,None]))
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        self.assertEqual(lines,[[],[]])

    def test_portrait_matrix_keeps_each_building_in_one_column(self):
        image=Image.new('RGB',(24,24),'gray')
        part=dict(id='wall',support=dict(size=[24,24],runs=[[y,3,19] for y in range(6,17)]))
        examples=[dict(layout=f'building-{i}',key=f'view-{i}',image=image,box=[0,0,24,24],part=part)
                  for i in range(8)]
        extra=dict(examples[1],extra='Second surface')
        sheet,records=build([*examples,extra],SimpleNamespace(provenance={'version':'26.3'}))
        self.assertGreater(sheet.height,sheet.width)
        self.assertEqual(records['columns'],[ex['layout'] for ex in examples])
        self.assertEqual([r['name'] for r in records['stages']],list(STAGES))
        self.assertEqual(records['views'][1]['sheet_column_xyxy'],records['views'][-1]['sheet_column_xyxy'])
        columns=[v['sheet_column_xyxy'] for v in records['views'][:8]]
        self.assertTrue(all(a[2]<=b[0] for a,b in zip(columns,columns[1:])))
        for view in records['views']:
            for stage,box in zip(records['stages'],view['sheet_stage_xyxy'],strict=True):
                self.assertTrue(stage['y0']<=box[1]<box[3]<=stage['y1'])
