from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import texture_guided_count as guided
from engine.surface_model import _count_regions


class TextureGuidedCountTests(unittest.TestCase):
    def test_complete_masked_count_keeps_original_chart_without_search(self):
        baseline = dict(method='masked-repeat', counts=[4, 2], phase=[None, None])
        sample = dict()
        with patch.object(guided, 'chart_options', side_effect=AssertionError('unneeded search')):
            result, chart = guided.count_surface(None, {}, sample, baseline)
        self.assertIs(result, baseline)
        self.assertIs(chart, sample)

    def test_saved_crop_and_cell_footprint_follow_selected_chart(self):
        photo = Image.new('RGB', (80, 50))
        plane = dict(id='wall', quad=[[2, 2], [22, 2], [22, 22], [2, 22]])
        quad = [[30., 6.], [70., 6.], [70., 26.], [30., 26.]]
        chosen = dict(quad=quad, mask=np.ones((20,40), bool),
                      rgb=np.full((20,40,3), 65, np.uint8), source_image=photo)
        measured = dict(method=guided.METHOD, counts=[2., 1.], phase=[0., 0.])
        with TemporaryDirectory() as dest, patch.object(guided, 'count_surface', return_value=(measured, chosen)):
            report = _count_regions(photo, [plane], '26.3', guided.METHOD, None, Path(dest))
            row = report['surfaces'][0]
            self.assertEqual(row['quad'], quad)
            self.assertEqual(row['dimensions']['blocks'], [2, 1])
            with Image.open(Path(dest)/row['sample']) as saved:
                self.assertEqual(saved.size, (40, 20))
                self.assertEqual(saved.getpixel((0,0)), (65,65,65,255))
            cells = row['visible_cells']
            self.assertEqual(len(cells), 2)
            self.assertAlmostEqual(min(q[0] for c in cells for q in c['quad']), 30.)
            self.assertAlmostEqual(max(q[0] for c in cells for q in c['quad']), 70.)

    def test_no_chart_does_not_register_an_accepted_fallback(self):
        with TemporaryDirectory() as dest:
            report = _count_regions(Image.new('RGB',(40,40)), [dict(id='unresolved')], '26.3',
                                    guided.METHOD, None, Path(dest))
            row = report['surfaces'][0]
            self.assertEqual(row['method'], 'masked-repeat')
            self.assertEqual(row['dimensions']['blocks'], [None, None])
