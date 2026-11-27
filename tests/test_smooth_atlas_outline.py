import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from engine import smooth_atlas_outline as outline
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


class SmoothOutlineTests(unittest.TestCase):
    def test_two_pixel_border_keeps_join_and_original_measurement(self):
        host = np.zeros((19, 23), bool)
        host[2:17, 3:11] = True
        new = [(y, 11, 20) for y in range(2, 17)]
        runs = outline.supported_runs(host, new, [11, 2, 20, 17], 2)
        self.assertEqual(runs, [(y, 11, 18) for y in range(4, 15)])
        image = Image.new('RGB', (23, 19))
        part = dict(id='host', support=dict(size=list(image.size), runs=row_runs(host)))
        updated = outline.apply_regions(image, [part], [dict(surface='host', runs=runs)])
        self.assertGreater(native_mask(updated[0], image.size).sum(), host.sum())
        measured = outline.measurement_parts(updated)[0]
        self.assertEqual(measured['support'], part['support'])
        self.assertNotIn('support_area', measured)
        self.assertNotIn('measured_smooth_support', part)

    def test_absent_candidate_or_atlas_keeps_masks(self):
        parts = []
        loader = Mock(return_value=None)
        with patch.object(outline, 'proposals', return_value=[]):
            result, status = outline.refine(Image.new('RGB', (16, 16)), parts, '26.3', atlas_loader=loader)
        self.assertIs(result, parts)
        loader.assert_not_called()
        with patch.object(outline, 'proposals', return_value=[{}]):
            result, status = outline.refine(Image.new('RGB', (16, 16)), parts, '26.3', atlas_loader=loader)
        self.assertIs(result, parts)
        self.assertEqual(status['status'], 'atlas unavailable')
