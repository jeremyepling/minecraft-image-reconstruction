import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import inset_returns, visible_inset_cells


class VisibleInsetTests(unittest.TestCase):
    def test_no_anchor_skips_camera_and_keeps_original_parts(self):
        image = Image.new('RGB', (32,32))
        parts = [{'id':'wall'}]
        with patch('engine.visible_placement.seam_camera', side_effect=AssertionError):
            result, events, secondary, camera = inset_returns.detect(image,parts,'26.3')
        self.assertIs(result,parts)
        self.assertEqual((events,secondary,camera),([],[],None))

    def test_unusable_projection_keeps_original_parts(self):
        image = Image.new('RGB',(32,32))
        parts = [{}, {}, {}]
        for camera in ({'P':None},{'P':np.zeros((3,4))}):
            result, events = inset_returns.refine(image,parts,camera)
            self.assertIs(result,parts)
            self.assertEqual(events,[])

    def test_clear_only_frontmost_cell_with_90_percent_support_in_rim(self):
        cells = [(0,1,4),(1,1,4),(2,1,4),(0,1,2)]
        owner = np.repeat(np.arange(len(cells)),10).reshape(4,10)
        covered = np.ones_like(owner,bool)
        covered[0,-1] = False
        covered[1,-2:] = False
        removed = visible_inset_cells._clear_front(cells,owner,covered,2,5,-2,[0,2,1,2],np.array([0,0,8]))
        self.assertEqual(removed,{cells[0]})
