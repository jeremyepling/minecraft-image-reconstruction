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
            result, events, secondary, outlines, camera = inset_returns.detect(image,parts,'26.3')
        self.assertIs(result,parts)
        self.assertEqual((events,secondary,outlines,camera),([],[],[],None))

    def test_public_counts_skip_removed_backdrop_and_validate_same_host_at_placement(self):
        from copy import deepcopy
        from engine.surface_model import _inset_host_counts, _inset_layout_counts

        original = [dict(id='host'), dict(id='backdrop')]
        prior_host = dict(id='host', dimensions=dict(blocks=[None, None], spans=[None, None]))
        public = dict(surfaces=[dict(id='host', dimensions=dict(blocks=[13, None], spans=[12.7, None])),
                                dict(id='visible inset')])
        placement_public = deepcopy(public)
        context = dict(surfaces=[dict(id='host'), dict(id='backdrop')])
        with patch('engine.surface_model._count_regions', return_value=dict(surfaces=[prior_host])):
            _inset_host_counts(None, '26.3', original, public, {'host'}, 'masked-repeat', None)
            fit = _inset_layout_counts(None, '26.3', original, context, {'host'}, 'masked-repeat',
                                       None, public_counts=placement_public)
        self.assertEqual(public, placement_public)
        self.assertEqual(public['surfaces'][0]['dimensions']['blocks'], [None, None])
        self.assertEqual([r['id'] for r in public['surfaces']], ['host', 'visible inset'])
        self.assertEqual(fit['surfaces'], [prior_host, dict(id='backdrop')])

    def test_unusable_projection_keeps_original_parts(self):
        image = Image.new('RGB',(32,32))
        parts = [{}, {}, {}]
        for camera in ({'P':None},{'P':np.zeros((3,4))}):
            result, events = inset_returns.refine(image,parts,camera)
            self.assertIs(result,parts)
            self.assertEqual(events,[])

    def test_recount_preserves_partial_host_uncertainty_and_uses_previous_layout(self):
        from engine.surface_model import _inset_layout_counts
        image = Image.new('RGB', (32, 32))
        original = [dict(id='host'), dict(id='other')]
        before = dict(id='host', dimensions=dict(blocks=[None, None], spans=[None, None]))
        host = dict(id='host', dimensions=dict(blocks=[13, None], spans=[12.7, None]))
        counts = dict(surfaces=[host, dict(id='new'), dict(id='other')])
        with patch('engine.surface_model._count_regions', return_value=dict(surfaces=[before])):
            stable = _inset_layout_counts(image, '26.3', original, counts, {'host'}, 'test', None)
        self.assertEqual([r['id'] for r in stable['surfaces']], ['host', 'other'])
        self.assertIs(stable['surfaces'][0], before)
        self.assertEqual(host['dimensions'], dict(blocks=[None, None], spans=[None, None]))
        self.assertEqual(host['unconfirmed_host_dimension'], [dict(axis=0, estimate=13)])

    def test_clear_only_frontmost_cell_with_90_percent_support_in_rim(self):
        cells = [(0,1,4),(1,1,4),(2,1,4),(0,1,2)]
        owner = np.repeat(np.arange(len(cells)),10).reshape(4,10)
        covered = np.ones_like(owner,bool)
        covered[0,-1] = False
        covered[1,-2:] = False
        removed = visible_inset_cells._clear_front(cells,owner,covered,2,5,-2,[0,2,1,2],np.array([0,0,8]))
        self.assertEqual(removed,{cells[0]})
