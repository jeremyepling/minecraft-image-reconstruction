"""Analytical camera rays for the foreground conflict experiment."""
import unittest
from unittest.mock import patch
import sys

import numpy as np
from PIL import Image

from engine import atlas_plane_depth, visible_placement
from engine.foreground_conflicts import ForegroundConflicts

from engine.foreground_conflicts import first_hit


class ForegroundOwnershipTest(unittest.TestCase):
    def test_observed_tile_front_side_and_miss(self):
        eye = np.array([.5, .5, -3.])
        own = np.tile([.25, 0., 1.], (4, 1))
        probes = np.array([[.25, 0., 1.], [.15, 0., 1.], [-.10, 0., 1.]])
        hit = first_hit(own, probes, eye, 2, 0)
        np.testing.assert_allclose(hit[:2], [3., 10/3], atol=2e-6)
        self.assertTrue(np.isinf(hit[2]))
        self.assertTrue(np.isinf(first_hit(own[:3], probes, eye, 2, 0)).all())

    def test_public_default_selects_joint_depth_and_rollback_selects_independent(self):
        image = Image.new('RGB', (32, 24))
        camera = dict(P=np.eye(3, 4), scale_status='supported', reason='fixture')
        with patch.object(visible_placement, 'seam_camera', return_value=camera), \
             patch.object(atlas_plane_depth, '_atlas', return_value=object()), \
             patch.object(atlas_plane_depth, 'place_with_camera', return_value=(set(), {})) as place:
            _, status = visible_placement.place(image, '26.3', [{}])
            self.assertEqual(status['foreground_depth']['method'], 'joint')
            self.assertIsNotNone(place.call_args.kwargs['reconcile'])
            _, status = visible_placement.place(image, '26.3', [{}], joint_foreground=False)
            self.assertEqual(status['foreground_depth']['method'], 'independent')
            self.assertIsNone(place.call_args.kwargs['reconcile'])

    def test_unusable_conflict_measurement_retains_previous_decisions(self):
        def consume(*args, **kwargs):
            initial = [(0, 2, 17)]
            self.assertEqual(kwargs['reconcile'](initial), initial)
            return {(0, 1, 16)}, {}
        image = Image.new('RGB', (32, 24))
        with patch.object(atlas_plane_depth, '_atlas', return_value=object()), \
             patch.object(atlas_plane_depth, 'place_with_camera', side_effect=consume), \
             patch.object(ForegroundConflicts, 'resolve', side_effect=ValueError('fixture')):
            cells, status = atlas_plane_depth.place_ranked(image, '26.3', [{}], None, {'P':np.eye(3,4)})
        self.assertEqual(cells, {(0, 1, 16)})
        self.assertEqual(status['foreground_depth']['method'], 'independent')
        self.assertIn('fixture', status['foreground_depth']['reason'])

    def test_cli_selects_previous_rule(self):
        import reconstruct
        args = ['reconstruct', 'build', 'in.png', '--version', '26.3', '--through', 'placement',
                '--independent-plane-depth', '--out', 'out']
        with patch.object(sys, 'argv', args), patch('engine.surface_model.build') as build:
            reconstruct.main()
        self.assertFalse(build.call_args.kwargs['joint_foreground'])
