import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import photo_edges
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha


class PhotoEdgeTests(unittest.TestCase):
    def test_shared_edge_moves_before_new_outlines_and_crops_are_created(self):
        owned = np.zeros((58,80),bool)
        owned[:,35:] = True
        rgb = np.full((*owned.shape,3),200,np.uint8)
        rgb[:,33:] = 40
        photo = Image.fromarray(rgb)
        parts = [dict(id=name,outline='old',source_crop='old',
                      support=dict(size=photo.size,runs=row_runs(mask)))
                 for name,mask in [('right',owned),('left',~owned)]]
        updated,stats = photo_edges.refine_surfaces(photo,parts)
        self.assertEqual(stats['applied_radius_px'],4)
        self.assertEqual(stats['changed_surfaces'],2)
        masks = []
        for pred in updated:
            self.assertNotIn('outline',pred)
            self.assertNotIn('source_crop',pred)
            alpha,box = _native_alpha(pred['support'],photo.size)
            full = Image.new('L',photo.size)
            full.paste(alpha,box)
            masks.append(np.asarray(full)>0)
        np.testing.assert_array_equal(masks[0],~masks[1])
        self.assertLessEqual(abs(np.flatnonzero(masks[0][20])[0]-33),1)

    def test_new_complete_count_retries_whole_frame_then_abstains(self):
        original = [{'id':'A','size':[None,5]}, {'id':'B','size':[1,4]}]
        def move(photo, parts, config):
            if config['radius']==4:
                return [dict(parts[0],size=[36,3]), parts[1]]
            return [parts[0], dict(parts[1],size=[None,None])]
        with patch.object(photo_edges,'refine_at_radius',side_effect=move), \
             patch.object(photo_edges,'_block_size',side_effect=lambda im,part:part['size']):
            updated,stats = photo_edges.refine_surfaces(None,original)
        self.assertIs(updated,original)
        self.assertEqual(stats['applied_radius_px'],0)
        self.assertEqual([r['rejected_by']['surface'] for r in stats['attempts']], ['A','B'])

    def test_smaller_adjustment_preserves_partial_and_complete_count_status(self):
        original = [{'id':'face','size':[None,5]}]
        def move(photo, parts, config):
            return [dict(parts[0],size=[4,5] if config['radius']==4 else [None,4])]
        with patch.object(photo_edges,'refine_at_radius',side_effect=move), \
             patch.object(photo_edges,'_block_size',side_effect=lambda im,part:part['size']):
            updated,stats = photo_edges.refine_surfaces(None,original)
        self.assertEqual(stats['applied_radius_px'],2)
        self.assertEqual(updated[0]['size'],[None,4])
