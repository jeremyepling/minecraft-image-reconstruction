import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import plane_repeat_evidence as repeat


class PlaneRepeatTests(unittest.TestCase):
    def test_model_repaired_during_same_process(self):
        model = json.loads(repeat.WEIGHTS.read_bytes())
        with TemporaryDirectory() as temp, patch.object(repeat, 'WEIGHTS', Path(temp)/'ranker.json'):
            self.assertIsNone(repeat.frozen_repeat_weights())
            for override in ({'schema':9}, {'version':'other'}, {'features':[]}, {'weights':[float('nan')]*17}):
                repeat.WEIGHTS.write_text(json.dumps(dict(model, **override)))
                self.assertIsNone(repeat.frozen_repeat_weights())
            repeat.WEIGHTS.write_text(json.dumps(model))
            np.testing.assert_equal(repeat.frozen_repeat_weights(), model['weights'])

    def test_missing_weights_use_the_previous_placement_without_wide_search(self):
        image = Image.new('RGB',(45,39))
        camera = {'P':np.eye(3,4)}
        from engine import atlas_plane_depth
        with patch.object(repeat,'frozen_repeat_weights',return_value=None), \
             patch.object(atlas_plane_depth,'_atlas',return_value=object()), \
             patch.object(atlas_plane_depth,'place_with_camera',return_value=(set(),{})) as fallback, \
             patch.object(repeat,'scored_rows') as scoring:
            _, status = atlas_plane_depth.place_ranked(image,'26.3',[{}],None,camera)
        self.assertIsNone(fallback.call_args.kwargs['recover_edges'])
        self.assertFalse(fallback.call_args.kwargs['walls_first'])
        self.assertEqual(status['plane_depth_search'],'original-window')
        scoring.assert_not_called()

    def test_blank_or_too_small_mask_provides_zero_repeat_evidence(self):
        rgb=np.zeros((45,49,3),np.uint8)
        mask=np.ones((11,12),bool)
        P=np.array([[36,0,24,0],[0,-36,22,0],[0,0,1,0]],float)
        self.assertEqual(repeat.repeat_scores(rgb,mask,P,np.zeros(3),2,[3,8]),[(0.,0.),(0.,0.)])
        mask[:]=False
        self.assertEqual(repeat.repeat_scores(rgb,mask,P,np.zeros(3),2,[3]),[(0.,0.)])
        self.assertEqual(repeat.select_repeat([],7,None),7)

    def test_known_block_pitch_scores_above_adjacent_wrong_plane(self):
        y,x=np.indices((149,189))
        rng=np.random.default_rng(18)
        tex=rng.integers(20,220,(16,16,3),dtype=np.uint8)
        # Frontal plane at distance four and focal length 128 gives 32px blocks.
        rgb=tex[y//2%16,x//2%16]
        P=np.array([[128,0,94,0],[0,-128,74,0],[0,0,1,0]],float)
        mask=np.ones(((rgb.shape[0]+1)//4,(rgb.shape[1]+1)//4),bool)
        wrong,true=repeat.repeat_scores(rgb,mask,P,np.zeros(3),2,[5,4])
        self.assertGreater(true[1],wrong[0]+.25)
