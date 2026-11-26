import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine.atlas_count import _prepare
from engine.atlas_plane_depth import frozen_weights, options, place_ranked
from engine.plane_offset_features import NAMES, ranked_choice


class AtlasPlaneDepthTests(unittest.TestCase):
    def test_registered_adjacent_sprites_select_the_visible_plane(self):
        # An oblique camera origin breaks ties between neighboring integer depths.
        w,h,f=384,256,220
        eye=np.array([.31,2.27,-.37])
        basis=np.array([[f,0,w/2],[0,-f,h/2],[0,0,1.]])
        P=np.column_stack([basis,-basis@eye])
        yy,xx=np.indices((h,w))
        rays=np.stack([(xx+.5-w/2)/f,-(yy+.5-h/2)/f,np.ones((h,w))],axis=2)
        level=5
        xyz=eye+(level-eye[2])*rays
        rng=np.random.default_rng(129)
        tiles=rng.integers(15,240,(40,16,16,3),dtype=np.uint8)
        atlas=_prepare(tiles,[str(i) for i in range(40)],[[str(i)] for i in range(40)],{})
        cells=np.floor(xyz[:,:,:2]).astype(int)
        uv=np.floor((xyz[:,:,:2]-cells)*16).astype(int)
        texture=(cells[:,:,0]+cells[:,:,1])%2
        rgb=tiles[texture,uv[:,:,1],uv[:,:,0]]
        # Select only the observed rectangle bounded by known unit-block seams.
        mask=(cells[:,:,0]>=-2)&(cells[:,:,0]<3)&(cells[:,:,1]>=0)&(cells[:,:,1]<4)
        small=mask[2::4,2::4]
        xy=4*np.indices(small.shape).transpose(1,2,0)[:,:,::-1]+2
        ray4=np.stack([(xy[:,:,0]-w/2)/f,-(xy[:,:,1]-h/2)/f,np.ones(small.shape)],axis=2)
        rows=options(small,P,eye,ray4,2,7,rgb,atlas)
        true=next(r for r in rows if r['level']==level)
        self.assertGreater(true['atlas'],.9)
        self.assertGreaterEqual(true['distinct_votes'],8)
        self.assertEqual(ranked_choice(rows,7,frozen_weights()),level)

    def test_unknown_metric_camera_yields_empty_placement(self):
        img=Image.new('RGB',(33,25))
        camera={'P':None,'scale_status':'uncertain','reason':'missing camera'}
        with patch('engine.atlas_plane_depth._atlas',return_value=None):
            cells,status=place_ranked(img,'26.3',[],None,camera)
        self.assertFalse(cells)
        self.assertEqual(status['placed_blocks'],0)

    def test_zero_relative_evidence_preserves_original_plane(self):
        rows=[dict(level=k,atlas=.5,edge=.1,boundary=.2,tiles=3,distinct_votes=0,scene_delta=0)
              for k in (2,3)]
        self.assertEqual(ranked_choice(rows,3,np.zeros(len(NAMES))),3)

    def test_unavailable_assets_leave_the_original_candidates_in_place(self):
        rgb=Image.new('RGB',(33,25))
        cells={(1,2,3)}
        with patch('engine.atlas_plane_depth._atlas',return_value=None), \
             patch('engine.atlas_plane_depth.place_with_camera',return_value=(cells,{})) as original:
            output,status=place_ranked(rgb,'26.3',[{}],None,{'P':np.eye(3,4)})
            self.assertIs(output,cells)
            self.assertIn('atlas',status['plane_depth_reason'])
            self.assertEqual(len(original.call_args.args),5)

    def test_missing_or_malformed_weights_can_be_repaired(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'model.json'
            with patch('engine.atlas_plane_depth.WEIGHTS',path):
                self.assertIsNone(frozen_weights())
                path.write_text('{"features":[],"weights":[]}')
                self.assertIsNone(frozen_weights())
                path.write_text(json.dumps(dict(schema=1,version='26.3',features=NAMES,weights=[1.]*len(NAMES))))
                self.assertEqual(frozen_weights().tolist(),[1.]*len(NAMES))
