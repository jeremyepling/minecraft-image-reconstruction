import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine.atlas_count import _prepare
from engine.atlas_plane_depth import choose, options, place
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
        self.assertEqual(choose(rows,7,'combined'),level)
        self.assertEqual(choose(rows,7,'atlas'),level)

    def test_unknown_metric_camera_yields_empty_placement(self):
        img=Image.new('RGB',(33,25))
        camera={'P':None,'scale_status':'uncertain','reason':'missing camera'}
        with patch('engine.atlas_plane_depth._atlas',return_value=None):
            cells,status=place(img,'26.3',[],None,camera,'ranked',weights=np.zeros(len(NAMES)))
        self.assertFalse(cells)
        self.assertEqual(status['placed_blocks'],0)

    def test_zero_relative_evidence_preserves_original_plane(self):
        rows=[dict(level=k,atlas=.5,edge=.1,boundary=.2,tiles=3,distinct_votes=0,scene_delta=0)
              for k in (2,3)]
        self.assertEqual(ranked_choice(rows,3,np.zeros(len(NAMES))),3)
