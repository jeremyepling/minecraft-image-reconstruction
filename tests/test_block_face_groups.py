"""Same-block association at convex corners and adjacent blocks at concave steps."""
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
from engine.block_face_groups import group_faces, assign_block_materials
from engine.face_evidence import effective_width


def observation(origin, first, second, normal, name):
    points = np.array([origin, origin+first, origin+first+second, origin+second])
    pixels = points[:,:2]/points[:,2,None]*320+np.array([320,240])
    return dict(quad=pixels.tolist(),normal_camera=normal.tolist(),surface_id=name)


class BlockFaceGroupTests(unittest.TestCase):
    def setUp(self):
        self.a=np.array([.8,0,.6])
        self.b=np.array([-.3,.8660254,.4])
        self.c=np.cross(self.a,self.b)
        self.o=np.array([0.,0.,8.])

    def test_two_and_three_visible_faces_share_one_read(self):
        a,b,c=self.a,self.b,self.c
        o=self.o
        faces=[observation(o,a,b,-c,'front'),observation(o,a,c,-b,'top'),observation(o,b,c,-a,'side')]
        for n in (1,2,3):
            groups=group_faces(faces[:n],(640,480))
            self.assertEqual([g['face_indices'] for g in groups],[list(range(n))])

    def test_concave_faces_touch_but_belong_to_separate_blocks(self):
        a,b,c=self.a,self.b,self.c
        o=self.o
        faces=[observation(o,a,-b,-c,'wall'),observation(o,a,-c,-b,'ledge')]
        self.assertEqual(len(group_faces(faces,(640,480))),2)

    def test_narrow_face_inherits_the_wider_faces_read(self):
        # At this camera angle the top has a measurable projected edge but
        # fewer than five independent pixels across its thickness.
        a=np.array([1.,0.,0.])
        b=np.array([0.,.9992,.04])
        c=np.cross(a,b)
        origin=np.array([0.,.3,10.])
        faces=[observation(origin,a,b,-c,'front'),observation(origin,a,c,-b,'top')]
        self.assertLess(effective_width(faces[1]['quad']),5)
        image=Image.new('RGB',(640,480))
        rank=Mock(return_value=dict(block_id='minecraft:stone',ranked_ids=['minecraft:stone']))
        groups=assign_block_materials(image,faces,rank)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['selected_index'],0)
        rank.assert_called_once()
        self.assertEqual(faces[1]['material'],{'block_id':'minecraft:stone'})

    def test_too_thin_single_face_stays_unresolved(self):
        thin=dict(quad=[[20.,20.],[40.,20.],[40.,21.],[20.,21.]],surface_id='top')
        rank=Mock()
        groups=assign_block_materials(Image.new('RGB',(64,64)),[thin],rank)
        self.assertIsNone(groups[0]['selected_index'])
        self.assertIsNone(groups[0]['material'])
        rank.assert_not_called()

    def test_duplicate_face_makes_join_ambiguous(self):
        a,b,c=self.a,self.b,self.c
        faces=[observation(self.o,a,b,-c,'front'),observation(self.o,a,c,-b,'top')]
        faces.append(dict(faces[1],surface_id='duplicate'))
        groups=group_faces(faces,(640,480))
        self.assertTrue(all(g['ambiguous_neighbors'] for g in groups))
        self.assertTrue(all(len(g['face_indices'])==1 for g in groups))

