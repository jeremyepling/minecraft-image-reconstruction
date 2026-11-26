import unittest

import numpy as np
from PIL import Image

from engine.plane_consensus import predict, row_runs, run_mask


class PlaneConsensusTest(unittest.TestCase):
    def setUp(self):
        self.photo = Image.new('RGB', (128,128), 'gray')
        self.wall = np.zeros((32,32), bool)
        self.wall[3:29,7:29] = True
        self.plane = np.zeros_like(self.wall)
        self.plane[8:21,10:17] = True

    def recover(self, preset):
        labels = np.zeros(self.wall.shape, np.int32)
        labels[self.wall] = 2
        labels[self.plane] = 1
        normals = np.asarray([[0.,0,0],[0.,0,1],[0.,0,1]])
        centers = np.asarray([[0.,0,0],[0.,0,4],[1.,0,4]])
        fields = dict(relaxed=labels, relaxed_normals=normals, relaxed_centers=centers)
        part = dict(id='wall', normal_camera=[0.,0,1], support=dict(size=[32,32],runs=row_runs(self.wall)))
        y,x = np.nonzero(self.plane)
        record = dict(quality=.95, stability=.95, bbox=[x.min(),y.min(),x.max()+1,y.max()+1])
        return predict(self.photo, [part], fields, preset,
                       sam_masks=self.plane[None], sam_records=[record])

    def test_neighbor_check_keeps_coplanar_wall_together(self):
        self.assertTrue(self.recover('agreement-only')[1])
        parts, events = self.recover('partial-agreement-and-step')
        self.assertEqual(events, [])
        self.assertEqual([p['id'] for p in parts], ['wall'])

    def test_agreed_extension_preserves_an_enclosed_hole_and_every_previous_pixel(self):
        self.plane[8:21,2:10] = True
        self.wall[13:15,10:12] = False
        parts, events = self.recover('partial-agreement-and-step')
        self.assertEqual(len(events), 1)
        masks = np.stack([run_mask(p['support']) for p in parts])
        covered = masks.any(axis=0)
        self.assertTrue(covered[self.wall].all())
        self.assertFalse(covered[13:15,10:12].any())
        self.assertTrue(covered[8:21,2:7].all())
        self.assertFalse(np.any(masks.sum(axis=0)>1))
