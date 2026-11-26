"""A close view can divide a surface without extending it into an opening."""
import unittest

import numpy as np
from PIL import Image

from benchmark.crop_split_round import audit_partition
from engine.crop_splits import predict


class CropSplitTest(unittest.TestCase):
    def test_partition_preserves_hole_and_uses_first_equally_supported_crop(self):
        photo = Image.new('RGB', (80,80))
        # Two vertical regions are visible on either side of an open rectangle.
        labels = np.full((20,20),3,np.uint16)
        labels[:,2:10] = 1
        labels[:,10:18] = 2
        labels[6:14,8:12] = 3
        support = []
        for y in range(20):
            if 6<=y<14:
                support.extend(([y,2,8],[y,12,18]))
            else:
                support.append([y,2,18])
        parent = dict(id='wall',support={'size':[20,20],'runs':support}, normal_camera=[0.,0.,1.])
        normals = np.array([[0.,0.,0.],[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        centers = np.array([[0.,0.,0.],[0.,0.,1.],[1.,0.,1.],[0.,0.,1.]])
        evidence = [(tile,labels,normals,centers) for tile in (2,4)]
        parts, events = predict(photo,[parent],evidence)
        self.assertEqual(len(parts),2)
        self.assertEqual([e['tile'] for e in events],[2])
        self.assertEqual(audit_partition([parent],parts,events)[0]['child_union_samples'],288)

    def test_no_standalone_region_from_empty_parent_list(self):
        photo = Image.new('RGB',(32,32))
        labels = np.ones((8,8),np.uint16)
        geometry = np.ones((2,3))
        self.assertEqual(predict(photo,[],[(0,labels,geometry,geometry)]),([],[]))
