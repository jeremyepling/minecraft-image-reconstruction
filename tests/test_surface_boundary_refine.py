import unittest

import numpy as np
from PIL import Image

from engine.camera_lines import _runs
from engine.surface_boundary_refine import _retain_topology, refine


def part(mask, name):
    runs = [list(run) for group in _runs(mask) for run in group]
    return dict(id=name,support=dict(size=list(mask.shape[::-1]),runs=runs),support_area=int(mask.sum()))


def decode(surface):
    mask = np.zeros(surface['support']['size'][::-1],bool)
    for y,lo,hi in surface['support']['runs']:
        mask[y,lo:hi] = True
    return mask


class BoundaryRefineTests(unittest.TestCase):
    def test_displaced_boundary_moves_to_within_one_sample_of_photographed_step(self):
        mask = np.zeros((58,80),bool)
        mask[:,35:] = True
        rgb = np.full((58,80,3),200,np.uint8)
        rgb[:,33:] = 40
        parts = [part(mask,'right'),part(~mask,'left')]
        right,left = refine(Image.fromarray(rgb),parts)
        for y in (15,30,42):
            self.assertLessEqual(abs(np.nonzero(decode(right)[y])[0][0]-33),1)
        np.testing.assert_array_equal(decode(right),~decode(left))
        self.assertEqual(right['support_area'],int(decode(right).sum()))
        self.assertEqual(parts[0]['support_area'],int(mask.sum()))

    def test_a_uniform_photo_leaves_a_stepped_region_and_hole_alone(self):
        mask = np.zeros((45,50),bool)
        mask[5:36,4:43] = True
        mask[5:13,35:43] = False
        mask[18:29,14:20] = False
        p = part(mask,'wall')
        self.assertEqual(refine(Image.new('RGB',(50,45),'gray'),[p]),[p])

    def test_closing_a_door_filling_a_hole_and_joining_islands_are_vetoed(self):
        labels = np.zeros((28,40),np.int32)
        labels[4:22,3:21] = 1
        labels[8:13,7:10] = 0
        labels[15:22,13:16] = 0
        labels[10:17,23:26] = 1
        target = labels.copy()
        target[8:13,7:10] = 1
        target[21,13:16] = 1
        target[11,21:23] = 1
        got = _retain_topology(labels,target)
        for label in (0,1):
            self.assertEqual(len(_runs(labels==label)),len(_runs(got==label)))
        self.assertTrue((got[8:13,7:10]==0).any())
        self.assertTrue((got[21,13:16]==0).any())
        self.assertTrue((got[11,21:23]==0).any())
