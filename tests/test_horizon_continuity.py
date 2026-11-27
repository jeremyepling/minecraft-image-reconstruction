import unittest

import numpy as np
from PIL import Image

from engine.horizon_continuity import refine
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


class HorizonContinuityTests(unittest.TestCase):
    def setUp(self):
        self.size = 320, 200
        self.rgb = np.empty((200,320,3),np.uint8)
        self.rgb[:90] = [179,205,250]
        self.rgb[90:] = [135,140,143]
        self.mask = np.zeros((200,320),bool)
        self.mask[89:93,30:170] = True
        self.part = dict(id='strip',support=dict(size=list(self.size),runs=row_runs(self.mask)))

    def test_connected_horizon_is_removed_without_mutating_measurement(self):
        before = native_mask(self.part,self.size)
        revised, events = refine(Image.fromarray(self.rgb),[self.part])
        self.assertFalse(revised)
        self.assertEqual(events[0]['removed'],int(before.sum()))
        np.testing.assert_array_equal(before,native_mask(self.part,self.size))

    def test_thin_real_detail_in_front_of_horizon_survives(self):
        self.rgb[91,90:123] = [153,95,43]
        revised, _ = refine(Image.fromarray(self.rgb),[self.part])
        self.assertEqual(len(revised),1)
        support = native_mask(revised[0],self.size)
        self.assertTrue(support[91,90:123].all())
        self.assertLess(support.sum(),self.mask.sum())

    def test_counted_platform_and_unconfirmed_skyline_are_preserved(self):
        surface = dict(self.part,quad=[[30,89],[170,89],[170,93],[30,93]])
        self.assertEqual(refine(Image.fromarray(self.rgb),[surface]),([surface],[]))
        self.rgb[70:95] = [80,78,80]
        self.assertEqual(refine(Image.fromarray(self.rgb),[self.part]),([self.part],[]))

    def test_disconnected_patch_between_upright_faces_is_preserved(self):
        self.rgb[60:120,28:40] = self.rgb[60:120,165:175] = [115,72,46]
        revised, events = refine(Image.fromarray(self.rgb),[self.part])
        self.assertFalse(events)
        self.assertEqual(revised,[self.part])
