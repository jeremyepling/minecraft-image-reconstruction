import unittest

import numpy as np
from PIL import Image

from benchmark.contained_splits import predict
from benchmark.crop_split_round import audit_partition
from engine.plane_consensus import row_runs


class ContainedSplitTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB',(120,80))
        self.parent = np.zeros((20,30),bool)
        self.parent[2:18,2:26] = True
        self.parent[6:9,5:8] = False
        self.region = self.parent.copy()
        self.region[:,14:] = False
        self.depth = np.zeros_like(self.parent,dtype=float)
        self.labels = np.ones_like(self.parent,dtype=int)
        self.fields = dict(relaxed=self.labels, relaxed_normals=np.array([[0.,0.,0.],[0.,0.,1.],[1.,0.,0.]]))

    def part(self, name, mask):
        return dict(id=name, support=dict(size=list(mask.shape[::-1]),runs=row_runs(mask)),normal_camera=[0.,0.,1.])

    def apply(self, neighbors=()):
        parents = [self.part('wall',self.parent),*neighbors]
        proposal = self.part('plane-consensus-2-1',self.region)
        parts, events = predict(self.image,parents,[proposal],self.depth,self.fields)
        return parents,parts,events

    def test_only_geometry_supported_seam_splits_and_keeps_native_hole(self):
        for signal in ('flat','depth','normal'):
            with self.subTest(signal=signal):
                self.labels[self.region] = 2 if signal=='normal' else 1
                self.depth[self.region] = 1 if signal=='depth' else 0
                parents,parts,events = self.apply()
                if signal=='flat':
                    self.assertEqual(parts,parents)
                    self.assertEqual(events[0]['reason'],'unsupported-seam')
                else:
                    self.assertTrue(events[0][signal+'_supported'])
                    self.assertEqual(events[0]['reason'],'accepted')
                    self.assertEqual(audit_partition(parents,parts,events)[0]['child_union_samples'],int(self.parent.sum()))

    def test_reject_opening_and_small_second_surface_before_clipping(self):
        self.labels[self.region] = 2
        self.region[6:9,5:8] = True
        parents,parts,events = self.apply()
        self.assertEqual(parts,parents)
        self.assertEqual(events[0]['reason'],'outside-parent')
        neighbor = np.zeros_like(self.parent)
        neighbor[6:9,5:8] = True
        parents,parts,events = self.apply([self.part('small-neighbor',neighbor)])
        self.assertEqual(parts,parents)
        self.assertEqual(events[0]['reason'],'crosses-second-parent')
