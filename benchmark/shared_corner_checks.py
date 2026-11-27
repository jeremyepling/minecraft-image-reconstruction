"""Synthetic regressions for independently measured row counts at one visible corner."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from benchmark.shared_corner_rows import pair
from engine.geometry import project


class SharedRowComparison(unittest.TestCase):
    def exercise(self, counts, strengths, receiver_height=12):
        size = 256
        lens = np.array([[180.,0,128],[0,180.,128],[0,0,1]])
        angle = .16
        camera = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        normals = np.array([[-1,0,-1],[1,0,-1]]) @ camera.T / np.sqrt(2)
        up = camera @ [0.,-1.,0.]
        origin = [0.,7.,17.]
        parts = []
        for i, (horizontal, height) in enumerate(zip((normals[1],-normals[0]),(12,receiver_height))):
            H = lens @ np.column_stack((horizontal, up, origin))
            start, finish = (-4,0) if i==0 else (0,4)
            q = project(H, [[start,height],[finish,height],[finish,0],[start,0]])
            mask = np.zeros((size,size),'uint8')
            cv2.fillConvexPoly(mask,np.rint(q).astype('int32'),1)
            parts.append(dict(id=str(i),label=f'S0{i+1}',q=q, normal_camera=normals[i],
                              rows=counts[i],width=4,strength=strengths[i],support_area=int(mask.sum()),
                              mask=mask,near=cv2.dilate(mask,np.ones((5,5),'uint8'))))
        rgb = np.zeros((size,size,3),'uint8')
        with patch('benchmark.shared_corner_rows.verify_sprite',return_value=None):
            return pair(*parts,rgb,np.ones(2),lens)

    def test_independent_counts_agree(self):
        result, charts = self.exercise([12,12],[.81,.78])
        self.assertEqual(result['row_comparison']['outcome'],'confirmed')
        self.assertEqual(result['receiver_rows'],12)
        self.assertEqual(len(charts),2)

    def test_stronger_receiver_becomes_donor(self):
        result, charts = self.exercise([8,12],[.49,.81])
        self.assertEqual(result['row_comparison']['outcome'],'stronger-donor')
        self.assertEqual((result['donor'],result['receiver_rows']),('S02',12))
        self.assertTrue(all(c['cells'] for c in charts))

    def test_unequal_totals_agree_on_partial_edge(self):
        result, charts = self.exercise([12,6],[.81,.78],receiver_height=6)
        self.assertEqual(result['row_comparison']['outcome'],'confirmed')
        self.assertEqual(result['receiver_rows'],6)
        self.assertEqual(max(c['v'] for c in charts[1]['cells']),5)

    def test_ambiguous_conflict_abstains(self):
        result, charts = self.exercise([8,12],[.61,.55])
        self.assertEqual(result['row_comparison']['outcome'],'ambiguous')
        self.assertEqual(charts,[])


if __name__=='__main__':
    unittest.main()
