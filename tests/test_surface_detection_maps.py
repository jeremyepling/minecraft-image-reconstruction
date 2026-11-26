import unittest

import numpy as np

from benchmark.camera_round import ROOT,read
from benchmark.count_round import FINDER
from benchmark.normal_round import groups
from benchmark.perimeter_sheet import load_parts
from benchmark.surface_detection_maps import detection_maps


class SurfaceDetectionMapsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=ROOT/'benchmark/reviews/perimeters-01'
        cls.method=read(cls.out/'selection.json')['selected']
        cls.cases={r['layout']:r for r in read(cls.out/'cases.json') if 'review' in r}
        cls.prior={r['key']:r for r in read(FINDER/'cases.json')}
        cls.frozen=read(FINDER/'manifest.json')
        splits,_,_=groups()
        cls.lookup={f'{c.floor}/{c.id}':c for cases in splits.values() for c in cases}

    def review(self,layout):
        row=self.cases[layout]
        parts=load_parts(self.out,row,self.prior[row['key']],self.frozen,self.method)
        return detection_maps(self.lookup[row['key']],parts,row['scores'][self.method])

    def test_native_reference_includes_visible_surfaces_between_scoring_samples(self):
        reference,predictions,data=self.review('courtyard')
        self.assertEqual(reference.shape,(900,1440))
        self.assertEqual(reference.shape,predictions.shape)
        unseen_at_stride=[r for r in data['references'] if r['samples']==0]
        self.assertEqual(len(unseen_at_stride),4)
        for r in unseen_at_stride:
            self.assertGreater(r['pixels'],0)
            self.assertEqual(r['pixels'],np.count_nonzero(reference==r['id']))
            self.assertEqual(r['status'],'small')
        self.assertEqual(data['counts']['visible'],12)
        self.assertEqual(data['counts']['found'],6)

    def test_secondary_matches_never_replace_official_ones(self):
        _,_,data=self.review('shopfront')
        p=data['predictions']
        found={x['reference'] for x in p if x['status']=='found'}
        small={x['reference'] for x in p if x['status']=='small_found'}
        self.assertEqual(len(found),12)
        self.assertEqual(len(small),2)
        self.assertTrue(found.isdisjoint(small))
        self.assertEqual(data['counts']['unpaired'],2)
        for r in data['references']:
            if r['status']=='small_found':
                self.assertEqual(r['overlap_grid'],'native')
                self.assertGreaterEqual(r['match_iou'],.5)

    def test_shared_region_explains_a_missed_plane(self):
        _,_,data=self.review('registry')
        footing=next(r for r in data['references'] if r['id']==1)
        self.assertEqual(footing['status'],'missed')
        self.assertGreater(footing['coverage'],.98)
        self.assertLess(footing['iou'],.5)
        self.assertEqual(footing['best_prediction_matched_ref'],2)
