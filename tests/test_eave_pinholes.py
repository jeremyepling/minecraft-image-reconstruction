import unittest

import numpy as np
from PIL import Image

from engine.eave_pinholes import measurement_parts, refine
from engine.plane_consensus import row_runs, run_mask


def part(name,mask,normal):
    return dict(id=name,normal_camera=normal,support=dict(size=list(mask.shape[::-1]),runs=row_runs(mask)))


class VisibleEaveTest(unittest.TestCase):
    def setUp(self):
        # A narrow observed underside with a 2 x 3 unlabeled island well inside its rim.
        self.full = np.zeros((40,110),bool)
        self.full[10:24,10:96] = True
        self.hole = np.zeros_like(self.full)
        self.hole[15:17,48:51] = True
        self.color = np.zeros((40,110,3),np.uint8)
        self.color[:] = [140,170,230]
        self.color[self.full] = [46,52,58]
        self.eave = part('eave',self.full & ~self.hole,[0.,1.,0.])

    def test_visible_underside_hole_only(self):
        source=Image.fromarray(self.color)
        result, events=refine(source,[self.eave])
        self.assertEqual(events,[dict(host='eave',pixels=6,box=[48,15,51,17])])
        np.testing.assert_array_equal(run_mask(result[0]['support']),self.full)
        self.assertEqual(refine(source,result)[1],[])
        original = measurement_parts(result)[0]
        np.testing.assert_array_equal(run_mask(original['support']),self.full & ~self.hole)

    def test_protect_source_evidence_of_openings_and_other_faces(self):
        for color in ([140,170,230],[96,45,24]):
            image=self.color.copy()
            image[self.hole]=color
            self.assertEqual(refine(Image.fromarray(image),[self.eave])[1],[])
        neighbor=part('inset-face',self.hole,[0.,0.,-1.])
        source=Image.fromarray(self.color)
        self.assertEqual(refine(source,[self.eave,neighbor])[1],[])
        self.assertEqual(refine(source,[self.eave,dict(self.eave,id='overlapping')])[1],[])
        self.assertEqual(refine(source,[dict(self.eave,normal_camera=[0.,0.,-1.])])[1],[])

    def test_no_fill_near_silhouette_or_via_open_notch(self):
        source=Image.fromarray(self.color)
        for region in (np.s_[11:13,46:48],np.s_[10:17,48:50]):
            mask=self.full.copy()
            mask[region]=False
            self.assertEqual(refine(source,[part('eave',mask,[0.,1.,0.])])[1],[])
