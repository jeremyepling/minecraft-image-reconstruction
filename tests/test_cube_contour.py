import unittest

import numpy as np
from PIL import Image

from engine.cube_contour import refine


class CubeContourTests(unittest.TestCase):
    def test_cube_runs_meet_and_preserve_opening_and_frame_edge(self):
        y,x = np.indices((165,400))
        target = (x>=60)&(y>=35)&(y<135)&~((x>=200)&(x<230)&(y>=75))
        noisy = target.copy()
        noisy[50:125:2,59] = True
        noisy[34,75:195:2] = True
        image = Image.fromarray(np.uint8(40+170*target)).convert('RGB')
        lines = [dict(p=[60,133], q=[60,36]), dict(p=[61,35], q=[397,35])]
        mask = Image.fromarray(np.uint8(noisy)*255)
        axes = [[1,0,0],[0,1,0],[200,82,1]]
        corrected, evidence = refine(image,mask,lines,axes)
        updated = np.asarray(corrected)>0
        self.assertLess(np.sum(updated!=target),np.sum(noisy!=target))
        self.assertTrue(np.all(updated[35:135,-1]))
        self.assertFalse(updated[80:,205:220].any())
        self.assertEqual(len(evidence['turns']),1)
        np.testing.assert_allclose(evidence['turns'][0]['point'],[60,35])

    def test_explicit_alpha_and_unresolved_frame_preserve_contour(self):
        mask = Image.new('L',(200,100),255)
        axes = [[1,0,0],[0,1,0],[100,50,1]]
        for image, frame in [(Image.new('RGBA',mask.size),axes),
                             (Image.new('RGB',mask.size),[])]:
            result, evidence = refine(image,mask,[],frame)
            self.assertIs(result,mask)
            self.assertEqual(evidence['turns'],[])

    def test_oblique_lines_join_at_projected_angle(self):
        y,x = np.indices((240,400))
        target = (y>=.35*x+25)&(x>=40)&(y<205)&(x<350)
        mask = Image.fromarray(np.uint8(target)*255)
        intrinsic = np.array([[400,0,200],[0,400,120],[0,0,1]])
        right = np.array([1,.35,0])
        upward = np.array([-.4,.4/.35,1])
        axes = [intrinsic@v for v in (right,upward,np.cross(right,upward))]
        # A 90 degree turn in world space does not require a right angle in the image.
        lines = [dict(p=[40,42],q=[40,203]),dict(p=[42,39.7],q=[340,144])]
        _,ev = refine(mask.convert('RGB'),mask,lines,axes)
        self.assertEqual(len(ev['turns']),1)
        np.testing.assert_allclose(ev['turns'][0]['point'],[40,39],atol=.02)
