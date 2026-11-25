"""Undersides can be proposed by a pair of lines on a weakly predicted strip."""
import unittest

import numpy as np
from PIL import Image

from engine.surface_strip_split import refine


class SurfaceStripTest(unittest.TestCase):
    def scene(self):
        width,height=120,80
        rgb=np.full((height,width),170,np.uint8)
        rgb[30:38]=52
        rgb[38:]=126
        normals=np.zeros((height,width,3),np.float32)
        normals[...,2]=-1.
        normals[30:38,:,1]=.34
        depth=np.full(rgb.shape,.85,np.float32)
        depth[30:38]=np.linspace(.83,.35,8)[:,None]
        depth[38:]=.32
        parts=[]
        for ident,(top,bottom) in enumerate(((8,30),(30,68))):
            runs=[[y,8,112] for y in range(top,bottom)]
            # A vertical gap through the scene must remain open after splitting.
            runs=[r for y,left,right in runs for r in [[y,left,88],[y,91,right]]]
            parts.append(dict(id=str(ident),normal_camera=[0.,0.,-1.],
                              support_area=sum(b-a for _,a,b in runs),support=dict(size=[width,height],runs=runs)))
        lines={'test-detector':dict(segments=[[8,30,112,30],[8,38,112,38]])}
        return Image.fromarray(rgb).convert('RGB'),parts,lines,normals,depth

    def pixels(self,parts):
        counts=np.zeros((80,120),np.int32)
        for p in parts:
            owned=0
            for y,left,right in p['support']['runs']:
                counts[y,left:right]+=1
                owned+=right-left
            self.assertEqual(p['support_area'],owned)
        return counts

    def test_two_lines_create_surface_without_a_strong_direction_seed(self):
        image,parts,lines,n,d=self.scene()
        out=refine(image,parts,lines,n,d)
        strips=[p for p in out if 'strip_evidence' in p]
        self.assertTrue(strips)
        self.assertTrue(all(p['strip_evidence']['source']=='two line segments' for p in strips))
        np.testing.assert_array_equal(self.pixels(parts),self.pixels(out))
        self.assertEqual(len(out),len({p['id'] for p in out}))

    def test_same_dark_band_on_flat_wall(self):
        image,parts,lines,n,d=self.scene()
        n[...,1]=0
        self.assertEqual(parts,refine(image,parts,lines,n,d))

    def test_requires_line_evidence(self):
        image,parts,_,n,d=self.scene()
        self.assertEqual(parts,refine(image,parts,{},n,d))
