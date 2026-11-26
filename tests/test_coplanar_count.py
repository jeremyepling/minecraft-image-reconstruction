"""Control for projective scale transfer using a textured donor and a flat recipient."""
import unittest

import numpy as np
from PIL import Image

from benchmark.coplanar_count import predict_frame
from engine.masked_count import count
from engine.surface_sample import sample_surface


class CoplanarCountTest(unittest.TestCase):
    def test_different_material_transfer_and_depth_veto(self):
        image=np.zeros((160,240,3),dtype=np.uint8)
        tile=np.random.default_rng(7).integers(30,190,(16,16,3),dtype=np.uint8)
        image[32:96,16:96]=np.tile(tile,(4,5,1))
        image[48:96,144:176]=220
        photo=Image.fromarray(image)
        parts=[]
        for name,(left,top,width,height) in [('donor',(16,32,80,64)),('target',(144,48,32,48))]:
            parts.append(dict(id=name,normal_camera=[0,0,-1],relative_inverse_depth=.5,
                              quad=[[left,top],[left+width,top],[left+width,top+height],[left,top+height]],
                              support=dict(size=[240,160],runs=[[y,left,left+width] for y in range(top,top+height)])))
        samples=[sample_surface(photo,p) for p in parts]
        base=[count(s) for s in samples]
        self.assertEqual(base[1]['counts'],[None,None])
        result=predict_frame(photo,parts,samples,base)[1]
        np.testing.assert_allclose(result['counts'],[2,3],atol=.05)
        self.assertEqual(result['evidence']['transfer']['recipient_offset'],[8,1])
        parts[1]['relative_inverse_depth']=.7
        self.assertEqual(predict_frame(photo,parts,samples,base)[1]['counts'],[None,None])
