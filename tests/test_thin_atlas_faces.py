import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import thin_atlas_faces as strips


class ThinAtlasFacesTest(unittest.TestCase):
    def setUp(self):
        self.photo=Image.new('RGB',(32,24))
        self.parts=[{'support':{'size':[32,24], 'runs':[[5,1,28]]}}]
        self.prior={(1,1,1)}
        self.cam={'P':np.eye(3,4)}
        self.atlas=SimpleNamespace(provenance={'version':'26.3','atlas_sha256':strips.ATLAS_SHA256})

    def test_uncertain_inputs_abstain_before_rasterization(self):
        bad=SimpleNamespace(provenance={'version':'26.3','atlas_sha256':'other'})
        with patch.object(strips,'StripEvidence') as compute:
            for atlas,camera,parts in [(None,self.cam,self.parts),(bad,self.cam,self.parts),
                                       (self.atlas,{'P':None},self.parts),(self.atlas,self.cam,[])]:
                self.assertEqual(strips.recover(self.photo,parts,camera,self.prior,atlas),self.prior)
            compute.assert_not_called()

    def test_missing_damaged_wrong_length_and_nonfinite_model_preserve_build(self):
        good=json.loads(strips.WEIGHTS.read_text())
        choices=[None,'{bad',json.dumps(dict(good,weights=[1])),json.dumps(dict(good,bias=float('nan'))),
                 json.dumps(dict(good,scale=[0]*len(strips.FEATURES))),
                 json.dumps(dict(good,mean=[float('inf')]*len(strips.FEATURES)))]
        with TemporaryDirectory() as temp,patch.object(strips,'StripEvidence') as compute:
            for i,content in enumerate(choices):
                path=Path(temp)/str(i)
                if content is not None:
                    path.write_text(content)
                with patch.object(strips,'WEIGHTS',path):
                    self.assertEqual(strips.recover(self.photo,self.parts,self.cam,self.prior,self.atlas),self.prior)
            compute.assert_not_called()

    def test_visible_strip_recovers_phase_on_independent_texel_halves(self):
        rng=np.random.default_rng(188)
        tiles=rng.integers(20,225,(32,16,16,3),dtype=np.uint8)
        atlas=SimpleNamespace(tiles=tiles,means=tiles.mean((1,2)))
        atlas.means[9]=np.array([128,128,128])
        uv=np.array([(x,y) for x in range(16) for y in (2,5,6,11)])
        colors=tiles[9,uv[:,1],uv[:,0],:]
        match=strips._match(uv,colors,atlas,strips._mips(atlas),[9,10])
        self.assertEqual(match['sprite'],9)
        self.assertGreater(match['split'],.99)
        self.assertGreater(match['phase'],.65)
        self.assertIsNone(strips._match(uv,np.zeros_like(colors),atlas,strips._mips(atlas)))


if __name__=='__main__':
    unittest.main()
