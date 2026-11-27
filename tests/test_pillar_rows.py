from types import SimpleNamespace
import unittest

from PIL import Image

from engine import pillar_rows
from engine.atlas_phase import ATLAS_SHA256
from engine.pillar_seams import split_pillars


class PillarRowsTests(unittest.TestCase):
    def test_missing_or_mismatched_atlas_preserves_input(self):
        prior={(1,0,2)}
        for atlas in (None,SimpleNamespace(provenance={'atlas_sha256':'other'})):
            placed,_=pillar_rows.recover(None,[{}],{'P':True},prior,atlas)
            self.assertEqual(placed,prior)

    def test_missing_camera_preserves_input(self):
        prior={(0,1,2)}
        atlas=SimpleNamespace(provenance={'atlas_sha256':ATLAS_SHA256})
        placed,_=pillar_rows.recover(None,[{}],{},prior,atlas)
        self.assertEqual(placed,prior)

    def test_empty_and_absent_masks_do_not_create_a_pillar(self):
        photo=Image.new('RGB',(80,80))
        camera={'P':[[70,0,40,0],[0,70,40,0],[0,0,1,1]]}
        parts=[{'id':'empty','normal_camera':[1,0,0],'support':{'runs':[],'size':[80,80]}},
               {'id':'absent','normal_camera':[1,0,0]}]
        self.assertEqual(split_pillars(photo,parts,camera),(parts,[]))
