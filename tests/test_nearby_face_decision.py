from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import nearby_face_decision as rule
from tests.test_visible_horizontal_faces import camera, mask_runs


class NearbyFaceDecisionTest(unittest.TestCase):
    def test_unavailable_weights_and_camera_keep_visible_input(self):
        image=Image.new('RGB',(100,80))
        prior={(0,0,0)}
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'weights.json'
            with patch.object(rule,'MODEL',path),patch.object(rule,'observations',side_effect=AssertionError):
                for raw in (None,b'incomplete',b'{"source":{},"destination":{}}'):
                    if raw is not None:
                        path.write_bytes(raw)
                    self.assertIs(rule.recover(image,[{}],{'P':np.eye(3,4)},prior),prior)
                self.assertIs(rule.recover(image,[{}],{'P':None},prior),prior)

    def test_no_surface_orientation_produces_no_candidate(self):
        size=(100,80)
        image=Image.new('RGB',size)
        P,_=camera(size,np.array([.5,.5,-5.]),[.5,.5,2.])
        mask={'support':mask_runs(np.ones(size[::-1],bool))}
        self.assertIsNone(rule.observations(image,[mask],{'P':P},{(0,0,0)}))

    def test_uncertain_destination_abstains(self):
        data=dict(edges=np.array([[0,1,2]]),features=np.zeros((1,56)))
        model=dict(destination=dict(bias=0.,trees=[]),source=dict(bias=0.,trees=[]),
                   add=.7,drop=.2,margin=.2)
        self.assertEqual(rule.choose(data,model),(set(),set(),set()))

    def test_unavailable_versioned_atlas_retains_original(self):
        size=(100,80)
        image=Image.new('RGB',size)
        P,rot=camera(size,np.array([.5,.5,-5.]),[.5,.5,2.])
        mask=dict(support=mask_runs(np.ones(size[::-1],bool)),normal_camera=rot[:,2])
        prior={(0,0,0)}
        with TemporaryDirectory() as empty:
            self.assertIs(rule.recover(image,[mask],{'P':P},prior,atlas_path=Path(empty)),prior)

    def test_reprojection_competes_with_observed_source(self):
        size=(100,80)
        image=Image.new('RGB',size)
        P,rot=camera(size,np.array([.5,.5,-5.]),[.5,.5,2.])
        mask=dict(support=mask_runs(np.ones(size[::-1],bool)),normal_camera=rot[:,2])
        prior={(0,0,0)}
        data=rule.observations(image,[mask],{'P':P},prior)
        candidates={tuple(data['points'][t]) for _,t,a in data['edges'] if a==2}
        self.assertIn((0,0,1),candidates)
        self.assertIn((0,0,-1),candidates)
        # The proposed face lies behind the visible input cube. Keeping the
        # front cube without a supported switch must reject the blocked face.
        t=next(i for i,p in enumerate(data['points']) if tuple(p)==(0,0,1))
        self.assertEqual(rule.apply(image,[mask],{'P':P},prior,data,(set(),{t},{(t,2)})),prior)


if __name__=='__main__':
    unittest.main()

