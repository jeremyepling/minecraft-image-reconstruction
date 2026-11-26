from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import grounded_depth_envelope as model
from engine import grounded_exposed_faces as exposed


class GroundedDepthTests(TestCase):
    def test_unknown_camera_and_no_support_preserve_cells(self):
        photo=Image.new('RGB',(128,96))
        cells={(2,0,4)}
        for camera,parts in [({'P':None},[{}]),({'P':np.eye(3,4)},[])]:
            updated,status=model.recover(photo,parts,camera,cells)
            self.assertEqual(updated,cells)
            self.assertEqual(status['removed'],0)

    def test_missing_or_corrupt_weights_skip_measurement(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'weights.json'
            with patch.object(model,'WEIGHTS',path),patch.object(model,'measure') as measurement:
                for data in (None,b'{}'):
                    if data is not None:
                        path.write_bytes(data)
                    before={(1,3,4)}
                    self.assertEqual(model.recover(None,[],{},before)[0],before)
                measurement.assert_not_called()

    def test_unchanged_build_and_removed_cells_cannot_enter_the_exposure_pass(self):
        prior={(0,0,4),(2,0,4)}
        kept={(0,0,4)}
        camera={'P':np.eye(3,4)}
        with patch.object(exposed.thin_atlas_faces,'recover',return_value=prior) as query:
            self.assertEqual(exposed.recover(None,[],camera,prior,prior),prior)
            query.assert_not_called()
            with patch.object(exposed,'_atlas',return_value=object()):
                self.assertEqual(exposed.recover(None,[],camera,prior,kept),kept)

    def test_exposure_requires_horizontal_pixels_inside_observed_structure(self):
        photo=Image.new('RGB',(8,8))
        camera={'P':np.array([[10,0,4,0],[0,10,4,0],[0,0,1,0]],dtype=float)}
        matrix=camera['P'].copy()
        original={(0,0,5),(1,0,5)}
        pruned={(0,0,5)}
        added={(2,0,5)}
        removed=np.full((8,8),1)
        new=np.full((8,8),1)
        horizontal=np.ones((8,8),int)
        support=[dict(support=dict(size=[8,8],runs=[[0,0,4]]))]
        for axis,parts,expect in [(horizontal,[],set()),(horizontal,support,added),
                                  (horizontal*0,support,set())]:
            with patch.object(exposed,'_frontmost_cells',side_effect=[removed,(new,axis)]):
                got=exposed.newly_exposed(photo,parts,camera,original,pruned,added)
                self.assertEqual(got,expect)
                np.testing.assert_array_equal(matrix,camera['P'])
