import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine.plane_consensus import row_runs
from engine.smooth_extent_placement import proposals, recover, supported_additions


class SmoothExtentPlacementTest(unittest.TestCase):
    def test_patch_must_continue_an_observed_material_and_stay_clear_of_the_exterior(self):
        rgb=np.full((96,140,3),200,np.uint8)
        rgb[15:85,20:120]=(42,68,80)
        m=np.zeros(rgb.shape[:2],bool)
        m[15:85,20:120]=True
        m[38:70,35:95]=False
        part=dict(id='wall',support=dict(size=[140,96],runs=row_runs(m)))
        hits=proposals(Image.fromarray(rgb),[part])
        self.assertEqual([(r['surface'],r['pixels']) for r in hits],[('wall',32*60)])
        self.assertEqual(proposals(Image.new('RGB',(140,96),(42,68,80)),[part]),[])

    def test_invalid_camera_abstains_before_proposal_generation(self):
        with patch('engine.smooth_extent_placement.proposals',side_effect=AssertionError('unexpected proposal')):
            for camera in (None,dict(P=None),dict(P=np.zeros((3,4)))):
                self.assertEqual(recover(None,'26.3',[],{(0,0,0)},camera,None),(set(),[]))

    def test_frontmost_samples_require_majority_and_correct_face_axis(self):
        seed=(0,0,0)
        added=(1,0,0)
        image=Image.new('RGB',(20,10))
        parts=[dict(id='wall',normal_camera=[1.,0.,0.])]
        regions=[dict(surface='wall',runs=[(y,0,20) for y in (0,2)])]
        from types import SimpleNamespace
        scene=SimpleNamespace(shape=(5,10),cells=[seed,added],axes=np.zeros(50,int),
                              owned={0:np.arange(30,50),1:np.arange(30)})
        with patch('engine.projected_appearance.Scene',return_value=scene),patch(
                'engine.smooth_extent_placement.orientation',return_value=(None,np.eye(3))):
            self.assertEqual(supported_additions(image,parts,{}, {seed},{seed,added},regions),{added})
            scene.axes[:5]=1
            self.assertEqual(supported_additions(image,parts,{}, {seed},{seed,added},regions),set())
