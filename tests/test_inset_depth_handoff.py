import unittest

from engine.inset_depth_handoff import levels, measurement


class InsetDepthHandoffTest(unittest.TestCase):
    def setUp(self):
        self.camera=dict(P=[[1,0,0,0],[0,1,0,-3],[0,0,1,8]])
        self.parts=[dict(id=k) for k in ('wall','jamb','sill','back','unrelated')]
        fit=dict(host='wall',side='jamb',sill='sill',level=1,depth=2,rim=[0,4,2,5])
        self.parts[3]['inset_back_plane']=measurement(self.camera,2,fit)

    def test_only_measured_back_uses_absolute_plane(self):
        self.assertEqual(levels(self.parts,self.camera),{'back':(2,3)})
        self.assertEqual(levels([dict(id='back')],self.camera),{})

    def test_stale_camera_missing_return_and_outward_hint_fall_back(self):
        self.assertEqual(levels(self.parts,dict(P=[[1,0,0,1],[0,1,0,-3],[0,0,1,8]])),{})
        self.assertEqual(levels(self.parts[1:],self.camera),{})
        self.parts[3]['inset_back_plane']['signed_depth']=-2
        self.assertEqual(levels(self.parts,self.camera),{})
