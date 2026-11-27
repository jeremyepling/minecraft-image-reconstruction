import unittest

import numpy as np
from PIL import Image, ImageDraw

from engine.exterior_foreground import exterior, misplaced, refine, roof_edits
from engine.plane_consensus import row_runs
from engine.surface_model import intrinsic


class ExteriorForegroundTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (180,120), '#aacdee')
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((0,60,179,119), fill='#8d8f8c')
        draw.rectangle((45,33,87,78), fill='#756451')
        self.camera = dict(P=intrinsic(self.image.size,130) @ np.column_stack((np.eye(3),[0,0,7])))

    def part(self, name, box, normal):
        x,y,r,b = box
        pixels = np.zeros(self.image.size[::-1],bool)
        pixels[y:b,x:r]=True
        return dict(id=name,support=dict(size=self.image.size,runs=row_runs(pixels)),normal_camera=normal)

    def test_exterior_continuity_does_not_require_foreground_connectivity(self):
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((129,50,139,76),fill='#826225')
        background=exterior(self.image)
        self.assertTrue(background[90,118])
        self.assertFalse(background[60,135])
        self.assertFalse(background[50,60])

    def test_staging_floor_patch_and_real_flush_platform_with_a_visible_rim(self):
        floorlike=self.part('floor',(96,80,125,87),[0,1,0])
        rim=self.part('rim',(96,87,125,91),[0,0,1])
        result, events=refine(self.image,[floorlike],self.camera)
        self.assertEqual([],result)
        self.assertEqual(floorlike['id'],events[0]['id'])
        self.assertEqual(([floorlike,rim],[]),refine(self.image,[floorlike,rim],self.camera))
        self.assertEqual(set(),misplaced(self.image,[floorlike],self.camera,{(0,0,0)}))

    def test_sky_color_and_invalid_camera_abstain(self):
        facade=self.part('facade',(100,12,120,32),[0,0,1])
        self.assertEqual(([facade],[]),refine(self.image,[facade],self.camera))
        for invalid in (None,{},dict(P=np.zeros((3,4))),dict(P=[[np.nan]*4]*3)):
            self.assertEqual(([facade],[]),refine(self.image,[facade],invalid))
            self.assertEqual(set(),misplaced(self.image,[facade],invalid,{(0,2,0)}))

    def test_smooth_uniform_frame_has_no_identifiable_backdrop(self):
        self.assertFalse(exterior(Image.new('RGB',(80,60),'#888888')).any())

    def test_apparent_horizontal_faces_seen_from_below_cannot_recover_roof_tops(self):
        upper=self.part('backdrop',(100,76,135,83),[0,1,0])
        lower=self.part('roof',(100,83,135,95),[0,1,0])
        status=dict(self.camera,plane_offsets=[dict(id='roof',axis=1,level=4)])
        self.assertEqual((set(),set()),roof_edits(
            self.image,[upper,lower],[lower],[dict(id='backdrop',trimmed=245,remaining=0)],
            status,{(0,4,0)}))
