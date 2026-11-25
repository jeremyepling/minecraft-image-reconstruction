"""Depth requires a projected return and a second rim inside the first one."""
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw
from engine.depth_layers import fit_reveal, floor_calibration, refine_depth


class RevealTests(unittest.TestCase):
    def test_front_marking_stays_flush(self):
        q=np.array([[40,60],[100,60],[100,120],[40,120]],float)
        gray=np.full((180,180),130.)
        gray[60:120,40:100]=50
        result=fit_reveal(gray,q,np.array([150.,30.]),20,200,[0]*9)
        self.assertIsNone(result['depth'])

    def test_visible_corner_and_second_rim(self):
        q=np.array([[40,60],[100,60],[100,120],[40,120]],float)
        vp=np.array([150.,30.])
        b=vp+(q-vp)/1.2
        image=Image.new('L',(180,180),130)
        draw=ImageDraw.Draw(image)
        draw.rectangle((40,60,100,120),fill=80)
        draw.polygon(list(map(tuple,[q[0],b[0],b[3],q[3]])),fill=30)
        draw.polygon(list(map(tuple,[q[3],b[3],b[2],q[2]])),fill=110)
        result=fit_reveal(np.asarray(image,float),q,vp,20,200,[0]*9)
        self.assertEqual(result['depth'],2)

    def test_unobservable_floor(self):
        self.assertIsNone(floor_calibration(np.full((180,200,3),127,np.uint8)))

    def test_recess_export_removes_foreground_voxel(self):
        quad=np.array([[30,50],[70,50],[70,90],[30,90]],float)
        vp=np.array([100.,20.])
        back=vp+(quad-vp)/1.2
        record=dict(status='measured',depth=2,rim=quad.tolist(),back_rim=back.tolist())
        audit=dict(recesses=[record],ledges=[],calibration=dict(horizon_y=20.))
        cell=dict(ij=[0,0],quad=quad.tolist(),xyz=[0,0,0],plane='test',facing='south')
        with patch('engine.depth_layers.inspect_depth',return_value=audit):
            _,cells=refine_depth(np.zeros((150,200,3),np.uint8),[dict(name='test')],[cell])
        self.assertIn([0,0,-2],[c['xyz'] for c in cells])
        self.assertNotIn([0,0,0],[c['xyz'] for c in cells])
        self.assertEqual(cell['xyz'],[0,0,0])

    def test_measured_door_uses_one_grid_and_closed_jambs(self):
        grid=np.array([[10,0,30],[0,10,40],[0,0,1]],float)
        q=np.array([[50,50],[70,50],[70,80],[50,80]],float)
        vp=np.array([100.,20.])
        back=vp+(q-vp)/1.3
        audit=dict(calibration=dict(horizon_y=20.),ledges=[],
                   recesses=[dict(status='measured',depth=3,rim=q.tolist(),back_rim=back.tolist())],
                   front_lattice=dict(grid_to_image=grid.tolist(),columns=6,rows=4,omitted_ij=[]))
        initial=dict(plane='wall',quad=q.tolist(),xyz=[0,0,0],ij=[0,0],facing='south')
        with patch('engine.depth_layers.inspect_depth',return_value=audit):
            planes,cells=refine_depth(np.zeros((150,200,3),np.uint8),[dict(name='wall')],[initial])
        self.assertEqual((planes[0]['columns'],planes[0]['rows']),(6,4))
        placed={tuple(c['xyz']):c for c in cells}
        self.assertEqual(len(placed),len(cells))
        for z in (0,-1,-2):
            for y in range(3):
                self.assertIn((1,y,z),placed)
                self.assertIn((4,y,z),placed)
                self.assertNotIn((2,y,z),placed)
        self.assertIn((2,0,-3),placed)
        self.assertTrue(all(y>=0 for x,y,z in placed))
        self.assertEqual(placed[(4,1,-1)]['sample_source_xyz'],[4,1,0])
