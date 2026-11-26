import unittest

import numpy as np

from engine import roof_height_rim as rim, roof_height_rim_band as band
from engine import roof_height_rim_exchange as exchange


class RoofHeightRimTests(unittest.TestCase):
    def test_single_horizontal_grid_plane_wins_over_neighboring_heights(self):
        eye=np.array([4.25,8.3,-10.5])
        optics=np.array([[95.,0,64],[0,-95,48],[0,0,1]])
        inv=np.linalg.inv(optics)
        grid=np.arange(.5,128,2)
        xx,yy=np.meshgrid(grid,grid)
        rays=rim.rays(xx.ravel(),yy.ravel(),inv)
        t=(4-eye[1])/rays[:,1]
        xz=eye[[0,2]]+t[:,None]*rays[:,[0,2]]
        mask=((t>0)&(xz[:,0]>=0)&(xz[:,0]<6)&(xz[:,1]>=1)&(xz[:,1]<7)).reshape(64,64)
        e=np.linspace(.05,.95,90)
        border=np.concatenate([np.column_stack((e*6,np.full(90,4),np.full(90,z))) for z in (1,7)]+
                              [np.column_stack((np.full(90,x),np.full(90,4),1+6*e)) for x in (0,6)])
        edge_rays=(border-eye)/(border[:,[2]]-eye[2])
        supports={(a,level,b) for a,b in ((0,1),(5,1),(0,6),(5,6)) for level in range(3)}
        tips={3:{(0,1),(5,1),(0,6),(5,6)}}
        ranked=rim.hypotheses(mask,edge_rays,inv,eye,tips,supports,2)
        self.assertEqual((4,3),ranked[0][1:3])
        self.assertGreater(ranked[0][0]-ranked[1][0],.08)

    def test_absent_camera_keeps_unknown_cells_out(self):
        prior={(1,3,2)}
        for method in (rim.recover,exchange.recover,band.recover):
            result,record=method(None,[{}],None,{'P':None},prior)
            self.assertEqual(result,prior)
            self.assertEqual(record,{'height_decisions':[],'primary_height':None})

    def test_disconnected_band_cannot_support_a_horizontal_surface(self):
        a=np.zeros((16,16),bool)
        a[3:7,3:8]=True
        b=np.zeros_like(a)
        b[11:14,3:8]=True
        self.assertEqual(band.adjoining(a,b),0)
        b[7,3:8]=True
        self.assertEqual(band.adjoining(a,b),5)
