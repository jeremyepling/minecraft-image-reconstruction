"""Recess pitch evidence must distinguish a flush panel from a compressed back face."""
import unittest

import numpy as np

from engine.corner_texture import predict


class CornerTextureTests(unittest.TestCase):
    def check_panel(self,depth):
        rng=np.random.default_rng(601)
        pattern=rng.uniform(45,130,(16,16,3))
        size=40
        vp=np.array([380.,25.]); distance=16.
        low=np.array([80,80]); high=np.array([250,240])
        rim=np.array([low,[high[0],low[1]],high,[low[0],high[1]]])
        xy=np.stack(np.meshgrid(np.arange(320)+.5,np.arange(310)+.5),axis=-1)
        frac=distance/(distance+depth)
        # The reveal is dark and featureless. Only the genuine back carries
        # the repeating motif, including a phase unrelated to the outer wall.
        rect=np.full((310,320,3),100.)
        inside=((xy>=low)&(xy<high)).all(-1)
        rect[inside]=40
        back=vp+(xy-vp)/frac
        visible=inside&((back>=low)&(back<high)).all(-1)
        ij=(np.floor((back+7)/size*16).astype(int))%16
        rect[visible]=pattern[ij[visible,1],ij[visible,0]]
        result=predict(rect,rim,vp,distance,size,[1,1])
        self.assertEqual(result['depth'],depth)

    def test_flush_insert_with_its_own_material(self):
        self.check_panel(0)

    def test_projected_two_block_recess(self):
        self.check_panel(2)

    def test_no_texture_abstains(self):
        rect=np.full((300,300,3),95.)
        q=np.array([[80,80],[220,80],[220,220],[80,220]])
        result=predict(rect,q,[420,20],16.,40,[1,1])
        self.assertIsNone(result['depth'])
