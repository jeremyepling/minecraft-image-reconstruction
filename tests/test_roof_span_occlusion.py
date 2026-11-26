from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from engine import roof_span_occlusion as pair


class RoofSpanOcclusionTests(TestCase):
    def test_only_photo_preference_can_displace_observed_prior(self):
        old=(0,1,3)
        proposed=(0,1,2)
        shape=(8,8)
        axes=np.full(shape,2)
        owner=np.zeros(shape,int)
        ctx=SimpleNamespace(eye=np.zeros(3),native=None,shape=shape,atlas=None,anchors={})
        ctx._rays=lambda x,y:np.column_stack(((x+.5)/16.,(y+.5)/16.,np.ones(len(x))))
        ctx.joint=lambda s:(owner,axes,{proposed:0,old:1})
        template=np.random.default_rng(5).uniform(25,150,256).astype('f')
        ctx.maps=template.reshape((1,1,1,256))
        y,x=np.indices(shape)
        bins=pair.texels(ctx,x.ravel(),y.ravel(),np.full(x.size,2),2)%16
        gray=template[bins[:,1]*16+bins[:,0]].reshape(shape)
        ctx.rgb=np.tile(gray[:,:,None],(1,1,3))
        rows={(proposed,2):dict(sprite=0,mip=0,turn=0)}
        with patch.object(pair,'_frontmost_cells',return_value=(owner,axes)):
            for rival,expected in [(.2,{proposed}),(.95,{old})]:
                with patch.object(pair,'_match',return_value={'corr':rival}):
                    self.assertEqual(pair.supported(ctx,{old},{proposed},rows),expected)
