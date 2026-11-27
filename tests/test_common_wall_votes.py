import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from engine import smooth_atlas_outline as outline, smooth_extent_placement as placer
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


def part(name, mask, direction):
    return dict(id=name, normal_camera=direction, support=dict(size=list(mask.shape[::-1]),runs=row_runs(mask)))


class SmoothWallVotes(unittest.TestCase):
    def setUp(self):
        wall=np.zeros((90,160),bool)
        wall[24:56,28:132]=True
        x=np.indices(wall.shape)[1]
        self.parts=[part('left',wall&(x<71),[0,0,-1]),part('right',wall&(x>=102),[0,0,-1])]
        self.wall=wall
        rgb=np.zeros((*wall.shape,3),np.uint8)
        rgb[:]=[140,175,230]
        rgb[wall]=[155,157,160]
        self.image=Image.fromarray(rgb)

    def test_only_unknown_wall_interior_gains_support(self):
        self.assertEqual(placer.proposals(self.image,self.parts),[])
        offered=placer.proposals(self.image,self.parts,common_wall=True)
        self.assertEqual(len(offered),1)
        self.assertEqual((offered[0]['surface'],offered[0]['pixels']),('left',31*32))
        observed=native_mask(self.parts[0],self.image.size)|native_mask(self.parts[1],self.image.size)
        additions=np.zeros_like(observed)
        for y,l,r in offered[0]['runs']:
            additions[y,l:r]=True
        np.testing.assert_array_equal(additions,self.wall&~observed)
        self.assertFalse((additions&observed).any())

    def test_different_axis_and_horizontal_patches_abstain(self):
        parts=[self.parts[0],dict(self.parts[1],normal_camera=[1,0,0])]
        self.assertEqual(placer.proposals(self.image,parts,common_wall=True),[])
        floor=[dict(p,normal_camera=[0,-1,0]) for p in self.parts]
        self.assertEqual(placer.proposals(self.image,floor,common_wall=True),[])

    def test_missing_atlas_and_prior_owners_are_preserved(self):
        loader=Mock(return_value=None)
        parts,status,regions=outline.with_common_wall(self.image,self.parts,'26.3',atlas_loader=loader)
        self.assertIs(parts,self.parts)
        self.assertEqual(regions,[])
        self.assertEqual(status['status'],'atlas unavailable')
        pending=dict(surface='new host',runs=[(30,68,76)],pixels=8)
        existing=dict(surface='initial host',runs=[(30,70,73)],pixels=3)
        after=outline.extra_interior(self.image,[pending],[existing])
        self.assertEqual(after[0]['runs'],[(30,68,70),(30,73,76)])
        self.assertEqual(after[0]['pixels'],5)
