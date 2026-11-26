"""Check ownership when the same projected strip could be a side or a second front."""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine.seam_ownership import textured_side_owners, resolve_side_owners


class ProjectedSeamTests(unittest.TestCase):
    def setUp(self):
        self.left = (0, 0, 0)
        self.right = (1, 0, 0)
        self.cells = [self.left, self.right]
        self.accepted = set(self.cells)
        self.texture = np.random.default_rng(4).uniform(50, 200, (16, 16))

    def resolve(self, right_front, *, strict=False):
        # All three test patches are clear. A darker side repeats the left cube's texture.
        def projected_face(gray, projection, cell, axis, eye, owner, sampler):
            if cell == self.left:
                return (self.texture, np.zeros(32)) if axis == 2 else (.8*self.texture, np.ones(32))
            return right_front, np.ones(32)

        with patch('engine.seam_ownership._face_samples', side_effect=projected_face):
            protected = {self.right} if strict else set()
            textured = textured_side_owners(Image.new('RGB',(64,64)),np.eye(3,4),np.array([5.,1.,5.]),
                                           self.cells,np.ones((16,16)),self.accepted,protected,None)
            return resolve_side_owners(textured, {}, protected)

    def test_two_supported_front_textures_preserve_both_neighboring_cubes(self):
        self.assertEqual(self.resolve(self.texture.copy()),set())

    def test_only_turned_texture_supports_single_cube(self):
        different = np.roll(self.texture,4,axis=1)
        self.assertEqual(self.resolve(different),{self.right})
        self.assertEqual(self.resolve(different,strict=True),set())

    def test_multiple_donors_and_mutual_claims_are_retained(self):
        a,b,c = self.left,self.right,(2,0,0)
        self.assertEqual(resolve_side_owners({a:{b}},{a:{c}},set()),set())
        self.assertEqual(resolve_side_owners({a:{b}},{b:{a}},set()),set())
        self.assertEqual(resolve_side_owners({a:{b}},{b:{c}},set()),{a})
        self.assertEqual(resolve_side_owners({a:{b}},{a:{b}},{a}),set())
