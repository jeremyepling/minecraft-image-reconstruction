import unittest

import numpy as np
from PIL import Image

from engine.normal_regions import _rgb_edges, detect


def support(patch):
    width, height = patch["support"]["size"]
    mask = np.zeros((height, width), bool)
    for y, left, right in patch["support"]["runs"]:
        mask[y, left:right] = True
    return mask


class NormalRegionTests(unittest.TestCase):
    def test_full_frame_has_one_plane_with_native_corners(self):
        image = Image.new("RGB", (720, 450), "gray")
        normals = np.zeros((45, 72, 3), np.float32)
        normals[:] = [.03, -.62, -.76]
        patches = detect(image, normals)
        self.assertEqual(len(patches), 1)
        np.testing.assert_allclose(patches[0]["quad"], [[0,0],[720,0],[720,450],[0,450]], atol=2.)
        self.assertTrue(support(patches[0]).all())

    def test_opening_is_excluded_from_pixel_support(self):
        wall = np.zeros((160, 240), bool)
        wall[34:135, 48:190] = True
        wall[55:110, 94:145] = False
        rgb = np.empty((*wall.shape, 3), np.uint8)
        rgb[:] = [160,195,237]
        rgb[wall] = [115,91,72]
        normals = np.zeros(rgb.shape, np.float32)
        normals[wall, 2] = -1.
        patches = detect(Image.fromarray(rgb), normals)
        self.assertTrue(patches)
        masks = [support(p) for p in patches]
        self.assertTrue(any(m[45,90] and m[125,150] for m in masks))
        self.assertFalse(any(m[80,120] for m in masks))
        for patch, mask in zip(patches, masks):
            self.assertEqual(patch["support_area"], int(mask.sum()))

    def test_color_edge_refines_an_offset_normal_boundary(self):
        field = np.zeros((80, 120, 1), np.float32)
        field[21:59, 25:91] = 90.
        quad = np.array([[28.,24.],[88.,24.],[88.,56.],[28.,56.]])
        corrected = _rgb_edges(field, quad)
        np.testing.assert_allclose(corrected, [[25,21],[91,21],[91,59],[25,59]], atol=1.5)

    def test_missing_or_invalid_normals(self):
        image = Image.new("RGB", (80, 60), "gray")
        self.assertEqual(detect(image, np.zeros((60,80,3))), [])
        with self.assertRaises(ValueError):
            detect(image, np.full((60,80,3), np.nan))
