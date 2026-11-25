import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine.mask_first import regions


def support(region):
    width, height = region['support']['size']
    mask = np.zeros((height, width), bool)
    for y, left, right in region['support']['runs']:
        mask[y, left:right] = True
    return mask


class MaskFirstTests(unittest.TestCase):
    def test_hole_and_disconnected_pillar_survive_without_an_outline(self):
        wall = np.zeros((100, 180), bool)
        wall[10:92, 24:110] = True
        wall[29:76, 50:91] = False
        wall[27:92, 130:149] = True
        rgb = np.full((*wall.shape, 3), [155, 198, 237], dtype=np.uint8)
        rgb[wall] = [115, 91, 72]
        normals = np.zeros(rgb.shape, dtype=np.float32)
        normals[wall, 2] = -1
        with patch('engine.mask_first._patch', return_value=None):
            surfaces = regions(Image.fromarray(rgb), normals, wall.astype(np.float32))
        self.assertEqual(len(surfaces), 2)
        masks = [support(surface) for surface in surfaces]
        self.assertTrue(masks[0][20, 45] and masks[0][86, 75])
        self.assertTrue(masks[1][40, 140])
        self.assertFalse(any(mask[40, 75] or mask[55, 120] for mask in masks))
        self.assertTrue(all(surface['geometry_status'] == 'mask only' for surface in surfaces))

    def test_numerical_depth_separates_parallel_planes_despite_identical_rgb(self):
        image = Image.new('RGB', (140, 90), 'gray')
        normals = np.zeros((90, 140, 3), np.float32)
        normals[:, :, 2] = -1
        depth = np.ones(normals.shape[:2], np.float32)
        depth[:, 70:] = 4
        old = regions(image, normals, depth)
        transformed = regions(image, normals, 8 + depth * 12)
        self.assertEqual(len(old), 2)
        self.assertEqual([p['support'] for p in old], [p['support'] for p in transformed])
        self.assertTrue(all(p['split_by_depth'] for p in old))
        masks = [support(p) for p in old]
        self.assertTrue(np.logical_xor(*masks).all())
        self.assertFalse(any(mask[40, 28] and mask[40, 118] for mask in masks))
        ramp = np.broadcast_to(np.arange(140, dtype=np.float32), depth.shape)
        self.assertEqual(len(regions(image, normals, ramp)), 1)

    def test_absent_or_invalid_directions(self):
        image = Image.new('RGB', (80, 60), 'gray')
        normals = np.zeros((60, 80, 3), np.float32)
        depth = np.zeros(normals.shape[:2], np.float32)
        self.assertEqual(regions(image, normals, depth), [])
        with self.assertRaises(ValueError):
            regions(image, normals + np.nan, depth)
