import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import planarity_crops as crops


class CropRegistrationTest(unittest.TestCase):
    def test_unaligned_native_edges_keep_global_cell_centers(self):
        source = Image.new('RGB', (83, 59))
        h, w = source.height//4, source.width//4
        normals = np.zeros((h, w, 3), np.float32)
        points = np.ones_like(normals)
        def inference(image, model, segmentation, device, keys):
            shape = image.height//4, image.width//4
            return {'relaxed': np.ones(shape, dtype=np.uint16)}, {'inference_s':0., 'segmentation_s':0.}
        with patch('engine.planarity_source.infer_image', side_effect=inference):
            fields, _ = crops.compute_crops(source, None, None, 'cpu', normals, points, 'relaxed')
        self.assertEqual(len(fields['boxes']), 9)
        self.assertEqual(fields['boxes'][-1][2:], source.size)
        for i, labels, _, _ in crops.registered_tiles(fields, complete_only=False):
            box = fields['boxes'][i]
            expected = np.zeros((h, w), dtype=np.uint16)
            expected[box[1]//4:box[3]//4, box[0]//4:box[2]//4] = 1
            np.testing.assert_array_equal(labels, expected)
        # Each synthetic plane hits an artificial tile edge and therefore cannot carve a wall.
        self.assertFalse(any(np.any(labels) for _, labels, _, _ in crops.registered_tiles(fields)))

    def test_insufficient_crop_detail_retains_the_original_field(self):
        fields = dict(image_size=(35,15), boxes=[], tile_labels=[])
        self.assertEqual(list(crops.registered_tiles(fields)), [])
