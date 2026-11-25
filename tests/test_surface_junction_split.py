import unittest

import numpy as np
from PIL import Image

from engine.surface_junction_split import _tracks, refine


class SurfaceJunctionSplitTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((80, 128), bool)
        self.mask[24:40, 20:34] = False
        runs = []
        for y, row in enumerate(self.mask):
            changes = np.flatnonzero(np.diff(np.r_[False, row, False]))
            runs.extend([y, int(left), int(right)] for left, right in changes.reshape(-1, 2))
        self.parent = dict(id='wall', support=dict(size=[128, 80], runs=runs), normal_camera=[0., 0., -1.])
        self.depth = np.full(self.mask.shape, .2, np.float32)
        self.depth[:, 64:] = .8
        self.image = Image.fromarray((self.depth * 255).astype(np.uint8)).convert('RGB')
        self.lines = dict(scalelsd=dict(segments=[[64, 0, 64, 38], [64, 42, 64, 80],
                                                    [24, 0, 82, 0], [30, 80, 80, 80]]))

    def test_bridge_fragments_and_preserve_every_original_pixel_and_hole(self):
        children = refine(self.image, [self.parent], self.lines, None, self.depth)
        self.assertEqual(len(children), 2)
        self.assertEqual(len({c['id'] for c in children}), 2)
        assigned = np.zeros(self.mask.shape, np.uint8)
        for child in children:
            for y, left, right in child['support']['runs']:
                assigned[y, left:right] += 1
        np.testing.assert_array_equal(assigned, self.mask)

    def test_texture_edge_without_depth_does_not_cut(self):
        flat_depth = np.full_like(self.depth, .5)
        self.assertEqual(refine(self.image, [self.parent], self.lines, None, flat_depth), [self.parent])

    def test_map_native_endpoints_to_mask_and_require_both_junctions(self):
        scaled = {'segments': (np.asarray(self.lines['scalelsd']['segments']) * [2, 3, 2, 3]).tolist()}
        tracks = _tracks(scaled, [2, 3])
        self.assertEqual(len(tracks), 1)
        endpoints = sorted([tracks[0][0].tolist(), tracks[0][1].tolist()])
        self.assertEqual(endpoints, [[64., 0.], [64., 80.]])
        self.assertEqual(_tracks({'segments': scaled['segments'][:-1]}, [2, 3]), [])


if __name__ == '__main__':
    unittest.main()
