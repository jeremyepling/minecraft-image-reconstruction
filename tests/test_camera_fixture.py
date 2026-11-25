import unittest

import numpy as np

from benchmark.camera_fixture import LAYOUTS, PALETTES
from benchmark.camera_labels import face_labels, rasterize
from benchmark.fixtures import Building
from calibration.world import GROUND


class CameraFixtureTest(unittest.TestCase):
    def test_palettes_preserve_geometry_and_layouts_never_cross_splits(self):
        seen = set()
        for split, shape in LAYOUTS:
            self.assertNotIn(shape, seen)
            seen.add(shape)
            builds = []
            for materials in PALETTES.values():
                building = Building(0, "test" if split == "holdout" else split, shape.__name__, {})
                shape(building, *materials)
                builds.append(building.cells)
            self.assertEqual(builds[0].keys(), builds[1].keys())
            self.assertNotEqual(builds[0], builds[1])

    def test_crop_counts_against_full_projected_area(self):
        # The x=0 frame edge bisects an 8x6 pixel face.
        quad = np.array([[-4, 2, 1], [-4, 8, 1], [4, 8, 1], [4, 2, 1]], dtype=float)
        _, mask, _, full_area, _ = rasterize(quad, 10, 10)
        self.assertEqual(mask.sum() / full_area, .5)

    def test_nearer_block_hides_separate_wall_and_interior_corners(self):
        y = GROUND + 1
        matrix = np.diag([1., 1., 1., 0.])
        matrix[3, 2] = -1.  # Looking down -Z with a 90-degree vertical FOV.
        pose = dict(id="overlap", image_size=[100, 100], eye_world=[.5, y+.5, 3],
                    view_rotation_projection=matrix.T.ravel().tolist())
        blocks = {(0, y, -2): dict(block_id="tuff", state="tuff"),
                  (0, y, 1): dict(block_id="stone", state="stone")}
        labels = face_labels(pose, blocks, {p: "wall" for p in blocks})
        self.assertEqual(len(labels), 1)
        face = labels[0]
        self.assertEqual(face["xyz"], [0, y, 1])
        self.assertEqual(face["direction"], "south")
        self.assertEqual(face["visible_fraction"], 1.)
        self.assertEqual(face["corner_visible"], [True]*4)
