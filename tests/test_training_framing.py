import unittest

import numpy as np

from training.framing import measure_framing


class WholeStructureFramingTest(unittest.TestCase):
    def setUp(self):
        matrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,-.05], [0,0,1,0]])
        self.pose = dict(image_size=[100, 50], eye_world=[0, 65, 0],
                         view_rotation_projection=matrix.T.ravel().tolist())
        self.visible = np.zeros((50, 100), np.uint16)
        self.visible[20:25, 50:60] = 1
        self.block = dict(x=0, y=65, z=5, block_id="minecraft:stone")

    def measure(self, *blocks):
        return measure_framing(self.pose, list(blocks), 64, self.visible)

    def test_keeps_complete_build_with_floor_and_air_probes_outside_frame(self):
        floor = dict(self.block, y=64, x=200)
        air = dict(self.block, z=-100, block_id="minecraft:air")
        same_outline = dict(self.block, z=6)
        result = self.measure(self.block, same_outline, floor, air)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["occupied_cells"], 2)
        self.assertEqual(result["projected_margins_px"], [50, 20, 40, 25])
        self.assertEqual(result["visible_margins_px"], [50, 20, 40, 25])

    def test_rejects_unseen_component_and_cube_crossing_camera(self):
        self.assertEqual(self.measure(self.block, dict(self.block, x=7))["reason"], "clipped")
        self.assertEqual(self.measure(self.block, dict(self.block, z=-1))["reason"], "at_or_behind_camera")

    def test_requires_margin_on_each_axis_in_both_measurements(self):
        self.pose["eye_world"][0] = -3.9
        self.assertEqual(self.measure(self.block)["reason"], "insufficient_border")
        self.pose["eye_world"][0] = 0
        self.visible[1, 2] = 1  # exactly 2% left and top
        self.assertTrue(self.measure(self.block)["accepted"])
        self.visible[1, 1] = 1
        self.assertEqual(self.measure(self.block)["reason"], "insufficient_border")

    def test_requires_build_and_visible_pixels(self):
        self.assertEqual(self.measure()["reason"], "empty_build")
        self.visible.fill(0)
        self.assertEqual(self.measure(self.block)["reason"], "no_visible_build_pixels")


if __name__ == "__main__":
    unittest.main()
