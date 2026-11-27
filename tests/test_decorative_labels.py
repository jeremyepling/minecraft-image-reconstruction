import unittest
from zipfile import ZipFile

import numpy as np

from calibration.world import GROUND
from engine.atlas import discover
from training.decorative_labels import VanillaModels, visible_labels


def row(block_id, properties=""):
    return {"block_id": block_id, "state": f"Block{{minecraft:{block_id}}}{properties}"}


class DecorativeLabelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.jar = discover("26.3")
        except FileNotFoundError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_foreground_fence_can_partially_occlude_a_visible_full_cube(self):
        y = GROUND + 1
        pose = {
            "id": "foreground-fence",
            "eye_world": [0.5, y + 0.5, 5.0],
            "image_size": [128, 128],
            "view_rotation_projection": [
                2, 0, 0, 0,  0, 2, 0, 0,  0, 0, -1, -1,  0, 0, 0, 0,
            ],
        }
        states = {
            (0, y, 1): row("spruce_fence", "[east=false,north=false,south=false,waterlogged=false,west=false]"),
            (0, y, 0): row("stone_bricks"),
            (0, y, -1): row("polished_andesite"),
        }
        metadata, ids, depth = visible_labels(pose, states, (0, y, -1), jar=self.jar)
        face_to_state = [""] + [metadata["blocks"][face["block"] - 1]["geometry_kind"]
                                for face in metadata["faces"]]
        self.assertEqual("fence", face_to_state[ids[64, 64]])
        self.assertEqual("full_cube", face_to_state[ids[64, 52]])
        self.assertLess(depth[64, 64], depth[64, 52])
        self.assertTrue(np.array_equal(np.isnan(depth), ids == 0))
        self.assertEqual({"full_cube": 1, "fence": 1}, metadata["block_kinds"])
        self.assertEqual(1, metadata["overlapping_visible_full_blocks"])
        self.assertGreater(metadata["occluded_overlap_pixels"], 0)

    def test_rotation_and_opacity_follow_vanilla_assets(self):
        with ZipFile(self.jar) as archive:
            models = VanillaModels(archive)
            north = models.resolve(row("spruce_trapdoor", "[facing=north,half=bottom,open=true]"))
            south = models.resolve(row("spruce_trapdoor", "[facing=south,half=top,open=true]"))
            south_bottom = models.resolve(row("spruce_trapdoor", "[facing=south,half=bottom,open=true]"))
            wall = models.resolve(row("stone_brick_wall", "[east=low,north=none,south=none,up=false,west=none]"))
            with self.assertRaisesRegex(ValueError, "non-opaque texture"):
                models.resolve(row("oak_trapdoor", "[facing=north,half=bottom,open=true]"))

        self.assertEqual((13 / 16, 1), self.extents(north, 2))
        self.assertEqual((0, 3 / 16), self.extents(south, 2))
        self.assertEqual((0, 3 / 16), self.extents(south_bottom, 2))
        self.assertEqual((0.5, 1), self.extents(wall, 0))

    @staticmethod
    def extents(model, axis):
        values = np.vstack([quad.corners for quad in model.quads])[:, axis]
        return values.min(), values.max()


if __name__ == "__main__":
    unittest.main()
