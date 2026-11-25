import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from benchmark.camera_fixture import LAYOUTS, PALETTES, stage
from benchmark.camera_labels import face_labels, rasterize
from benchmark.fixtures import Building
from calibration.world import GROUND, PACK
from engine.nbt import read_nbt


class CameraFixtureTest(unittest.TestCase):
    def test_worldgen_floor_changes_readbacks_but_preserves_structure_and_poses(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jar = root/"meta.jar"
            with ZipFile(jar, "w") as z:
                z.writestr("version.json", json.dumps(dict(id="26.3", world_version=1,
                           pack_version=dict(data_major=1, data_minor=0))))
            stage(root/"default", jar)
            stage(root/"concrete", jar, "light_gray_concrete")
            controls = []
            for name, floor in (("default", "smooth_stone"), ("concrete", "light_gray_concrete")):
                out = root/name
                world = out/"fixture/saves/Palette Lab"
                settings = world/"data/minecraft/world_gen_settings.dat"
                dim = read_nbt(gzip.decompress(settings.read_bytes()))["data"]["dimensions"]["minecraft:overworld"]
                self.assertEqual(dim["generator"]["settings"]["layers"][-1], dict(block="minecraft:"+floor, height=1))
                layout = json.loads((out/"layout.json").read_text())
                self.assertEqual(layout["floor"], floor)
                probes = [b for b in layout["blocks"] if b["role"] == "floor_probe"]
                self.assertEqual(len(probes), 24)
                self.assertTrue(all(b["y"] == GROUND and b["block_id"] == floor for b in probes))
                with ZipFile(world/"datapacks"/PACK) as pack:
                    commands = [pack.read(n) for n in pack.namelist() if n.endswith(".mcfunction")]
                controls.append(((out/"views.json").read_bytes(), (out/"buildings.json").read_bytes(), commands,
                                 [b for b in layout["blocks"] if b["role"] != "floor_probe"]))
            self.assertEqual(controls[0], controls[1])

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
