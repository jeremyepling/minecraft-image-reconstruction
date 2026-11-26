"""Native projection, occlusion and material checks for dataset labels."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.visible_labels import write_labels

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmark/data/camera-gray-concrete-v1"


class VisibleLabelsTest(unittest.TestCase):
    def test_native_ownership_and_archive_agreement(self):
        stem = "dev-wing-masonry-low-left"
        pose = json.loads((FIXTURE / "captures" / f"{stem}.json").read_text())
        rows = json.loads((FIXTURE / "captures" / f"{stem}-blocks.json").read_text())
        states = {tuple(row[k] for k in "xyz"): row for row in rows if row["y"] > 64}
        reference = json.loads((FIXTURE / "labels" / f"{stem}.json").read_text())["faces"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = write_labels(pose, states, (0, 65, 0), output)
            data = json.loads((output / summary["labels"]).read_text())
            with np.load(output / summary["raster"]) as raster:
                ids, depth = raster["face_id"], raster["depth_z"]
        expected = {(tuple(f["xyz"]), f["direction"]): f for f in reference}
        self.assertEqual(len(expected), len(data["faces"]))
        self.assertTrue(np.array_equal(ids == 0, np.isnan(depth)))
        counts = np.bincount(ids.ravel())
        for face in data["faces"]:
            ref = expected[tuple(face["xyz_world"]), face["direction"]]
            self.assertEqual(ref["visible_pixels"], counts[face["id"]])
            self.assertEqual(ref["corner_visible"], face["corner_visible"])
            np.testing.assert_allclose(face["corners_px"], ref["image_corners"], atol=1e-7)

        # Back-project deterministic foreground samples using stored intrinsics, transform
        # and metric depth. Each point must lie on the labeled block's unit face.
        ys, xs = np.nonzero(ids)
        ys, xs = ys[::499], xs[::499]
        uv1 = np.column_stack((xs + .5, ys + .5, np.ones_like(xs)))
        cam = data["camera"]
        camera_points = (uv1 @ np.linalg.inv(cam["K"]).T) * depth[ys, xs, None]
        world = camera_points @ np.asarray(cam["world_to_camera_rotation"]) + cam["eye_world"]
        for p, face_id in zip(world, ids[ys, xs]):
            face = data["faces"][face_id-1]
            block_min = np.array(face["xyz_world"])
            self.assertTrue(np.all(p >= block_min - .0001) and np.all(p <= block_min + 1.0001))
            normal = np.array(face["normal_world"])
            np.testing.assert_allclose(p @ normal, block_min @ normal + int(1 in normal), atol=.0001)

    def test_aligned_cubes_drop_hidden_cell_and_retain_background(self):
        width, height = 80, 60
        # Native clip x grows right, y grows up, camera forward goes toward positive Z.
        matrix = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,-.05], [0,0,1,0]], dtype=float)
        pose = dict(id="two-cubes", image_size=[width,height], eye_world=[0.5,65.5,-2.0],
                    view_rotation_projection=matrix.T.reshape(-1).tolist())
        states = {(0,65,z): {"block_id":"stone","state":"Block{minecraft:stone}"} for z in [0,2]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_labels(pose, states, (0,65,0), output)
            data = json.loads((output / "labels.json").read_text())
            self.assertEqual([b["xyz_local"] for b in data["blocks"]], [[0,0,0]])
            self.assertEqual([f["direction"] for f in data["faces"]], ["north"])
            with np.load(output / "labels.npz") as raster:
                self.assertEqual(raster["face_id"][30,40], 1)
                self.assertEqual(raster["depth_z"][30,40], 2.0)
                self.assertTrue(np.isnan(raster["depth_z"][0,0]))
