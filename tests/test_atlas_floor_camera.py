import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from engine import camera_atlas_floor as camera
from engine.camera_texel_floor import _ground, _pencils


class AtlasFloorCameraTests(unittest.TestCase):
    def test_unverified_floor_patterns_abstain(self):
        y, x = np.mgrid[:380, :512]
        rng = np.random.default_rng(31)
        noise = rng.integers(124, 129, x.shape, dtype=np.uint8)
        patterns = [
            np.zeros_like(x, dtype=np.uint8),
            noise,
            126 + x % 8,
            126 + ((x // 6 + y // 6) % 2),
            np.where((y > 325) & (x < 192), noise, 126),
        ]
        for values in patterns:
            with self.subTest(std=values.std()):
                photo = Image.fromarray(values.astype(np.uint8)).convert("RGB")
                self.assertIsNone(camera.predict(photo)["P"])
        self.assertIsNone(camera.predict(Image.new("RGB", (32, 48)))["P"])

    def test_sparse_rows_and_degenerate_perspective_abstain(self):
        edges = np.zeros((150, 512))
        edges[16:24, ::7] = 1
        self.assertEqual(_pencils(edges, np.arange(150) + 410., 550), [])
        for matrix in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.diag([1e-19,1.,1.])):
            self.assertIsNone(_ground(matrix, 512, 384))

    def test_missing_assets_can_be_repaired_and_invalid_candidates_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken, repaired = root/"broken", root/"repaired"
            broken.mkdir()
            repaired.mkdir()
            for folder in [broken, repaired]:
                (folder/"atlas.json").write_text("{}")
                (folder/"atlas.png").write_bytes(b"test")
            with patch.object(camera, "prepared_atlas_paths") as paths, patch.object(camera, "load_atlas") as load:
                paths.return_value = []
                self.assertIsNone(camera._atlas("26.3"))
                paths.return_value = [broken, repaired]
                for exc in [TypeError, IndexError, EOFError, ValueError, OSError]:
                    camera._verified_atlas.cache_clear()
                    sentinel = object()
                    load.side_effect = [exc(), sentinel]
                    self.assertIs(camera._atlas("26.3"), sentinel)
                # Same process picks up an updated local atlas.
                paths.return_value = [repaired]
                (repaired/"atlas.json").write_text('{"updated": true}')
                load.side_effect = [EOFError()]
                self.assertIsNone(camera._atlas("26.3"))
                # An explicit broken path never falls through to a discovered one.
                paths.reset_mock()
                self.assertIsNone(camera._atlas("26.3", root/"missing"))
                paths.assert_not_called()

    def test_unbenchmarked_version_abstains_without_loading(self):
        with patch.object(camera, "_atlas", side_effect=AssertionError):
            self.assertIsNone(camera.predict(Image.new("RGB", (800, 600)), "unknown")["P"])

