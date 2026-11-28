"""Check that the same site stages a public index and a private single-run index."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmark import floating_review_app as app


class ReviewAppStagingTest(unittest.TestCase):
    def test_shared_landing_keeps_private_entry_out_of_public_bundle(self):
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            front = root / "frontend" / "review-site"
            front.mkdir(parents=True)
            for name in app.SCORED_FILES:
                (front / name).write_text(name, encoding="utf-8")
            market = app.RUN_MANIFESTS / "market" / "manifest.json"
            (root / market).parent.mkdir(parents=True)
            (root / market).write_text(json.dumps(dict(id="market", title="Market")), encoding="utf-8")

            with patch.object(app, "ROOT", root), patch.object(
                app.subprocess, "check_output", return_value=market.as_posix()
            ):
                app.stage_scored_review(front.parent, root / "public")
                app.stage_scored_review(front.parent, root / "protected", single_title="Tower")

            public = json.loads((root / "public/review-home.json").read_text(encoding="utf-8"))
            protected = json.loads((root / "protected/review-home.json").read_text(encoding="utf-8"))
            self.assertIsNone(public["single"])
            self.assertEqual(protected["single"], dict(title="Tower", review="/single/review", scores="/single/scores"))
            self.assertEqual(public["experiments"], protected["experiments"])
            for name in app.SCORED_FILES:
                self.assertEqual((root / "public" / name).read_bytes(), (root / "protected" / name).read_bytes())
            self.assertEqual((root / "public/runs/market/manifest.json").read_bytes(), (root / market).read_bytes())
