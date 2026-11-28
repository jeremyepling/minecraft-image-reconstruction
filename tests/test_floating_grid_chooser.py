"""Artificial packets exercise provenance and selection without claiming a measured build gain."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmark import floating_grid_chooser as pilot


class ChooserTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.source, self.dev, self.test = (self.folder / f"{x}.json" for x in ("source", "dev", "test"))
        self.packet = dict(cohort="synthetic", revision="fixture", views=[])
        self.labels = {"development": {}, "holdout": {}}
        for idx, (family, split) in enumerate((("A", "development"), ("B", "development"), ("C", "holdout"))):
            key = str(idx)
            options = {"keep": dict(edge_alignment=.8, atlas_agreement=.2, visible_support=.5),
                       "refit": dict(edge_alignment=.4, atlas_agreement=.8, visible_support=.6)}
            options = {k: dict(grid=self.file(f"grid-{idx}-{k}"), features=v) for k, v in options.items()}
            surroundings = self.file(f"fixed-{idx}")
            view = dict(id=key, family=family, split=split, decision="wall-A", incumbent="keep",
                        screenshot=self.file(f"image-{idx}"), surroundings=surroundings, options=options)
            self.packet["views"].append(view)
            scores = {}
            for name, hits in (("keep", 3), ("refit", 5)):
                count = dict(correct=hits, missed=8-hits, extras=2)
                scores[name] = dict(surroundings_sha256=surroundings["sha256"], grid_sha256=options[name]["grid"]["sha256"],
                                    export=self.file(f"{idx}-{name}"),
                                    scores={m: count.copy() for m in pilot.METRICS})
            self.labels[split][key] = scores
        self.save()

    def file(self, name):
        path = self.folder / name
        path.write_bytes(name.encode())
        return dict(path=name, sha256=pilot.digest(path))

    def save(self):
        self.source.write_text(json.dumps(self.packet))
        for path, split in ((self.dev, "development"), (self.test, "holdout")):
            path.write_text(json.dumps(dict(source_sha256=pilot.digest(self.source), evaluator=self.file("scorer"),
                                            views={k: dict(visible_reference=self.file(f"ref-{k}"), options=v)
                                                   for k, v in self.labels[split].items()})))

    def run_pilot(self):
        return pilot.run(self.source, self.dev, self.test)

    def test_rules_holdout_and_scores(self):
        result = self.run_pilot()
        self.assertEqual("evaluated", result["status"])
        self.assertEqual("atlas_agreement", result["model"]["rank"])
        self.assertEqual(1, result["holdout"]["development_rule"]["wins"])
        self.assertEqual(0, result["holdout"]["edge_alignment"]["wins"])
        self.assertEqual(2, result["holdout"]["development_rule"]["delta_from_incumbent"]["scores_facade"]["correct"])
        labels = self.labels["holdout"]["2"]
        labels["keep"]["scores"], labels["refit"]["scores"] = labels["refit"]["scores"], labels["keep"]["scores"]
        self.save()
        reversed_truth = self.run_pilot()
        self.assertEqual(result["model"], reversed_truth["model"])
        self.assertEqual("no_holdout_headroom", reversed_truth["status"])
        self.assertEqual(1, reversed_truth["holdout"]["development_rule"]["losses"])

    def test_fitted_gate_abstains_on_a_conflicting_view(self):
        # Both development frames prefer the alternate by edge strength. Only the
        # first has corroborating atlas support and a better whole export.
        for v in self.packet["views"]:
            v["options"]["keep"]["features"] = dict(edge_alignment=.2, atlas_agreement=.4, visible_support=.5)
            v["options"]["refit"]["features"] = dict(edge_alignment=.8, atlas_agreement=.4, visible_support=.5)
        self.packet["views"][1]["options"]["refit"]["features"]["atlas_agreement"] = .2
        scores = self.labels["development"]["1"]
        scores["keep"]["scores"], scores["refit"]["scores"] = scores["refit"]["scores"], scores["keep"]["scores"]
        self.save()
        result = self.run_pilot()
        self.assertEqual({"rank": "edge_alignment", "gate": ["atlas_agreement", 0]}, result["model"])
        self.assertEqual(0, result["development"]["development_rule"]["losses"])
        self.assertEqual(1, result["development"]["edge_alignment"]["losses"])

    def test_mismatched_grid_and_scorer_are_rejected(self):
        self.labels["holdout"]["2"]["keep"]["grid_sha256"] = "wrong"
        self.save()
        with self.assertRaisesRegex(ValueError, "different grid"):
            self.run_pilot()
        self.labels["holdout"]["2"]["keep"]["grid_sha256"] = self.packet["views"][2]["options"]["keep"]["grid"]["sha256"]
        self.save()
        receipt = json.loads(self.test.read_text())
        receipt["evaluator"] = self.file("other-scorer")
        self.test.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "Evaluator differs"):
            self.run_pilot()

    def test_source_ties_keep_incumbent(self):
        options = self.packet["views"][0]["options"]
        options["third"] = options["refit"]
        self.assertEqual("keep", pilot.choose(options, "keep", dict(rank="atlas_agreement", gate=None)))

    def test_no_dev_headroom_keeps_holdout_sealed(self):
        for view in self.labels["development"].values():
            view["refit"]["scores"] = view["keep"]["scores"]
        self.save()
        self.test.unlink()
        self.assertEqual("no_development_headroom", self.run_pilot()["status"])

    def test_fewer_than_two_development_families(self):
        self.packet["views"][1]["family"] = "A"
        self.save()
        self.assertEqual("insufficient_building_families", self.run_pilot()["status"])

    def test_family_overlap_and_leaked_features(self):
        self.packet["views"][2]["family"] = "A"
        self.save()
        with self.assertRaisesRegex(ValueError, "family crosses"):
            self.run_pilot()
        self.packet["views"][2]["family"] = "C"
        self.packet["views"][1]["options"]["refit"]["features"]["truth_overlap"] = .99
        self.save()
        with self.assertRaisesRegex(ValueError, "Expected fields"):
            self.run_pilot()

    def test_receipt_requires_paired_surroundings_and_export_bytes(self):
        self.labels["holdout"]["2"]["keep"]["surroundings_sha256"] = "different"
        self.save()
        with self.assertRaisesRegex(ValueError, "Changed surroundings"):
            self.run_pilot()
        self.labels["holdout"]["2"]["keep"]["surroundings_sha256"] = self.packet["views"][2]["surroundings"]["sha256"]
        self.save()
        (self.folder / "2-keep").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            self.run_pilot()


if __name__ == "__main__":
    unittest.main()
