"""Compose decoded scores from the frozen decisions, verifying the exact selected camera."""

from benchmark.camera_model_controls import summary
from benchmark.expanded_baseline import COHORTS, SOURCE, WORKTREES, digest, frames, read, save
from benchmark.expanded_scores import score_build
from benchmark.camera_fit_confidence import AUDIT, CONTROLS, NATIVE, REVIEW, ROOT


def main():
    base = read(CONTROLS)
    inventory = {r["id"]: r for r in read(CONTROLS.parent / "inventory.json.gz")}
    frame_list = read(ROOT / "benchmark/reviews/camera-model-controls-01/frames.json")
    expanded = {e["id"]: e for e in frames(SOURCE, COHORTS)}
    attempts = {}
    for split in ("train", "validation", "test"):
        for row in read(REVIEW / f"{split}-decisions.json.gz"):
            attempts[row["id"]] = row
    incumbent, challenger, new = [], [], []
    forced = WORKTREES / "camera-recovery-next/runs/camera-recovery-next-01/rejected-options"
    for control in base:
        key = control["id"]
        decision = attempts.get(key)
        source = control
        if decision and decision["incumbent"]:
            score = NATIVE / "scores" / (key + ".json")
            source = read(score)
            assert digest(NATIVE / "builds" / control["dataset"] / key / "reconstruction.zip") == source["pack_sha256"]
        incumbent.append(source)
        if decision and decision["selected"]:
            option = decision["selected"]
            receipt = read(NATIVE / option["arm"] / "metric" / (key + ".json"))
            camera = receipt["camera"]["evidence"]["refined_proposals"][option["index"]]["P"]
            assert receipt["rgb_sha256"] == control["source_sha256"]
            assert read(forced / key / "build-manifest.json")["placement"]["P"] == camera
            source = score_build(expanded[key], forced / key)
            assert source["pack_sha256"] == digest(forced / key / "reconstruction.zip")
            new.append(dict(id=key, selection=option, pack_sha256=source["pack_sha256"],
                            geometry=source["geometry"], scores=source["scores"]))
        expected = inventory[key]
        assert source["source_sha256"] == control["source_sha256"] == expected["rgb_sha256"]
        if "reference_sha256" in source:
            assert source["reference_sha256"] == expected["source_reference_sha256"]
        else:
            assert control["dataset"] == "fixed-104" and source["pack_sha256"] == expected["ordinary"]["zip_sha256"]
            source = dict(source, reference_sha256=expected["source_reference_sha256"])
        challenger.append(source)
    save(REVIEW / "summary.json", dict(incumbent=summary(incumbent, frame_list), learned=summary(challenger, frame_list), new=new))
    save(REVIEW / "cases.json.gz", challenger)
    save(REVIEW / "manifest.json", dict(
        base="62611ee", source_freeze="38eb566", native_options_freeze="d6990f9", minecraft="Java 26.3",
        split="Whole building families. Harbor and Castle are previously reviewed regression families.",
        frozen_controls_sha256=digest(CONTROLS), reference_option_checks_sha256=digest(AUDIT),
        model_sha256=digest(REVIEW / "model.json"), source_cases_sha256=digest(REVIEW / "cases.json.gz"),
        summary_sha256=digest(REVIEW / "summary.json"), attempted_frames=70, inherited_views=218,
        negative_calibration_sha256=digest(REVIEW / "calibration-negatives.json"),
        extra_controls_sha256=digest(REVIEW / "additional-controls.json"),
    ))


if __name__ == "__main__":
    main()
