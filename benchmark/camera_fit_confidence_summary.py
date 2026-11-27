"""Compose decoded scores from the frozen decisions, verifying the exact selected camera."""

from benchmark.camera_model_controls import summary
from benchmark.expanded_baseline import digest, read, save, WORKTREES
from benchmark.camera_fit_confidence import CONTROLS, NATIVE, REVIEW, ROOT


def main():
    base = read(CONTROLS)
    inventory = {r["id"]: r for r in read(CONTROLS.parent / "inventory.json.gz")}
    frames = read(ROOT / "benchmark/reviews/camera-model-controls-01/frames.json")
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
            from benchmark.expanded_baseline import COHORTS, SOURCE, frames as source_frames
            from benchmark.expanded_scores import score_build
            entry = next(e for e in source_frames(SOURCE, COHORTS) if e["id"] == key)
            source = score_build(entry, forced / key)
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
    save(REVIEW / "summary.json", dict(incumbent=summary(incumbent, frames), learned=summary(challenger, frames), new=new))
    save(REVIEW / "cases.json.gz", challenger)


if __name__ == "__main__":
    main()
