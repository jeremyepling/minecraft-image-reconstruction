"""Assemble camera review panels from the independent export grades and score witnesses."""

from benchmark.camera_recovery_panel import panel
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save


REVIEWS = ROOT / "benchmark/reviews"
OUT = REVIEWS / "camera-recovery-next-01"


def artifact(worktree, name):
    candidates = [REVIEWS / name, WORKTREES / worktree / "benchmark/reviews" / name]
    return next(path for path in candidates if path.is_file())


def main():
    old = REVIEWS / "camera-model-controls-01/independent-matched-cases.json.gz"
    native = artifact("camera-native-recovery-controls", "camera-native-recovery-controls-01/independent-default-cases.json.gz")
    combined = artifact("camera-native-recovery-controls", "camera-native-recovery-controls-01/independent-composed-cases.json.gz")
    learned = artifact("camera-native-fit-confidence", "camera-native-fit-confidence-01/cases.json.gz")
    silhouette = artifact("camera-silhouette-ruler", "camera-silhouette-ruler-01/manifest.json")
    incumbent = {r["id"]: r["pack_sha256"] for r in read(old)}
    proof = read(silhouette)
    if dict(proof["incumbent_pack_hashes"]) != incumbent or digest(old) != proof["controls"]:
        raise ValueError("Silhouette incumbent inventory differs")
    methods = [
        ("camera-native-floor", "Native pavement texture fit", old, native,
         "A model-free image-boundary search verifies light gray concrete in native pixels. "
         "It adds 63 cameras after the previous camera stages abstain."),
        ("camera-native-geo-seed", "Optional GeoCalib seed for the native pavement fit", native, combined,
         "The control already includes the model-free native fit. A learned horizon and focal seed "
         "recovers one training Pillars image and one previously reviewed Castle high-right image."),
        ("camera-silhouette-ruler", "Camera from visible silhouette steps", old, old,
         "Independent scale and ground checks accepted zero of 70. The 288 decoded control packs carry forward. "
         "Verifiable training steps rose one block. Verifiable Castle steps rose two blocks."),
        ("camera-native-fit-confidence", "Learned confidence for native camera proposals", combined, learned,
         "The control includes the native pavement fit and optional learned seed. Texture and unit-repeat "
         "confidence recover one training Pillars image. No additional Harbor or Castle view passed."),
    ]
    panels = []
    inputs = {}
    for tag, name, before, after, note in methods:
        result, scores = panel(after, tag, name, note, before)
        target = OUT / f"{tag}-summary.json"
        save(target, dict(cases_sha256=digest(after), baseline_cases_sha256=digest(before), datasets=scores))
        result["sha256"] = digest(target)
        save(OUT / f"{tag}-panel.json", result)
        panels.append(result)
        inputs[tag] = {"baseline": digest(before), "source": digest(after)}
    save(OUT / "panel-entries.json", dict(experiments=panels))
    save(OUT / "manifest.json", dict(
        base="62611ee", minecraft="Java 26.3", views={"fixed-104": 104, "visible-builds-v3": 144, "decorative-builds-v1": 40},
        frozen_frames_sha256=digest(REVIEWS / "camera-model-controls-01/frames.json"), method_inputs_sha256=inputs,
        source_freezes={"experiment/camera-native-floor-inverse": "d6990f9", "experiment/camera-silhouette-ruler": "493350c",
                        "experiment/camera-native-fit-confidence": "38eb566"},
        native_runtime_candidate="aac7da7", camera_scoring="Fit one integer XYZ shift on alternating unique visible native vertices, test the others.",
        placement_scoring="Strict and facade-priority geometry after one global integer translation. Evaluate IDs using the same shift.",
        cohort_note="Original family splits. Harbor and Castle are reviewed regression families.",
        assets_sha256={p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix in (".md", ".json", ".gz", ".jpg")
                       and p.name != "manifest.json"},
    ))


if __name__ == "__main__":
    main()
