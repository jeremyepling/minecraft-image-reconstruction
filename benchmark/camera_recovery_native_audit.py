"""Grade frozen native-floor cameras without feeding reference data into prediction."""

from benchmark.camera_horizon_replay import camera_validation
from benchmark.camera_horizon_report import ground_validation
from benchmark.camera_relative_calibration import native_corner_score
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save


def main():
    reviews = ROOT / "benchmark/reviews"
    cohort = read(reviews / "camera-model-controls-01/independent-matched-cases.json.gz")
    run = WORKTREES / "camera-native-floor-inverse/runs/camera-native-floor-inverse-01"
    rows = []
    for base in cohort:
        if base["camera_usable"]:
            continue
        key = base["id"]
        file = run / "metric" / (key + ".json")
        record = read(file)
        camera = record["camera"]
        assert record["rgb_sha256"] == base["source_sha256"]
        frame = SOURCE / base["dataset"] / "data" / base["split"] / "frames" / key
        row = dict(id=key, split=base["split"], seed=record["arm"], source_sha256=base["source_sha256"],
                   receipt_sha256=digest(file), accepted=camera["P"] is not None, reason=camera["reason"])
        if row["accepted"]:
            reference = base["reference_sha256"]
            for name in ("camera.json", "labels.json"):
                assert digest(frame / name) == reference[name]
            entry = dict(frame=str(frame))
            row["registered"] = camera_validation(entry, camera)
            row["ground"] = ground_validation(entry, camera)
            row["relative"] = native_corner_score(camera["P"], read(frame / "camera.json"),
                                                  read(frame / "labels.json")["faces"])
        rows.append(row)
    summary = {}
    for split in ["all", "train", "validation", "test"]:
        subset = [r for r in rows if split in ("all", r["split"])]
        good = [r for r in subset if r["accepted"]]
        summary[split] = {
            "attempts": len(subset), "accepted": len(good),
            "relative_check_vertices": sum(r["relative"]["check_vertices"] for r in good),
            "relative_check_within_2px": sum(r["relative"]["check_within_2px"] for r in good),
            "relative_frames_p90_within_2px": sum(r["relative"]["p90_check_px"] <= 2 for r in good),
            "relative_worst_p90_px": max(r["relative"]["p90_check_px"] for r in good),
            "registered_worst_p90_px": max(r["registered"]["p90_reprojection_px"] for r in good),
            "ground_max_median_metric_span_fraction_error": max(r["ground"]["ground_metric_span_relative_error"] for r in good),
        }
    save(reviews / "camera-recovery-next-01/native-calibration.json.gz", dict(summary=summary, cases=rows))
    for name, data in summary.items():
        print(name, data)


if __name__ == "__main__":
    main()
