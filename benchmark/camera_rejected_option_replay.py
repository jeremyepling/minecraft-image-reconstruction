"""Diagnostic export of the best calibrated *actual rejected* option on each remaining image."""

from concurrent.futures import ProcessPoolExecutor
import os

from benchmark.camera_horizon_replay import replay
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, WORKTREES, digest, frames, read, save


REVIEW = ROOT / "benchmark/reviews/camera-recovery-next-01"
EXPERIMENT = WORKTREES / "camera-native-floor-inverse"
RUN = EXPERIMENT / "runs/camera-native-floor-inverse-01"
OLD = WORKTREES / "full-expanded-controls/runs/expanded-baseline-01"


def rebuild(job):
    entry, arm, index, p90 = job
    key = entry["id"]
    file = RUN / arm / "metric" / (key + ".json")
    record = read(file)
    assert record["rgb_sha256"] == entry["rgb_sha256"]
    candidate = record["camera"]["evidence"]["refined_proposals"][index]
    assert not candidate["accepted"]
    camera = dict(P=candidate["P"], scale_status="supported", camera_source="reference-selected-rejected-source-option",
                  reason="Diagnostic forced replay of a numerically refined source chart")
    build = ROOT / "runs/camera-recovery-next-01/rejected-options" / key
    row = replay(entry, camera, OLD / entry["dataset"] / key, build)
    return dict(id=key, split=entry["split"], source_sha256=entry["rgb_sha256"], source_file_sha256=digest(file),
                arm=arm, index=index, p90_native_px=p90, pack_sha256=row["pack_sha256"],
                geometry=row["geometry"], scores=row["scores"], translation=row["translation"])


def main():
    os.environ.update(MCIR_IMAGE_ONLY="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    grades = read(EXPERIMENT / "benchmark/reviews/camera-native-floor-inverse-01/native-cameras.json.gz")
    entries = {e["id"]: e for e in frames(SOURCE, COHORTS)}
    tasks = []
    for row in grades:
        if row["camera"]["P"] is not None:
            continue
        candidates = [(v["native"]["p90_check_px"], arm, index) for arm, tried in row["arms"].items()
                      for index, v in enumerate(tried["refined"]) if v.get("native")]
        if candidates:
            p90, arm, index = min(candidates)
            if p90 <= 2:
                tasks.append((entries[row["id"]], arm, index, p90))
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(rebuild, tasks))
    save(REVIEW / "rejected-option-exports.json", rows)
    for row in rows:
        print(row["id"], row["geometry"]["strict"], row["scores"]["strict"], flush=True)


if __name__ == "__main__":
    main()
