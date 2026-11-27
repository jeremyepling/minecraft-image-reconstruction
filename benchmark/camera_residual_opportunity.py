"""Join the verified default and recorded-camera replay, for diagnosis only."""

import gzip
import hashlib
import json
from pathlib import Path


REVIEWS = Path(__file__).resolve().parent / "reviews"
INPUTS = {
    "default": REVIEWS / "camera-model-controls-01/independent-matched-cases.json.gz",
    "recorded": REVIEWS / "camera-recorded-pose-01/cases.json.gz",
    "floor": REVIEWS / "expanded-baseline-01/camera-diagnosis.json",
}


def read(name):
    path = INPUTS[name]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def metrics(rows):
    output = {"views": len(rows)}
    for field, prefix in [("geometry", "position"), ("scores", "position_id")]:
        for mode in ("strict", "facade"):
            values = {
                key: sum(x[field][mode][key] for x in rows)
                for key in ("correct", "missed", "extras")
            }
            denom = 2 * values["correct"] + values["missed"] + values["extras"]
            values["f1"] = 2 * values["correct"] / denom if denom else None
            output[f"{prefix}_{mode}"] = values
    return output


def main():
    default = read("default")
    recorded = {r["id"]: r for r in read("recorded") if r["replayed"]}
    floor = {r["id"]: r for r in read("floor")["cases"]}
    residual = [r for r in default if not r["camera_usable"]]
    groups = {}
    for row in residual:
        frame = row["id"]
        replay = recorded[frame]
        assert row["dataset"] == "visible-builds-v3"
        assert row["exported"] == 0
        assert row["source_sha256"] == replay["source"]["source_sha256"] == floor[frame]["source_sha256"]
        assert row["reference_sha256"] == replay["source"]["reference_sha256"]
        result = replay["source"]
        chart = "pass" if floor[frame]["true_atlas_accepted"] else "fail"
        for label in ["all", "split/" + row["split"], "family/" + row["family"],
                      "known_chart/" + chart, "split_chart/" + row["split"] + "/" + chart]:
            groups.setdefault(label, []).append(result)

    composed = [recorded[r["id"]]["source"] if not r["camera_usable"] else r for r in default]
    report = {
        "kind": "diagnostic-only composition of existing decoded measurements",
        "files_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in INPUTS.items()},
        "current_v3": metrics([r for r in default if r["dataset"] == "visible-builds-v3"]),
        "residual_with_recorded_camera": {k: metrics(v) for k, v in sorted(groups.items())},
        "incumbent_plus_recorded_on_residual": {
            d: metrics([r for r in composed if r["dataset"] == d])
            for d in sorted(set(r["dataset"] for r in default))
        },
    }
    directory = REVIEWS / "camera-recovery-next-01"
    directory.mkdir(exist_ok=True)
    (directory / "opportunity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
