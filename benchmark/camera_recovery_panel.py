"""Build a pipeline-impact panel from an independently graded, residual-only camera arm."""

import argparse
from pathlib import Path

from benchmark.camera_model_controls import summary
from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.pipeline_impact import challenger_rows


def panel(cases_file, method_id, name, note):
    controls = ROOT / "benchmark/reviews/camera-model-controls-01"
    baseline = read(controls / "independent-matched-cases.json.gz")
    frames = read(controls / "frames.json")
    old = {r["id"]: r for r in baseline}
    cases = read(cases_file)
    by_id = {r["id"]: r for r in cases}
    if len(cases) != 288 or by_id.keys() != old.keys():
        raise ValueError("Complete 288-frame decoded case inventory required")
    for key, before in old.items():
        after = by_id[key]
        for field in ("dataset", "split", "source_sha256", "reference_sha256", "visible_truth", "required_truth"):
            if before[field] != after[field]:
                raise ValueError(f"Frozen {field} differs for {key}")
        if before["camera_usable"] or not after["camera_usable"]:
            for field in ("pack_sha256", "geometry", "scores"):
                if before[field] != after[field]:
                    raise ValueError(f"Retained {field} differs for {key}")
    pooled = summary(cases, frames)
    prior = read(controls / "independent-matched-summary.json")["datasets"]

    def arm(row):
        return {
            "geometry": {scope: row["geometry"][scope] for scope in ("strict", "facade")},
            "scores": {scope: row["scores"][scope] for scope in ("strict", "facade", "material")},
        }

    cohorts = {
        key: dict(views=stats["tested"], note=note, arms={"baseline": arm(prior[key]), "source": arm(stats)})
        for key, stats in pooled.items()
    }
    method = dict(id=method_id, name=name, artifact=f"camera-recovery-next-01/{method_id}-summary.json",
                  cohorts=cohorts)
    # Exercise the ordinary chart reader for all five measured quantities across all cohorts.
    chart_rows = {}
    challenger_rows({"experiments": [method]}, chart_rows)
    if len(chart_rows) != 3 or any(len(arm) != 5 for row in chart_rows.values() for arm in row.values()):
        raise ValueError("Incomplete chart row")
    return method, pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/reviews/camera-recovery-next-01")
    args = parser.parse_args()
    method, pooled = panel(args.cases, args.id, args.name, args.note)
    target = args.out / f"{args.id}-summary.json"
    save(target, {"cases_sha256": digest(args.cases), "datasets": pooled})
    method["sha256"] = digest(target)
    save(args.out / f"{args.id}-panel.json", method)


if __name__ == "__main__":
    main()
