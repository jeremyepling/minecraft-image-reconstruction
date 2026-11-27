"""Build a pipeline-impact panel from an independently graded, residual-only camera arm."""

import argparse
from pathlib import Path

from benchmark.camera_model_controls import summary
from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.pipeline_impact import challenger_rows


def panel(cases_file, method_id, name, note, baseline_file):
    controls = ROOT / "benchmark/reviews/camera-model-controls-01"
    baseline = read(baseline_file)
    frames = read(controls / "frames.json")
    manifest = read(controls / "manifest.json")["hashes"]
    for file in ("frames.json", "inventory.json.gz", "baseline-cases.json.gz"):
        if digest(controls / file) != manifest[file]:
            raise ValueError(f"Frozen control digest mismatch: {file}")
    inventory = {r["id"]: r for r in read(controls / "inventory.json.gz")}
    archived = {r["id"]: r for r in read(controls / "baseline-cases.json.gz")["ordinary"]}
    old = {r["id"]: r for r in baseline}
    cases = read(cases_file)
    by_id = {r["id"]: r for r in cases}
    if len(cases) != 288 or len(old) != 288 or by_id.keys() != old.keys() or old.keys() != inventory.keys():
        raise ValueError("Complete 288-frame decoded case inventory required")
    for key, before in old.items():
        after = by_id[key]
        for field in ("dataset", "split", "source_sha256", "visible_truth", "required_truth"):
            if before[field] != after[field]:
                raise ValueError(f"Frozen {field} differs for {key}")
        record = inventory[key]
        required_refs = record["source_reference_sha256"]
        if not required_refs or record["rgb_sha256"] != before["source_sha256"]:
            raise ValueError(f"Incomplete or mismatched inventory for {key}")
        for r in (before, after):
            if "reference_sha256" in r:
                if required_refs != r["reference_sha256"]:
                    raise ValueError(f"Changed reference for {key}")
            elif (r["dataset"] != "fixed-104" or r["pack_sha256"] != record["ordinary"]["zip_sha256"]
                  or required_refs != archived[key]["reference_sha256"]):
                raise ValueError(f"Cannot recover original verified fixed reference for {key}")
        if before["camera_usable"] or not after["camera_usable"]:
            for field in ("pack_sha256", "geometry", "scores"):
                if before[field] != after[field]:
                    raise ValueError(f"Retained {field} differs for {key}")
    pooled = summary(cases, frames)
    prior = summary(baseline, frames)

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
    parser.add_argument("--baseline-cases", type=Path, default=ROOT / "benchmark/reviews/camera-model-controls-01/independent-matched-cases.json.gz")
    parser.add_argument("--id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/reviews/camera-recovery-next-01")
    args = parser.parse_args()
    method, pooled = panel(args.cases, args.id, args.name, args.note, args.baseline_cases)
    target = args.out / f"{args.id}-summary.json"
    save(target, {"cases_sha256": digest(args.cases), "baseline_cases_sha256": digest(args.baseline_cases), "datasets": pooled})
    method["sha256"] = digest(target)
    save(args.out / f"{args.id}-panel.json", method)


if __name__ == "__main__":
    main()
