"""Build the offline sub-stage impact chart from the paired replay scores."""
import json
from pathlib import Path

REVIEWS = Path(__file__).resolve().parent / "reviews"
OUT = REVIEWS / "pipeline-impact-01"


def read(name):
    return json.loads((REVIEWS / name).read_bytes())


def metrics(geometry, material, overall):
    return [
        geometry["strict"]["f1"],
        material["accuracy"],
        overall["strict"]["f1"],
        overall["facade"]["f1"],
        geometry["facade"]["f1"],
    ]


def early_metrics(row):
    return metrics(row["geometry"], row["material"], row["typed"])


def main():
    early = read("reference-surface-feedback-01/summary.json")["cohorts"]
    late = read("material-feedback-surfaces-01/material-oracle-summary.json")
    released = read("typed-structures-01/summary.json")
    export_controls = {
        row["id"]: row for row in read("material-feedback-surfaces-01/perfect-export-control.json")["cases"]
    }
    material_cases = read("material-feedback-surfaces-01/material-oracle-cases.json")
    if any(export_controls[row["id"]][metric]["f1"] != 1.0
           for row in material_cases for metric in ("strict", "facade")):
        raise ValueError("Visible reference packs failed the perfect-export control")
    rows = {}
    for group in ("fixed", "other", "all"):
        rows["early-" + group] = {
            name: early_metrics(row) for name, row in early[group].items()
        }

    for group, cohort in [("fixed", released["cohorts"]["fixed"]), ("additional", released["other_unseen"])]:
        geometry = cohort["geometry"]
        result = {}
        for mode, row in late[group].items():
            scores = metrics(geometry, row["material"], row)
            if mode in ("positions", "both"):
                scores[0] = scores[4] = 1.0
            result[mode] = scores
        # On the existing positions, perfect IDs make the joint scores equal the geometry scores.
        result["perfect_ids"] = metrics(geometry, {"accuracy": 1.0}, geometry)
        result["perfect_export"] = [1.0] * 5
        rows["late-" + group] = result

    fixed = read("reference-surface-feedback-01/fixed-confirmation.json")
    rows["full-masks"] = {name: early_metrics(fixed[name]) for name in ("baseline", "mask")}
    for group, filename in [
        ("bypass-ranked", "component-bypass-summary.json"),
        ("sampling", "primary-ray-calibration-summary.json"),
    ]:
        ranked = read("material-feedback-geometry-01/" + filename)
        rows[group] = {
            name: metrics(row["geometry"], row["result"]["material"], row["result"])
            for name, row in ranked.items()
        }
    counters = read("material-feedback-grid-01/count-only-oracle.json")["summary"]
    rows["bypass-counts"] = {}
    for name, row in counters.items():
        scores = row["metrics"]
        accuracy = scores["typed_strict"]["correct"] / scores["geometry_strict"]["correct"]
        rows["bypass-counts"][name] = [
            scores["geometry_strict"]["f1"], accuracy, scores["typed_strict"]["f1"],
            scores["typed_facade"]["f1"], scores["geometry_facade"]["f1"],
        ]
    retries = read("material-feedback-surfaces-01/retry-bypass.json")["summary"]["all"]
    for source, group in retries.items():
        rows["retry-" + source] = {
            name: metrics(group[name]["geometry"], group[name]["typed"]["material"], group[name]["typed"])
            for name in ("with_retry", "without_retry")
        }
    template = Path(__file__).with_name("pipeline_impact.html").read_text(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    (OUT / "viewer.html").write_text(template.replace("__MEASURED__", json.dumps(rows)), encoding="utf-8")


if __name__ == "__main__":
    main()
