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


def pooled_decorative(cases, arm):
    def f1(section, metric):
        scores = [case[arm][section][metric] for case in cases]
        correct = sum(score["correct"] for score in scores)
        errors = sum(score["missed"] + score["extras"] for score in scores)
        return 2 * correct / (2 * correct + errors)

    materials = [case[arm]["result"]["material"] for case in cases]
    accuracy = sum(row["correct"] for row in materials) / sum(row["positions"] for row in materials)
    return [f1("geometry", "strict"), accuracy, f1("result", "strict"),
            f1("result", "facade"), f1("geometry", "facade")]


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
    by_image = []
    for case in fixed["cases"]:
        pair = {}
        for name in ("baseline", "mask"):
            row = case[name]
            pair[name] = metrics(row["geometry"], row["scores"]["material"], row["scores"])
        by_image.append(dict(id=case["id"], changes=case["changes"], **pair))
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
    fixed_retry = read("material-feedback-grid-01/retry-bypass-fixed.json")["summary"]
    rows["retry-fixed"] = {
        name: metrics(fixed_retry[name]["geometry"], fixed_retry[name]["scores"]["material"], fixed_retry[name]["scores"])
        for name in ("ordinary", "without_retry")
    }
    decorative = read("material-feedback-geometry-01/decorative.json")
    for group in ("development", "confirmation"):
        subset = [case for case in decorative if case["id"].startswith("train-") == (group == "development")]
        rows["decorative-" + group] = {arm: pooled_decorative(subset, arm) for arm in ("baseline", "after")}
    for source, group in retries.items():
        rows["retry-" + source] = {
            name: metrics(group[name]["geometry"], group[name]["typed"]["material"], group[name]["typed"])
            for name in ("with_retry", "without_retry")
        }
    template = Path(__file__).with_name("pipeline_impact.html").read_text(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    html = template.replace("__MEASURED__", json.dumps(rows)).replace("__MASK_CASES__", json.dumps(by_image))
    cohorts = [
        ("Fixed trend", "104 / 104", rows["late-fixed"]["default"]),
        ("Visible Builds v3", "0 / 144", None),
        ("Decorative Builds v1 · development", "4 / 40", rows["decorative-development"]["baseline"]),
        ("Decorative Builds v1 · confirmation", "4 / 40", rows["decorative-confirmation"]["baseline"]),
    ]
    table = []
    for name, coverage, scores in cohorts:
        values = ('<td colspan="4">Unmeasured</td>' if scores is None else
                  "".join(f"<td>{scores[i]:.2%}</td>" for i in (0, 4, 2, 3)))
        table.append(f"<tr><th scope='row'>{name}</th><td>{coverage}</td>{values}</tr>")
    html = html.replace("__COHORT_ROWS__", "".join(table))
    (OUT / "viewer.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
