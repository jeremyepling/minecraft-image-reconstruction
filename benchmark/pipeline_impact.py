"""Build the offline sub-stage impact chart from the paired replay scores."""
import gzip
import json
from html import escape
from pathlib import Path

REVIEWS = Path(__file__).resolve().parent / "reviews"
OUT = REVIEWS / "pipeline-impact-01"


def read(name):
    payload = (REVIEWS / name).read_bytes()
    return json.loads(gzip.decompress(payload) if name.endswith('.gz') else payload)


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


def pooled_metrics(cases, geometry_key, result_key):
    def f1(section, metric):
        scores = [case[section][metric] for case in cases]
        correct = sum(score["correct"] for score in scores)
        errors = sum(score["missed"] + score["extras"] for score in scores)
        return 2 * correct / (2 * correct + errors)

    materials = [case[result_key]["material"] for case in cases]
    positions = sum(row["positions"] for row in materials)
    accuracy = sum(row["correct"] for row in materials) / positions if positions else None
    return [f1(geometry_key, "strict"), accuracy, f1(result_key, "strict"),
            f1(result_key, "facade"), f1(geometry_key, "facade")]


DATASETS = {"fixed-104": "Fixed trend", "visible-builds-v3": "Visible Builds v3",
            "decorative-builds-v1": "Decorative Builds v1"}


def measured_cohorts(summary):
    rows, splits = [], []
    for key, name in DATASETS.items():
        cohort = summary[key]
        if cohort["tested"] != cohort["available"]:
            raise ValueError(f"Incomplete frame set for {key}")
        def entry(label, group):
            scores = metrics(group["geometry"], group["scores"]["material"], group["scores"])
            return label, f"{group['tested']} / {group['available']}", scores
        rows.append(entry(name, cohort))
        if key != "fixed-104":
            for split in ("train", "validation", "test"):
                splits.append(entry(f"{name} · {split}", cohort["splits"][split]))
    return rows + splits


def camera_tables(cases, summary):
    totals, failures = [], []
    for dataset, name in DATASETS.items():
        group = [row for row in cases if row["dataset"] == dataset]
        failed = [row for row in group if not row["camera_usable"]]
        camera = summary[dataset]["camera"]
        if any(row["exported"] for row in failed) or camera["unusable"] != len(failed):
            raise ValueError("Camera failure counts differ from the scored exports")
        supported = camera["usable_geometry"]["strict"]
        conditional = f"{supported['f1']:.2%}" if camera["usable"] else "—"
        empty = sum(row["exported"] == 0 and row["camera_usable"] for row in group)
        totals.append(f"<tr><th scope='row'>{name}</th><td>{len(failed)} / {len(group)}</td>"
                      f"<td>{camera['misses_strict']:,}</td><td>{supported['missed']:,}</td>"
                      f"<td>{empty}</td><td>{conditional}</td></tr>")
        for row in failed:
            failures.append((row["visible_truth"], name, row))
    details = []
    for _, name, row in sorted(failures, key=lambda entry: (-entry[0], entry[2]["id"])):
        details.append(f"<tr><td>{name}</td><td>{escape(row['id'])}</td>"
                       f"<td>{escape(row['camera_reason'] or 'Reason unavailable')}</td>"
                       f"<td>{escape(row.get('atlas_camera_reason') or '—')}</td>"
                       f"<td>{escape(row.get('repeat_camera_reason') or '—')}</td>"
                       f"<td>{escape(row.get('native_floor_camera_reason') or '—')}</td>"
                       f"<td>{row['visible_truth']:,}</td><td>{row['required_truth']:,}</td></tr>")
    return "".join(totals), "".join(details)


def challenger_rows(report, rows):
    """Attach each intervention to its own same-image control and verified final metrics."""
    labels = {"baseline": "Same-path control", "source": "Source-image decision",
              "oracle": "Best offered choice using reference labels"}
    options, specs, gains = [], {}, []
    for method in report["experiments"]:
        for dataset, name in DATASETS.items():
            cohort = method["cohorts"][dataset]
            ids = cohort["arms"]
            control = ids["baseline"]
            item = ids["source"]
            key = method["id"] + "-" + dataset
            modes = {arm: labels[arm] for arm in labels if arm in ids}
            rows[key] = {arm: metrics(ids[arm]["geometry"], ids[arm]["scores"]["material"], ids[arm]["scores"])
                         for arm in modes}
            title = f"{method['name']} · {name}"
            options.append(f'<option value="{escape(key)}">{escape(title)}</option>')
            specs[key] = [modes, title, cohort["note"]]
            cells = []
            for section, metric in (("geometry", "strict"), ("geometry", "facade"),
                                    ("scores", "strict"), ("scores", "facade")):
                delta = (item[section][metric]["f1"] - control[section][metric]["f1"]) * 100
                if abs(delta) < 1e-9:
                    value = "0.00"
                else:
                    value = f"{delta:+.4f}" if abs(delta) < .01 else f"{delta:+.2f}"
                cells.append(f"<td>{value}</td>")
            gains.append(f"<tr><th scope='row'>{escape(method['name'])}</th><td>{name}</td>"
                         f"<td>{cohort['views']}</td>{''.join(cells)}</tr>")
    return "".join(options), specs, "".join(gains)


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
        rows["decorative-" + group] = {
            arm: pooled_metrics([case[arm] for case in subset], "geometry", "result")
            for arm in ("baseline", "after")
        }
    for source, group in retries.items():
        rows["retry-" + source] = {
            name: metrics(group[name]["geometry"], group[name]["typed"]["material"], group[name]["typed"])
            for name in ("with_retry", "without_retry")
        }
    round_path = REVIEWS / "expanded-round-01" / "summary.json"
    report = json.loads(round_path.read_bytes()) if round_path.is_file() else {"experiments": []}
    options, descriptions, gains = challenger_rows(report, rows)
    template = Path(__file__).with_name("pipeline_impact.html").read_text(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    html = template.replace("__MEASURED__", json.dumps(rows)).replace("__MASK_CASES__", json.dumps(by_image))
    expanded = read("camera-native-recovery-controls-01/independent-default-cases.json.gz")
    expanded_summary = read("camera-native-recovery-controls-01/independent-default-summary.json")["datasets"]
    cohorts = measured_cohorts(expanded_summary)
    table = []
    for name, coverage, scores in cohorts:
        values = ('<td colspan="4">Unmeasured</td>' if scores is None else
                  "".join(f"<td>{scores[i]:.2%}</td>" for i in (0, 4, 2, 3)))
        table.append(f"<tr><th scope='row'>{name}</th><td>{coverage}</td>{values}</tr>")
    html = html.replace("__COHORT_ROWS__", "".join(table))
    camera_summary, camera_cases = camera_tables(expanded, expanded_summary)
    html = html.replace("__CAMERA_SUMMARY__", camera_summary).replace("__CAMERA_CASES__", camera_cases)
    html = html.replace("__ROUND_OPTIONS__", options).replace("__ROUND_DESCRIPTIONS__", json.dumps(descriptions))
    html = html.replace("__ROUND_GAINS__", gains)
    if not gains:
        html = html.replace('<section id="experiment-gains">', '<section id="experiment-gains" hidden>')
    (OUT / "viewer.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
