"""Isolate discovered wall-chart substitutions with the normal horizontal assembler and decoded pack."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from benchmark.floating_one_face_control import (
    ROOT, WORKTREES, PANELS, base, combined, digest, inputs, read, save, summarize,
)

GROUPS = (*PANELS, "holdout3", "holdout4")
OUT = ROOT / "runs/extra-isolation"
REVIEW = ROOT / "benchmark/reviews/floating-extra-isolation-01"
PREFIX = Path("connected-atlas/sky")
TRACE = {
    "holdout4": {"train-riverside-mill-1-warm-high-right", "train-riverside-mill-1-warm-low-left"},
    "sky": {"validation-hillside-chapel-1-warm-low-left"},
    "original": {"test-castle-01-limestone-low-left", "test-twinspire2-earth-high-right"},
    "fresh": {"validation-harbor-02-seafoam-low-left"},
    "supplemental": {"train-watchpost2-cool-low-left"},
    "holdout2": {"train-shrine-02-ochre-low-left", "train-market-court-1-warm-high-right"},
    "holdout3": {"train-sawtooth2-cool-high-right"},
}


def handoffs(group):
    if group == "holdout5":
        root = WORKTREES / "floating-extra-audit/runs/floating-holdout5"
        return (root / "ordinary", root / "ordinary/atlas/connected",
                root / "source_material/charts", read(root / "manifest.json")["cases"])
    return inputs(group)


def baseline(group):
    if group == "holdout5":
        return WORKTREES / "floating-extra-audit/runs/floating-holdout5/horizontal-cells/assembly"
    return base(group)


def finder(group):
    if group == "holdout5":
        return WORKTREES / "floating-extra-audit/runs/face-proposals/holdout5"
    key = {"holdout3": "new6", "holdout4": "new7"}.get(group, group)
    return WORKTREES / "floating-face-proposals/runs/face-proposals" / key


def large_contraction(old, new):
    old_cells, new_cells = len(old["cells"]), len(new["cells"])
    return old_cells >= 256 and 4 * new_cells <= old_cells


def local(chart):
    cells = chart["cells"]
    return dict(cells=len(cells), materials=Counter(c["block_id"] for c in cells),
                uv_bounds=[[min(c[k] for c in cells), max(c[k] for c in cells)]
                           for k in ("u", "v")] if cells else None)


def run(job):
    group, case = job
    name = case["id"]
    trace = name in TRACE.get(group, ())
    rel = Path("sky") / name / "charts.json"
    root = OUT / group
    original_path = handoffs(group)[2] / rel
    proposal_path = finder(group) / "charts" / rel
    receipt_path = finder(group) / "offers" / (name + ".json.gz")
    raw, proposed, receipt = read(original_path), read(proposal_path), read(receipt_path)
    assert digest(original_path) == proposed["source_packet_sha256"]
    assert digest(Path(case["variants"]["sky"]["image"])) == receipt["source_sha256"] == raw["source_sha256"]
    available = {r["id"] for r in receipt["charts"] if r.get("picked") is not None}
    pairs = list(zip(raw["charts"], proposed["charts"], strict=True))
    assert all(a["id"] == b["id"] and (a["id"] in available or a == b) for a, b in pairs)
    selected = {old["id"] for old, new in pairs if old["id"] in available and large_contraction(old, new)}
    variants = {"large-contraction": selected}
    if trace:
        variants.update({"only-" + cid: {cid} for cid in sorted(available)})
        variants.update({"drop-" + cid: available - {cid} for cid in sorted(available)})
    chart_details = {a["id"]: dict(old=local(a), new=local(b)) for a, b in pairs if a["id"] in available}
    roots = {"baseline": baseline(group), "source-one": finder(group) / "assembly"}
    # Each distinct substitution runs once. Even an empty gate gets the full consumer replay.
    replayed = {}
    for tag, ids in variants.items():
        key = frozenset(ids)
        if key not in replayed:
            folder = root / tag
            charts = [b if a["id"] in ids else a for a, b in pairs]
            save(folder / "charts" / rel, dict(raw, charts=charts, proposal_sha256=digest(proposal_path)))
            with patch.object(combined, "inputs", handoffs):
                combined.run((group, case, folder / "charts", folder / "assembly"))
            replayed[key] = folder / "assembly"
        roots[tag] = replayed[key]
    extra_diagnostic = {}
    if group == "holdout5" and name == "train-amphitheater-01-orchid-low-left":
        pid = "mask-0"
        seeds = read(handoffs(group)[1] / rel)
        assert seeds["source_sha256"] == raw["source_sha256"]
        sparse = next(chart for chart in seeds["charts"] if chart["id"] == pid)
        target = next(chart for chart in raw["charts"] if chart["id"] == pid)
        extra_diagnostic = dict(chart=pid, expanded=local(target), sparse=local(sparse),
                                picked=None, sparse_packet_sha256=digest(handoffs(group)[1] / rel))
        assert pid not in available
        for tag, replacement in (("sparse-mask-0", [sparse]), ("delete-mask-0", [])):
            charts = [chart for original in raw["charts"]
                      for chart in (replacement if original["id"] == pid else [original])]
            folder = root / tag
            save(folder / "charts" / rel, dict(raw, charts=charts))
            with patch.object(combined, "inputs", handoffs):
                combined.run((group, case, folder / "charts", folder / "assembly"))
            roots[tag] = folder / "assembly"
    summary = summarize([case], roots, PREFIX)["rows"][0]
    records = {}
    for arm, location in roots.items():
        record = read(location / PREFIX / name / "score.json")
        score = summary["score"][arm]
        records[arm] = dict(
            components=record["component_sizes"], origins=record["selected_origins"],
            yaw=score["yaw_quarters"], translation=score["translation"],
            pack_sha256=record["pack_sha256"], scoring_reference_sha256=record["reference_sha256"],
        )
    if not selected:
        assert summary["score"]["baseline"] == summary["score"]["large-contraction"]
        assert all(v == 0 for v in summary["paired"]["large-contraction"].values())
    if selected == available:
        assert summary["score"]["source-one"] == summary["score"]["large-contraction"]
    row = dict(**summary, charts=chart_details, selected=sorted(selected), diagnostic=extra_diagnostic,
               arms={tag: sorted(ids) for tag, ids in variants.items()}, exports=records,
               source_sha256=receipt["source_sha256"], old_packet_sha256=digest(original_path),
               proposal_packet_sha256=digest(proposal_path), offers_sha256=digest(receipt_path))
    save(root / "rows" / (name + ".json"), row)


def publish(groups):
    for group in groups:
        source, _, _, cases = handoffs(group)
        default = source / ("relative-face-assembly" if group == "sky" else "assembly")
        controls = summarize(cases, {"baseline": baseline(group), "floating-default": default}, PREFIX)
        rows = []
        for case, ctrl in zip(cases, controls["rows"], strict=True):
            row = read(OUT / group / "rows" / (case["id"] + ".json"))
            assert row["id"] == ctrl["id"] and row["score"]["baseline"] == ctrl["score"]["baseline"]
            row["score"]["floating-default"] = ctrl["score"]["floating-default"]
            row["paired"]["floating-default"] = ctrl["paired"]["floating-default"]
            rows.append(row)
        save(REVIEW / (group + ".json.gz"), rows)
    panels = {p.stem.split(".")[0]: read(p) for p in REVIEW.glob("*.json.gz")}
    panels["pooled-86"] = [row for g in GROUPS for row in panels.get(g, [])]
    lines = ["# Decoded placement by panel", "",
             "Counts are TP / FP / FN. Percentages are micro precision / recall / F1. "
             "Each export gets one whole-build registration shared by geometry and IDs.", "",
             "| Panel | Views | Arm | Metric | TP | FP | FN | Precision | Recall | F1 |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in (*GROUPS, "pooled-86", "holdout5"):
        if group not in panels:
            continue
        rows = panels[group]
        for arm in ("floating-default", "baseline", "source-one", "large-contraction"):
            for field in ("geometry", "scores"):
                for metric in ("strict", "facade"):
                    counts = [sum(r["score"][arm][field][metric][k] for r in rows)
                              for k in ("correct", "extras", "missed")]
                    tp, fp, fn = counts
                    rates = [100 * tp / max(1, tp + fp), 100 * tp / max(1, tp + fn),
                             200 * tp / max(1, 2 * tp + fp + fn)]
                    cols = [group, str(len(rows)), arm, metric + (" + ID" if field == "scores" else ""),
                            *map(str, counts), *(f"{v:.2f}" for v in rates)]
                    lines.append("| " + " | ".join(cols) + " |")
    (REVIEW / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    traces = ["# Individual wall substitutions", "",
              "Each row shows an actual decoded export. The component lists the source charts its blocks came from.", "",
              "| View | Arm | Correct | Extra | Missed | Exact IDs | Gained | Lost | Added extras | Removed extras | Selected component |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for group in (*GROUPS, "holdout5"):
        for r in panels.get(group, []):
            if r["id"] not in TRACE.get(group, set()) and not r.get("diagnostic"):
                continue
            for arm, export in r["exports"].items():
                score, paired = r["score"][arm], r["paired"][arm]
                counts = [score["geometry"]["strict"][k] for k in ("correct", "extras", "missed")]
                diff = [paired[k] for k in ("gained", "lost", "added", "removed")]
                values = [r["id"], arm, *counts, score["scores"]["strict"]["correct"],
                          *diff, ", ".join(export["origins"])]
                traces.append("| " + " | ".join(map(str, values)) + " |")
    (REVIEW / "chart-traces.md").write_text("\n".join(traces) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="*", default=GROUPS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        jobs = [(g, c) for g in args.groups for c in handoffs(g)[3]]
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run, jobs))
    publish(args.groups)
