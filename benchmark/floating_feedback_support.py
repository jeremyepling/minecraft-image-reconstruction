"""Test fallback to sparse visible wall cells when no full-face proposal supports expansion."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import digest, read, save
from benchmark.floating_bridges_report import summarize
from benchmark.floating_extra_budget import LOCATION, PANELS, ROOT, WORKTREES, combined, paths
from benchmark.floating_extra_feedback import grid
from engine.surface_perimeter import _native_alpha

PANELS = (*PANELS, "holdout3", "holdout4", "holdout5")
REGRESSION_LAYOUTS = {"amphitheater-01", "hillside-chapel-1", "riverside-mill-1", "sawtooth2", "twinspire2"}
OUT = ROOT / "runs/feedback-support"
REVIEW = ROOT / "benchmark/reviews/floating-feedback-support-01"
ISOLATION = WORKTREES / "floating-extra-isolation/runs/extra-isolation"


def needs_backoff(seed, chart, verification):
    """Use source measurements only. An unattempted face search supplies no veto."""
    return (verification is not None and bool(verification["offers"])
            and verification["picked"] is None
            and 0 < len(seed["cells"]) <= 2 and len(chart["cells"]) >= 256)


def low_contrast_fraction(rgb, support):
    """Fraction of native interior 3x3 neighborhoods spanning less than four luma levels."""
    alpha, (x, y, right, bottom) = _native_alpha(support, rgb.shape[1::-1])
    gray = cv2.cvtColor(rgb[y:bottom, x:right].astype(np.float32), cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = (np.asarray(alpha) > 0).astype(np.uint8)
    interior = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    span = cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)
    return float(np.mean(span[interior] < 4)) if np.any(interior) else 0.


def evaluate(job):
    group, case = job
    name = case["id"]
    (source, sparse_root, _, _), _, proposals = paths(group)
    contraction = ISOLATION / group / "large-contraction"
    rel = Path("sky") / name / "charts.json"
    sparse_path = sparse_root / rel
    chart_path = contraction / "charts" / rel
    receipt_path = proposals.parent / "offers" / (name + ".json.gz")
    seeds, packet, receipt = map(read, (sparse_path, chart_path, receipt_path))
    source_hash = digest(Path(case["variants"]["sky"]["image"]))
    assert seeds["source_sha256"] == packet["source_sha256"] == receipt["source_sha256"] == source_hash
    sparse = {c["id"]: c for c in seeds["charts"]}
    offers = {c["id"]: c for c in receipt["charts"]}
    decisions = {}
    for chart in packet["charts"]:
        key = chart["id"]
        if needs_backoff(sparse[key], chart, offers.get(key)):
            decisions[key] = dict(seeds=len(sparse[key]["cells"]), expanded=len(chart["cells"]),
                                  rejected_faces=len(offers[key]["offers"]))
    previous = contraction / "assembly"
    if not (previous / LOCATION / name / "score.json").is_file():
        # Cases with no distinct previous export can use the archived identical horizontal assembly.
        previous = paths(group)[1]
        archived = read(ISOLATION / group / "rows" / (name + ".json"))
        assert not archived["selected"]
        assert archived["score"]["large-contraction"] == archived["score"]["baseline"]
    roots = {"large-contraction": previous}
    if decisions:
        with Image.open(case["variants"]["sky"]["image"]) as im:
            rgb = np.array(im.convert("RGB"))
        planes = read(source / "sky" / name / "source-handoff/surfaces.json")["planes"]
        for plane in planes:
            if plane["id"] in decisions:
                decisions[plane["id"]]["smooth_fraction"] = low_contrast_fraction(rgb, plane["support"])
    smooth = {k for k, v in decisions.items() if v["smooth_fraction"] > .5}
    for arm, selected in (("sparse", set(decisions)), ("abstain", set(decisions)), ("smooth", smooth)):
        if arm == "smooth" and selected == set(decisions):
            roots[arm] = roots["sparse"]
        elif selected:
            charts = [sparse[c["id"]] if c["id"] in selected else c for c in packet["charts"]
                      if c["id"] not in selected or arm != "abstain"]
            dest = OUT / group / arm
            save(dest / "charts" / rel, dict(packet, charts=charts))
            with patch.object(combined, "inputs", lambda panel: paths(panel)[0]):
                combined.run((group, case, dest / "charts", dest / "assembly"))
            roots[arm] = dest / "assembly"
        else:
            # The identical chart list goes through the identical deterministic downstream path.
            roots[arm] = previous
    row = summarize([case], roots, LOCATION)["rows"][0]
    exports = {}
    for label, folder in roots.items():
        record = read(folder / LOCATION / name / "score.json")
        exports[label] = {k: record[k] for k in ("pack_sha256", "reference_sha256", "selected_origins")}
    assert all(v["reference_sha256"] == exports["large-contraction"]["reference_sha256"] for v in exports.values())
    return dict(row, group=group, layout=case["layout"], partition="feedback" if
                case["layout"] in REGRESSION_LAYOUTS else "transfer", decisions=decisions,
                source_sha256=source_hash, seed_packet_sha256=digest(sparse_path),
                chart_packet_sha256=digest(chart_path), offers_sha256=digest(receipt_path), exports=exports)


def totals(rows, arm, field, metric):
    tp, fn, fp = [sum(r["score"][arm][field][metric][k] for r in rows) for k in ("correct", "missed", "extras")]
    return [tp, fn, fp, 200 * tp / max(1, 2 * tp + fp + fn)]


def evidence_sheet(rows, cases):
    examples = {"train-amphitheater-01-orchid-low-left", "train-amphitheater-02-orchid-high-right",
                "test-castle-03-limestone-rear-oblique"}
    sheet = Image.new("RGB", (900, len(examples) * 238), "#182331")
    draw = ImageDraw.Draw(sheet)
    examples = [r for r in rows if r["id"] in examples]
    for row_i, row in enumerate(examples):
        name, group = row["id"], row["group"]
        chart_id, = row["decisions"]
        source, seed_root, _, _ = paths(group)[0]
        rel = Path("sky") / name / "charts.json"
        charts = []
        for path in (ISOLATION / group / "large-contraction/charts" / rel, seed_root / rel):
            charts.append(next(c for c in read(path)["charts"] if c["id"] == chart_id))
        planes = read(source / "sky" / name / "source-handoff/surfaces.json")["planes"]
        surface = next(p for p in planes if p["id"] == chart_id)
        with Image.open(cases[group, name]["variants"]["sky"]["image"]) as im:
            rgb = im.convert("RGB")
        _, box = _native_alpha(surface["support"], rgb.size)
        bound = (max(0, box[0]-12), max(0, box[1]-12), min(rgb.width, box[2]+12), min(rgb.height, box[3]+12))
        labels = ["Source crop", "Expanded grid", "Sparse atlas grid"]
        for i, img in enumerate([rgb, *(grid(rgb, chart) for chart in charts)]):
            panel = img.crop(bound)
            panel.thumbnail((295, 175), Image.Resampling.LANCZOS)
            x, y = 300*i, 238*row_i
            sheet.paste(panel, (x + (300-panel.width)//2, y+27))
            draw.text((x+6, y+5), labels[i], fill="white")
            if i:
                arm = ["large-contraction", "sparse"][i-1]
                stats = row["score"][arm]["geometry"]["strict"]
                caption = f"{len(charts[i-1]['cells']):,} wall cells | {stats['correct']:,} correct / {stats['extras']:,} extra"
            else:
                caption = name.split("-", 1)[1]
            draw.text((x+6, y+208), caption, fill="white")
    sheet.save(REVIEW / "wall-support.jpg", quality=90)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()
    jobs = [(g, c) for g in PANELS for c in paths(g)[0][3]]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(evaluate, jobs))
    assert len(rows) == len({(r["group"], r["id"]) for r in rows}) == 92
    assert sum(r["partition"] == "transfer" for r in rows) == 81
    save(REVIEW / "paired.json.gz", rows)
    evidence_sheet(rows, {(g, c["id"]): c for g, c in jobs})
    for subset, selected in [("all", rows), ("transfer", [r for r in rows if r["partition"] == "transfer"]),
                             ("feedback", [r for r in rows if r["partition"] == "feedback"]),
                             *((g, [r for r in rows if r["group"] == g]) for g in PANELS)]:
        for arm in ("large-contraction", "sparse", "abstain", "smooth"):
            print(subset, arm, [v for field in ("geometry", "scores") for k in ("strict", "facade")
                                for v in [totals(selected, arm, field, k)]])
    for r in rows:
        if r["decisions"]:
            print(r["group"], r["id"], r["decisions"], r["paired"])


if __name__ == "__main__":
    main()
