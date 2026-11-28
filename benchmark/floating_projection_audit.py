"""Trace the Mill's displayed grid footprints to the exact decoded round-two export."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_scores import load_reference
from benchmark.floating_extra_budget import LOCATION, paths
from benchmark.floating_grid_signoff import CASES, SQUARE
from benchmark.floating_second_signoff import ISOLATION, PART, REFIT, depth_options
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import axis_vector, oriented
from benchmark.relative_face_scoring import evaluate_relative_voxels
from engine.floating_sky import sky_support
from engine.geometry import project
from engine.surface_perimeter import _native_alpha


_, PANEL, NAME, WALL, _, _ = CASES[1]
REL = Path("sky") / NAME / "charts.json"
REBUILD = WORKTREES / "floating-depth-refit/runs/depth-refit" / PANEL / "source-refit"
CANDIDATE = "source-refit-with-small-steps"


def classify_pixels(owners, image, pose):
    height, width = owners.shape
    assert pose["image_size"] == [width, height]
    pitch = np.deg2rad(pose["yaw_pitch_deg"][1])
    focal = height / (2 * np.tan(np.deg2rad(pose["vertical_fov_deg"]) / 2))
    horizon = height / 2 - focal * np.tan(pitch)
    background = owners == 0
    # The capture has a level staging floor and no other labeled structures.
    # Above the horizontal vanishing line, native non-building rays see sky/cloud.
    ray_sky = background & ((np.arange(height)[:, None] + .5) < horizon)
    masks = dict(background=background, reference_sky=ray_sky, source_sky=sky_support(image))
    return float(horizon), masks


def footprint_mask(shape, homography, u, v):
    """Rasterize a projected cell by testing native pixel centers in its half-open UV square."""
    height, width = shape
    xy = project(homography, SQUARE + [u, v])
    low = np.floor(np.min(xy, axis=0)).astype(int).clip(0, [width, height])
    high = np.ceil(np.max(xy, axis=0)).astype(int).clip(0, [width, height])
    yy, xx = np.mgrid[low[1]:high[1], low[0]:high[0]]
    local = project(np.linalg.inv(homography), np.stack((xx + .5, yy + .5), axis=-1))
    inside = ((local >= [u, v]) & (local < [u + 1, v + 1])).all(axis=-1)
    return yy[inside], xx[inside]


def footprint_rows(groups, mask, classes):
    rows = []
    for group_id, group in groups.items():
        h = np.asarray(group["H"])
        for cell in group["cells"]:
            u, v = cell["u"], cell["v"]
            yn, xn = footprint_mask(mask.shape, h, u, v)
            assert len(yn) > 0
            record = dict(group=group_id, uv=[u, v], normal_offset=cell.get("normal_offset", 0),
                          pixels=len(yn), source_mask_pixels=int(mask[yn, xn].sum()))
            record.update({key: int(m[yn, xn].sum()) for key, m in classes.items()})
            record["reference_sky_in_mask"] = int((classes["reference_sky"][yn, xn] & mask[yn, xn]).sum())
            record["source_sky_in_mask"] = int((classes["source_sky"][yn, xn] & mask[yn, xn]).sum())
            record["center_px"] = project(h, [[u + .5, v + .5]])[0].tolist()
            rows.append(record)
    return rows


def trace_cells(charts, receipt, pack, aligned, truth):
    """Replay selected component writer order and track the actual material/position owner."""
    oriented_charts, _ = oriented(charts)
    origins = receipt["selected_origins"]
    score = receipt["score"]["fixed"]
    assert score["yaw_quarters"] == 0 and receipt["components"] == 1
    offset = np.asarray(score["translation"])
    written = {}
    records = {}
    for chart in sorted((c for c in oriented_charts if c["id"] in origins), key=lambda c: -c["confidence"]):
        u, v = [axis_vector(d) for d in chart["axes"]]
        normal = axis_vector(chart["normal_axis"])
        base = np.asarray(origins[chart["id"]]) + np.minimum(0, u) + np.minimum(0, v) - np.maximum(0, normal)
        for cell in chart.get("export_cells", chart["cells"]):
            step = cell.get("normal_offset", 0)
            local = base + cell["u"] * u + cell["v"] * v + step * normal
            key = tuple(map(int, local))
            world = tuple(map(int, local + offset))
            tag = (chart["id"], cell["u"], cell["v"], step)
            material = "minecraft:" + cell["block_id"].removeprefix("minecraft:")
            records[tag] = dict(local_xyz=key, aligned_xyz=world, owns_export=key not in written,
                                in_reference=world in truth, material=material)
            written.setdefault(key, material)
    assert written == pack
    assert {tuple(np.asarray(p) + offset): m for p, m in written.items()} == aligned
    return records


def counts(rows):
    return dict(cells=len(rows), owned=sum(r.get("owns_export", False) for r in rows),
                correct=sum(r.get("owns_export", False) and r["in_reference"] for r in rows),
                extra=sum(r.get("owns_export", False) and not r["in_reference"] for r in rows))


def totals(rows):
    output = {"all": counts(rows)}
    for key in ("background", "reference_sky", "source_sky"):
        tests = {"any": lambda n, total: n > 0, "majority": lambda n, total: 2 * n > total,
                 "all": lambda n, total: n == total}
        output[key] = {k: counts([r for r in rows if test(r[key], r["pixels"])]) for k, test in tests.items()}
        output[key]["sampled_pixels"] = sum(r[key] for r in rows)
    output["in_mask_reference_sky_pixels"] = sum(r["reference_sky_in_mask"] for r in rows)
    output["in_mask_source_sky_pixels"] = sum(r["source_sky_in_mask"] for r in rows)
    return output


def deletion_diagnostics(case, pack, rows, score):
    reference = load_reference(case)
    options = {
        "source_sky_majority": [r for r in rows if 2 * r["source_sky"] > r["pixels"]],
        "reference_sky_any": [r for r in rows if r["reference_sky"] > 0],
        "reference_sky_majority": [r for r in rows if 2 * r["reference_sky"] > r["pixels"]],
    }
    result = {}
    for name, candidates in options.items():
        deleted = {r["local_xyz"] for r in candidates if r["owns_export"]}
        retained = {p: m for p, m in pack.items() if p not in deleted}
        scored, _ = evaluate_relative_voxels(retained, reference, yaw_quarters=score["yaw_quarters"])
        result[name] = dict(dropped=counts(candidates), score=scored["fixed"])
    assert result["source_sky_majority"]["score"]["geometry"] == score["geometry"]
    return result


def verify_viewer(candidate_score):
    html = ROOT / "benchmark/reviews/floating-grid-signoff-03/review.html"
    raw, _ = html.read_text(encoding="utf-8").split("const cases=", 1)[1].split(", $=s", 1)
    page = next(r for r in json.loads(raw) if r["id"] == NAME + "-depth")
    expected = [candidate_score["geometry"]["strict"][k] for k in ("correct", "extras")]
    expected.append(candidate_score["scores"]["strict"]["correct"])
    assert page["options"][1]["score"] == expected
    return digest(html)


def audit(feedback):
    assert read(feedback)["reviews"][NAME + "-depth"]["relative_preference"] == "B"
    src, atlas_seeds, expanded_charts, cases = paths(PANEL)[0]
    case = next(c for c in cases if c["id"] == NAME)
    frame = Path(case["frame"])
    image_path = Path(case["variants"]["sky"]["image"])
    receipt_root = REBUILD / "assembly" / LOCATION / NAME
    recent = next(r for r in read(REFIT / "conditions.json.gz") if r["id"] == NAME and r["panel"] == PANEL)
    charts = read(REBUILD / "charts" / REL)["charts"]
    viewed = depth_options(PART / PANEL / "source", recent, REL, WALL)
    a, b = viewed[0][0], viewed[1][0]
    groups = {"parent": b[0], **{f"patch-{i}": c for i, c in enumerate(b[1:], 1)}}
    displayed = Counter((c["u"], c["v"], c.get("normal_offset", 0)) for group in b for c in group["cells"])
    saved_wall = next(c for c in charts if c["id"] == WALL)
    assert displayed == Counter((c["u"], c["v"], c.get("normal_offset", 0)) for c in saved_wall["export_cells"])
    labels = read(frame / "labels.json")
    truth = {tuple(c["xyz_world"]): c["block_id"] for c in labels["blocks"]}
    with np.load(frame / "labels.npz", allow_pickle=False) as arrays:
        owners = arrays["face_id"]
    with Image.open(image_path) as raw:
        source = raw.convert("RGB")
    horizon, classes = classify_pixels(owners, source, read(frame / "camera.json"))
    handoff = src / "sky" / NAME / "source-handoff"
    surfaces = read(handoff / "surfaces.json")
    plane = next(p for p in surfaces["planes"] if p["id"] == WALL)
    alpha, (x, y, right, bottom) = _native_alpha(plane["support"], source.size)
    mask = np.zeros(owners.shape, bool)
    mask[y:bottom, x:right] = np.asarray(alpha) > 0
    pack_path = receipt_root / "reconstruction.zip"
    pack = decoded_pack(pack_path)
    receipt = read(receipt_root / "score.json")
    score = receipt["score"]["fixed"]
    assert score == recent["score"][CANDIDATE]
    assert digest(pack_path) == receipt["pack_sha256"] == recent["pack_sha256"][CANDIDATE]
    assert digest(image_path) == receipt["source_sha256"] == recent["source_sha256"]
    assert all(digest(frame / fn) == value for fn, value in receipt["reference_sha256"].items())
    assert receipt["surfaces_sha256"] == digest(handoff / "surfaces.json")
    assert receipt["chart_sha256"] == digest(REBUILD / "charts" / REL)
    positions = read(receipt_root / "placements.json")
    assert pack == {tuple(p): m for p, m in positions["local"]}
    aligned = {tuple(p): m for p, m in positions["aligned"]}
    traces = trace_cells(charts, receipt, pack, aligned, truth)
    rows = footprint_rows(groups, mask, classes)
    for r in rows:
        r.update(traces[(WALL, *r["uv"], r["normal_offset"])])
    prior_rows = footprint_rows({"prior-depth": a[0]}, mask, classes)
    prior_root = PART / PANEL / "source" / "assembly" / LOCATION / NAME
    prior_pack = decoded_pack(prior_root / "reconstruction.zip")
    prior_receipt = read(prior_root / "score.json")
    prior_aligned = {tuple(p): m for p, m in read(prior_root / "placements.json")["aligned"]}
    prior_traces = trace_cells(read(PART / PANEL / "source" / "charts" / REL)["charts"],
                              prior_receipt, prior_pack, prior_aligned, truth)
    assert prior_aligned.keys() == aligned.keys()
    assert digest(prior_root / "reconstruction.zip") == prior_receipt["pack_sha256"] == recent["pack_sha256"]["source-depth"]
    for r in prior_rows:
        r.update(prior_traces[(WALL, *r["uv"], r["normal_offset"])])
    def cell_index(path):
        chart = next(c for c in read(path)["charts"] if c["id"] == WALL)
        return {(c["u"], c["v"]): c for c in chart["cells"]}
    origins = {label: cell_index(folder / REL) for label, folder in
               (("atlas_seed", atlas_seeds), ("dense_expansion", expanded_charts),
                ("large_contraction", ISOLATION / PANEL / "large-contraction" / "charts"))}
    majority = [r for r in rows if 2 * r["reference_sky"] > r["pixels"]]
    for r in majority:
        px, py = np.floor(r["center_px"]).astype(int)
        r["source_rgb"] = source.getpixel((px, py))
        r["row_border_rgb"] = [source.getpixel((x, py)) for x in (0, source.width - 1)]
        r["upstream_cells"] = {stage: index.get(tuple(r["uv"])) for stage, index in origins.items()}
    assert not any(r["in_reference"] for r in majority)
    result = dict(
        id=NAME, candidate=CANDIDATE, feedback_sha256=digest(feedback),
        source_sha256=receipt["source_sha256"], surface_handoff_sha256=receipt["surfaces_sha256"],
        chart_sha256=receipt["chart_sha256"], reference_sha256=receipt["reference_sha256"],
        viewer_sha256=verify_viewer(score), pack_sha256=receipt["pack_sha256"],
        source_fit=recent["patches"][0]["edge_axes"], horizon_y_px=horizon, native_mask_pixels=int(mask.sum()),
        source_mask_class_pixels={key: int((m & mask).sum()) for key, m in classes.items()},
        full_frame_class_pixels={key: int(m.sum()) for key, m in classes.items()},
        native_sky_missed_by_source=int((classes["reference_sky"] & ~classes["source_sky"]).sum()),
        source_sky_on_structures=int((classes["source_sky"] & (owners != 0)).sum()),
        selected_charts=receipt["selected_origins"], component_sizes=receipt["component_sizes"], score=score,
        whole_wall=totals(rows), baseline_a=totals(prior_rows),
        same_aligned_coordinates_as_a=True,
        deletion_diagnostics=deletion_diagnostics(case, pack, rows, score),
        groups={key: totals([r for r in rows if r["group"] == key]) for key in groups},
        per_cell=rows,
    )
    return result, source, classes, groups, mask


def draw_audit(path, result, source, classes, groups, source_mask, native_path=None):
    leaked = classes["reference_sky"] & source_mask
    shown = source.copy()
    shown.paste((255, 172, 62), mask=Image.fromarray(leaked.astype("uint8") * 110))
    draw = ImageDraw.Draw(shown)
    rows = {(r["group"], *r["uv"]): r for r in result["per_cell"]}
    highlights = []
    for group, chart in groups.items():
        for cell in chart["cells"]:
            u, v = cell["u"], cell["v"]
            quad = project(np.asarray(chart["H"]), SQUARE + [u, v])
            points = [tuple(pt) for pt in quad]
            color = "#26eaf5" if group == "parent" else "#ed83ce"
            draw.line(points + points[:1], fill=color, width=2)
            row = rows[group, u, v]
            if row["reference_sky"] and (2 * row["reference_sky"] > row["pixels"] or row["in_reference"]):
                highlights.append((row, points))
    font = ImageFont.truetype("arial.ttf", 17)
    for row, points in highlights:
        color = "#ff6057" if not row["in_reference"] else "#97ff93"
        draw.line(points + points[:1], fill=color, width=4)
        if not row["in_reference"]:
            cx, cy = row["center_px"]
            text = f"{round(100 * row['reference_sky'] / row['pixels'])}%"
            draw.text((cx, cy), text, anchor="mm", fill="white", stroke_width=2, stroke_fill="#491718", font=font)
    if native_path:
        native = Image.new("RGB", (shown.width, shown.height + 66), "#18202a")
        native.paste(shown)
        legend = ImageDraw.Draw(native)
        legend.text((15, shown.height + 8), "Red outline  five majority-sky extras    Green outline  three correct cells touching sky    Orange  native sky/cloud admitted by mask-0",
                    font=font, fill="#ffddaa")
        legend.text((15, shown.height + 36), "Cyan  parent grid    Pink  projecting depth patch    All cell footprints shown at the source resolution",
                    font=font, fill="#c1f5ef")
        native_path.parent.mkdir(parents=True, exist_ok=True)
        native.save(native_path, quality=95)
    if path is None:
        return
    # Crop to the upper-left roof where the review pointed out retessellated clouds.
    # Keep a fixed native rectangle so successive runs are visually comparable.
    region = (270, 137, 610, 365)
    context = source.copy()
    ImageDraw.Draw(context).rectangle(region, outline="#ffcb65", width=5)
    context.thumbnail((272, 300), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1336, 445), "#18202a")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype("arial.ttf", 20)
    draw.text((15, 12), "Riverside Mill · source", font=title, fill="white")
    draw.text((310, 12), "Original cloud and roof pixels", font=title, fill="white")
    draw.text((822, 12), "Round-two B · native support audit", font=title, fill="white")
    canvas.paste(context, (14, 46))
    for x, panel in ((310, source), (822, shown)):
        panel = panel.crop(region)
        panel = panel.resize((498, round(498 * panel.height / panel.width)), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, 46))
    draw.text((15, 240), "The five red cells are extras in", font=font, fill="#ffb1aa")
    draw.text((15, 267), "the selected 208-block export.", font=font, fill="#ffb1aa")
    draw.text((310, 391), "Orange   sky/cloud admitted by mask-0     Red outline   majority sky and extra placement", font=font, fill="#ffddaa")
    draw.text((310, 416), "Cyan   parent grid     Pink   refit depth patch     Green outline   correct placement touching sky", font=font, fill="#c1f5ef")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--native-overlay", type=Path)
    args = parser.parse_args()
    result, source, classes, groups, mask = audit(args.feedback)
    save(args.out, result)
    if args.figure or args.native_overlay:
        draw_audit(args.figure, result, source, classes, groups, mask, args.native_overlay)


if __name__ == "__main__":
    main()
