"""Verify the frozen floating controls and link seven signoffs to the exact exports."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark import __path__ as benchmark_paths
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.floating_extra_budget import LOCATION, paths
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import axis_vector, oriented
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.typed_structures import geometry_score, overlap, score
from engine.geometry import bilinear, project
from engine.surface_perimeter import _native_alpha


REVIEWS = Path("benchmark/reviews")
ISOLATION = WORKTREES / "floating-extra-isolation/runs/extra-isolation"
DEPTH_WORKTREE = WORKTREES / "floating-part-grids"
DEPTH = DEPTH_WORKTREE / "runs/part-depth"
RULER_WORKTREE = WORKTREES / "floating-cross-ruler"
RULER = RULER_WORKTREE / "runs/cross-ruler"
SIGNOFF = WORKTREES / "floating-extra-review"
COHORT_ARCHIVE = WORKTREES / "floating-textured-cohort" / REVIEWS / "floating-textured-cohort-01/scores.json.gz"
DEPTH_ARCHIVE = DEPTH_WORKTREE / REVIEWS / "floating-part-depth-01/conditions.json.gz"
RULER_ARCHIVE = RULER_WORKTREE / REVIEWS / "floating-cross-ruler-01/conditions.json.gz"
DEPTH_ARMS = ("source", "visible-reference", "reference-source-step")
COUNTS = ("correct", "missed", "extras")
METRICS = (("geometry", "strict"), ("geometry", "facade"), ("scores", "strict"), ("scores", "facade"))


def verify_export(folder, expected, truth, optional, source_hash, reference_hash, pack_hash=None):
    receipt = read(folder / "score.json")
    pack = folder / "reconstruction.zip"
    assert receipt["pack_sha256"] == digest(pack)
    assert pack_hash is None or receipt["pack_sha256"] == pack_hash
    assert receipt["source_sha256"] == source_hash
    assert receipt["reference_sha256"] == reference_hash
    assert (receipt["score"]["fixed"] or receipt["score"]["relaxed_yaw"]) == expected
    decoded = decoded_pack(pack)
    positions = read(folder / "placements.json")
    assert decoded == {tuple(p): m for p, m in positions["local"]}
    yaw = expected["yaw_quarters"]
    t = expected["translation"]
    aligned = {}
    for (x, y, z), material in decoded.items():
        x, z = ((x, z), (-z, x), (-x, -z), (z, -x))[yaw]
        aligned[x + t[0], y + t[1], z + t[2]] = material
    assert aligned == {tuple(p): m for p, m in positions["aligned"]}
    measured = {"geometry": geometry_score(aligned, truth, optional), "scores": score(aligned, truth, optional, set())}
    for field, kind in METRICS:
        assert measured[field][kind] == expected[field][kind], (folder, field, kind)
    return aligned, receipt


def paired(before, after, truth):
    gains, losses = after.keys() - before.keys(), before.keys() - after.keys()
    return dict(gained=len(gains & truth.keys()), lost=len(losses & truth.keys()),
                added=len(gains - truth.keys()), removed=len(losses - truth.keys()))


def castle_trace(case, depth_row, frame, owners, camera_depth, labels, receipts, aligned):
    if str(DEPTH_WORKTREE / "benchmark") not in benchmark_paths:
        benchmark_paths.append(str(DEPTH_WORKTREE / "benchmark"))
    from benchmark.floating_part_depth import SAMPLES, boundary_order, fields, image_offsets, reference_offsets

    panel, name = depth_row["panel"], case["id"]
    source = paths(panel)[0][0]
    folder = ISOLATION / panel / "large-contraction"
    chart_file = folder / "charts/sky" / name / "charts.json"
    assert digest(chart_file) == depth_row["chart_sha256"]
    charts = read(chart_file)["charts"]
    chart = next(c for c in charts if c["id"] == "mask-0")
    oriented_chart = next(c for c in oriented(charts)[0] if c["id"] == chart["id"])
    support = next(p["support"] for p in read(source / "sky" / name / "source-handoff/surfaces.json")["planes"]
                   if p["id"] == chart["id"])
    size = np.asarray(owners.shape[::-1])
    alpha, (x, y, right, bottom) = _native_alpha(support, tuple(size))
    uv = np.array([(c["u"], c["v"]) for c in chart["cells"]])
    pixels = project(np.array(chart["H"]), uv[:, None, :] + SAMPLES)
    ix, iy = np.floor(pixels).astype(int).transpose(2, 0, 1)
    in_mask = (ix >= x) & (ix < right) & (iy >= y) & (iy < bottom)
    in_mask &= np.asarray(alpha)[(iy - y).clip(0, bottom - y - 1), (ix - x).clip(0, right - x - 1)] > 0
    supported = in_mask.sum(1) >= 5
    field, field_hashes = fields(panel, case, source)
    assert field_hashes == depth_row["field_sha256"]
    predicted, evidence = image_offsets(chart, uv, pixels, supported, field, size)
    native, depths, base = reference_offsets(chart, pixels, supported, owners, labels, read(frame / "camera.json"))
    record = next(c for c in depth_row["charts"] if c["chart"] == chart["id"])
    assert evidence == record["source"] and base == record["base_plane"]
    for actual, key in ((predicted, "source_offsets"), (native, "reference_offsets")):
        assert [[int(u), int(v), int(offset)] for (u, v), offset in zip(uv, actual) if offset] == record[key]
    assert boundary_order(uv, predicted, depths) == record["boundary_order"]
    source_depth = field["depth"]
    disparities = bilinear(source_depth[:, :, None], pixels * source_depth.shape[1::-1] / size)[:, :, 0]
    world_normal = tuple(axis_vector(oriented_chart["normal_axis"]))
    by_id = {row["id"]: row for row in labels["faces"]}
    full_cubes = {row["id"] for row in labels["blocks"] if row.get("geometry_kind", "full_cube") == "full_cube"}
    in_frame = (ix >= 0) & (ix < size[0]) & (iy >= 0) & (iy < size[1])
    ix, iy = ix.clip(0, size[0] - 1), iy.clip(0, size[1] - 1)
    sampled_ids = owners[iy, ix]
    comparable = []
    camera_inverse = np.full(len(uv), np.nan)
    image_inverse = np.full(len(uv), np.nan)
    for i in np.flatnonzero(np.isfinite(depths)):
        keep = [j for j, face_id in enumerate(sampled_ids[i])
                if in_frame[i, j] and face_id and by_id[face_id]["block"] in full_cubes
                and tuple(by_id[face_id]["normal_world"]) == world_normal
                and np.dot(by_id[face_id]["xyz_world"], world_normal) == depths[i]]
        good = camera_depth[iy[i, keep], ix[i, keep]]
        assert len(keep) >= 5 and np.all(good > 0)
        camera_inverse[i] = np.median(1 / good)
        image_inverse[i] = np.median(disparities[i, keep])
    lookup = {tuple(key): i for i, key in enumerate(uv)}
    for i in np.flatnonzero(predicted):
        u, v = uv[i]
        for neighbor in ((u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)):
            j = lookup.get(neighbor)
            if j is None or predicted[j] or not np.isfinite(depths[[i, j]]).all():
                continue
            comparable.append(dict(moved_uv=[int(u), int(v)], adjacent_uv=list(map(int, uv[j])),
                                   native_moved_minus_adjacent_blocks=int(depths[i] - depths[j]),
                                   estimated_inverse_depth_difference=float(image_inverse[i] - image_inverse[j]),
                                   recorded_inverse_camera_depth_difference=float(camera_inverse[i] - camera_inverse[j])))

    def raw_order(rows):
        hits = sum(np.sign(row["estimated_inverse_depth_difference"]) ==
                   np.sign(row["recorded_inverse_camera_depth_difference"]) for row in rows)
        return dict(agrees=int(hits), total=len(rows))

    t = np.asarray(depth_row["score"]["large-contraction"]["translation"])
    assert all(r["selected_origins"] == receipts["large-contraction"]["selected_origins"] and
               r["joins"] == receipts["large-contraction"]["joins"] for r in receipts.values())
    assert all(s["yaw_quarters"] == 0 and np.array_equal(s["translation"], t) for s in depth_row["score"].values())
    u, v = [axis_vector(axis) for axis in oriented_chart["axes"]]
    n = np.asarray(world_normal)
    origin = np.asarray(receipts["source"]["selected_origins"][chart["id"]])
    anchor = t + origin + np.minimum(0, u) + np.minimum(0, v) - np.maximum(0, n)
    moved = {tuple(anchor + key[0] * u + key[1] * v): (tuple(key), int(delta), i)
             for i, (key, delta) in enumerate(zip(uv, predicted)) if delta}
    removed = aligned["large-contraction"].keys() - aligned["source"].keys()
    added = aligned["source"].keys() - aligned["large-contraction"].keys()
    assert removed <= moved.keys()
    assert added <= {tuple(np.asarray(p) + delta * n) for p, (_, delta, _) in moved.items()}
    faces = {}
    for face in labels["faces"]:
        p = tuple(face["xyz_world"])
        if p in removed:
            faces.setdefault(p, []).append(dict(surface=face["surface_id"], direction=face["direction"],
                                                visible_pixels=face["visible_pixels"]))
    lost = []
    for p in sorted(faces):
        key, delta, i = moved[p]
        shifted = tuple(int(q) for q in np.asarray(p) + delta * n)
        lost.append(dict(from_xyz=list(map(int, p)), to_xyz=shifted, uv=list(map(int, key)),
                         center_px=pixels[i, 4].tolist(), faces=faces[p]))
    assert len(lost) == depth_row["paired"]["source"]["lost"]
    native_levels = {}
    for label, mask in (("moved", predicted != 0), ("unmoved", predicted == 0)):
        known = depths[mask & np.isfinite(depths)]
        native_levels[label] = dict(Counter(int(d - base) for d in known))
    return dict(chart_native_offsets_from_mode=native_levels, source_depth_record=record, boundary_pairs=comparable,
                source_inverse_camera_order=raw_order(comparable),
                source_inverse_camera_order_distinct_planes=raw_order(
                    [p for p in comparable if p["native_moved_minus_adjacent_blocks"] != 0]),
                selected_origins=receipts["source"]["selected_origins"], translation=t.tolist(),
                lost_correct_positions=lost)


def verify_condition(job):
    cohort_row, depth_row, ruler_row, case, review = job
    panel, name = cohort_row["panel"], cohort_row["id"]
    frame = Path(case["frame"])
    image = Path(case["variants"]["sky"]["image"])
    sha = digest(image)
    assert sha == case["variants"]["sky"]["source_sha256"] == cohort_row["source_sha256"]
    assert sha == depth_row["source_sha256"] == ruler_row["source_sha256"]
    reference_hash = {key: digest(frame / key) for key in depth_row["reference_sha256"]}
    assert reference_hash == depth_row["reference_sha256"] == ruler_row["reference_sha256"]
    labels = read(frame / "labels.json")
    with np.load(frame / "labels.npz", allow_pickle=False) as data:
        owners = data["face_id"]
        camera_depth = data["depth_z"] if review and "castle" in name else None
    truth = {tuple(b["xyz_world"]): b["block_id"] for b in labels["blocks"]}
    optional = horizontal_slivers(labels, owners)
    baseline = ISOLATION / panel / "large-contraction/assembly"
    baseline_hash = depth_row["pack_sha256"]["large-contraction"]
    assert depth_row["score"]["large-contraction"] == ruler_row["score"]["control"]
    assert baseline_hash == ruler_row["control_pack_sha256"]
    specs = {"large-contraction": (baseline, depth_row["score"]["large-contraction"], baseline_hash)}
    for arm in DEPTH_ARMS:
        pack_hash = depth_row["pack_sha256"][arm]
        root = DEPTH / panel / arm / "assembly" if pack_hash != baseline_hash else baseline
        specs[arm] = root, depth_row["score"][arm], pack_hash
    ruler_hash = ruler_row["candidate_pack_sha256"]
    ruler_root = RULER / panel / "assembly" if ruler_hash != baseline_hash else baseline
    specs["cross-ruler"] = ruler_root, ruler_row["score"]["candidate"], ruler_hash
    backgrounds = paths(panel)[0][0]
    integration = backgrounds / ("relative-face-assembly" if panel == "sky" else "assembly")
    receipt = read(integration / LOCATION / name / "score.json")
    specs["connected-atlas"] = integration, receipt["score"]["fixed"] or receipt["score"]["relaxed_yaw"], None
    aligned, receipts, scores = {}, {}, {}
    for arm, (root, expected, pack_hash) in specs.items():
        aligned[arm], receipts[arm] = verify_export(root / LOCATION / name, expected, truth, optional, sha,
                                                   reference_hash, pack_hash)
        scores[arm] = {f"{field}_{kind}": expected[field][kind] for field, kind in METRICS}
    assert scores["connected-atlas"] == cohort_row["arms"]["floating-default"]
    assert scores["large-contraction"] == cohort_row["arms"]["large-contraction"]
    delta = {arm: paired(aligned["large-contraction"], output, truth) for arm, output in aligned.items()}
    assert all(delta[k] == v for k, v in depth_row["paired"].items())
    assert delta["cross-ruler"] == ruler_row["paired"]["candidate"]
    output = {key: cohort_row[key] for key in ("panel", "id", "dataset", "split", "layout", "source_sha256")}
    output.update(reference_sha256=reference_hash, scores=scores, paired=delta,
                  pack_sha256={key: r["pack_sha256"] for key, r in receipts.items()})
    if review:
        kind = review["id"].rsplit("-", 1)[1]
        if kind == "depth":
            comparison = ("large-contraction", "source")
        elif kind == "cross":
            comparison = ("large-contraction", "cross-ruler")
        else:
            if kind == "origin":
                variant = SIGNOFF / "runs/feedback-boundary" / panel
                old = read(ROOT / REVIEWS / "floating-wall-feedback-01/boundary-and-origin.json.gz")[name]["scores"]
                comparison = ("resample", "origin")
                roots = {key: variant / key / "assembly" for key in comparison}
            else:
                variant = SIGNOFF / "runs/visible-top-row" / panel
                old = read(ROOT / REVIEWS / "floating-parts-feedback-02/top-row.json.gz")[name]["stats"]
                comparison = ("previous", "row-with-material")
                roots = {"previous": WORKTREES / "floating-feedback-scale/runs/feedback-scale" / panel / "assembly",
                         "row-with-material": variant / "row-with-material/assembly"}
            for key, root in roots.items():
                measured, receipt = verify_export(root / LOCATION / name, old["score"][key], truth, optional,
                                                  sha, reference_hash)
                aligned[key], receipts[key] = measured, receipt
            assert paired(aligned[comparison[0]], aligned[comparison[1]], truth) == old["paired"][comparison[1]]
        a, b = comparison
        review_case = dict(review=review, panel=panel, frame_id=name, source_sha256=sha, dataset=case["dataset"],
                           layout=case["layout"], a=a, b=b, paired=paired(aligned[a], aligned[b], truth),
                           pack_sha256={key: receipts[key]["pack_sha256"] for key in comparison},
                           selected_charts={key: receipts[key]["selected_origins"] for key in comparison},
                           scores={key: receipts[key]["score"]["fixed"] or receipts[key]["score"]["relaxed_yaw"]
                                   for key in comparison})
        if camera_depth is not None:
            castle = castle_trace(case, depth_row, frame, owners, camera_depth, labels,
                                  {k: v for k, v in receipts.items() if k in depth_row["score"]}, aligned)
            review_case["castle_trace"] = castle
        output["review_case"] = review_case
    return output


def pool(rows):
    return dict(conditions=len(rows), unique_captures=len({(r["dataset"], r["id"]) for r in rows}),
                scores={arm: {metric: overlap(*(sum(r["scores"][arm][metric][k] for r in rows) for k in COUNTS))
                              for metric in rows[0]["scores"][arm]} for arm in rows[0]["scores"]},
                paired={arm: {k: sum(r["paired"][arm][k] for r in rows) for k in rows[0]["paired"][arm]}
                        for arm in rows[0]["paired"]})


def castle_figure(path, case, trace):
    frame = Path(case["frame"])
    lost_xyz = {tuple(row["from_xyz"]) for row in trace["lost_correct_positions"]}
    labels = read(frame / "labels.json")
    affected = [face["id"] for face in labels["faces"] if tuple(face["xyz_world"]) in lost_xyz]
    with np.load(frame / "labels.npz", allow_pickle=False) as data:
        mask = np.isin(data["face_id"], affected)
    with Image.open(case["variants"]["sky"]["image"]) as source:
        rgb = source.convert("RGB")
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()) - 70, int(ys.min()) - 80, int(xs.max()) + 71, int(ys.max()) + 81)
    marked = rgb.copy()
    marked.paste((255, 93, 82), mask=Image.fromarray(np.uint8(mask) * 170))
    context = rgb.copy()
    ImageDraw.Draw(context).rectangle(box, outline=(255, 184, 75), width=5)
    context.thumbnail((496, 360), Image.Resampling.LANCZOS)
    detail = marked.crop(box)
    detail = detail.resize((490, round(490 * detail.height / detail.width)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1038, 430), (24, 32, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf", 21)
    draw.text((16, 14), "Castle · Original high-right source", font=font, fill="white")
    draw.text((528, 14), "Visible faces of the 14 lost blocks", font=font, fill="white")
    canvas.paste(context, (16, 52 + (310 - context.height) // 2))
    canvas.paste(detail, (528, 52 + (310 - detail.height) // 2))
    detail_font = ImageFont.truetype("arial.ttf", 17)
    draw.text((528, 376), "The parapet moved from Z = 17 to Z = 16.", font=detail_font, fill="white")
    draw.text((528, 400), "Eight blocks also show east-facing returns.", font=detail_font, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=93)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("feedback", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--castle-figure", type=Path)
    cli.add_argument("--workers", default=4, type=int)
    args = cli.parse_args()
    feedback = read(args.feedback)
    assert feedback["schema"] == "floating-grid-signoff-v2"
    published_view = ROOT / REVIEWS / "floating-parts-feedback-02/review.html"
    payload, _ = published_view.read_text(encoding="utf-8").split("const cases=", 1)[1].split(", $=s", 1)
    display = {r["id"]: r for r in json.loads(payload)}
    cohort = read(COHORT_ARCHIVE)
    controls = {(r["panel"], r["id"]): r for r in read(DEPTH_ARCHIVE)}
    rulers = {(r["group"], r["id"]): r for r in read(RULER_ARCHIVE)}
    all_rows = cohort["rows"]
    active_rows = [r for r in all_rows if r["active"]]
    assert len(all_rows) == len(controls) == len(rulers) == 92 and len(active_rows) == 84
    panels = {r["panel"] for r in active_rows}
    cases = {(panel, r["id"]): r for panel in panels for r in paths(panel)[0][3]}
    reviews = {}
    for review in feedback["reviews"].values():
        name, kind = review["id"].rsplit("-", 1)
        assert review["wall"] == display[review["id"]]["wall"]
        matches = [r for r in active_rows if r["id"] == name and (kind not in ("cross", "origin") or r["panel"] == "sky")
                   and (kind != "depth" or "castle" not in name or r["panel"] == "original")]
        assert len(matches) == 1
        reviews[matches[0]["panel"], name] = review
    jobs = []
    for row in active_rows:
        key = row["panel"], row["id"]
        jobs.append((row, controls[key], rulers[key], cases[key], reviews.get(key)))
    with ProcessPoolExecutor(args.workers) as workers:
        rows = list(workers.map(verify_condition, jobs))
    seen = {(r["dataset"], r["layout"]) for r in rows if "review_case" in r}
    for r in rows:
        r["partition"] = "reviewed" if (r["dataset"], r["layout"]) in seen else "transfer"
    sky = [r for r in rows if r["panel"] == "sky"]
    assert len(sky) == 6 and len(seen) == 7
    groups = {"active": rows, "fixed_sky": sky}
    for name, group in list(groups.items()):
        groups.update({f"{name}_{p}": [r for r in group if r["partition"] == p] for p in ("reviewed", "transfer")})
    review_cases = {}
    for row in rows:
        if "review_case" not in row:
            continue
        case = row.pop("review_case")
        review = case["review"]
        shown = []
        for key in (case["a"], case["b"]):
            arm = case["scores"][key]
            shown.append([arm["geometry"]["strict"]["correct"], arm["geometry"]["strict"]["extras"],
                          arm["scores"]["strict"]["correct"]])
        assert shown == [option["score"] for option in display[review["id"]]["options"]]
        case["explicit_ratings"] = {k: v for k, v in review.items()
                                    if re.fullmatch(r"[AB]_(coverage|alignment|size)|relative_preference|depth_quality|proposal_decision", k)
                                    and v.strip()}
        case["comments"] = {k: v.strip() for k, v in review.items() if k.endswith("_comment") and v.strip()}
        review_cases[review["id"]] = case
    archive_sha = {key: digest(path) for key, path in {"cohort": COHORT_ARCHIVE, "depth": DEPTH_ARCHIVE,
                                                      "ruler": RULER_ARCHIVE, "review_html": published_view}.items()}
    save(args.out, dict(schema="floating-signoff-audit-v1", feedback_sha256=digest(args.feedback), archive_sha256=archive_sha,
                        cohort=cohort["cohort"], cohort_manifest_lf_sha256=cohort["cohort_manifest_lf_sha256"],
                        reviewed_layouts=sorted(seen), scoring="Decoded pack, one whole-build yaw and integer translation, visible-only labels.",
                        cohorts={name: pool(group) for name, group in groups.items()},
                        review_cases=review_cases, conditions=rows))
    if args.castle_figure:
        castle = review_cases["test-castle-01-ashlar-high-right-depth"]
        castle_figure(args.castle_figure, cases[castle["panel"], castle["frame_id"]], castle["castle_trace"])


if __name__ == "__main__":
    main()
