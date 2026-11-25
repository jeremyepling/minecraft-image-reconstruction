"""Evaluate opening coverage and shared-edge depth consistency on saved surface predictions."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.camera_labels import rasterize
from benchmark.camera_round import read, save
from benchmark.surface_metrics import DOWNSAMPLE, truth


DEPTH_TOLERANCE = .5
DEFINITIONS = dict(
    openings="Quarter-resolution background sample with a building sample above in the same column and on each side in the same row. Gaps may open onto the floor. This image-space proxy can include space between separate structures.",
    filled="Covered by the union of predicted outline polygons; each opening pixel counts once per frame.",
    joined="Distinct matched truth patches whose planes differ in optical-axis depth by at most 0.5 blocks at both endpoint rays. Each ray uses the midpoint of the two proposed corner pixels. This tests depth consistency on extended planes; physical contact also requires their visible extents to meet.",
)


def opening_mask(id_map):
    building = id_map != 0
    above = np.logical_or.accumulate(building, axis=0)
    left = np.logical_or.accumulate(building, axis=1)
    right = np.logical_or.accumulate(building[:, ::-1], axis=1)[:, ::-1]
    return ~building & above & left & right


def opening_coverage(openings, proposals):
    covered = np.zeros_like(openings)
    for p in proposals:
        q = np.asarray(p["quad"], dtype=float)
        if q.shape != (4,2) or not np.isfinite(q).all():
            continue
        homogeneous = np.column_stack((q/DOWNSAMPLE, np.ones(4)))
        try:
            result = rasterize(homogeneous, openings.shape[1], openings.shape[0])
        except np.linalg.LinAlgError:
            continue
        if result is not None:
            box, polygon = result[:2]
            covered[box] |= polygon
    pixels, filled = int(openings.sum()), int(np.count_nonzero(covered & openings))
    return dict(opening_pixels=pixels, filled_pixels=filled, filled_fraction=filled/pixels if pixels else None)


def _endpoint_depths(chart, endpoints):
    rays = np.column_stack((endpoints, np.ones(2)))
    inverse_z = np.linalg.solve(chart, rays.T)[2]
    if not np.isfinite(inverse_z).all() or np.any(inverse_z <= 0):
        return None
    return 1/inverse_z


def junctions(t, model, matches):
    planes = model["planes"]
    by_id = {p["id"]: i for i, p in enumerate(planes)}
    if len(by_id) != len(planes):
        raise ValueError("Shared edges require unique model plane IDs")
    match = {m["proposal_index"]: m["patch_id"] for m in matches}
    edges = []
    for edge in model["shared_edges"]:
        indices = [by_id[i] for i in edge["planes"]]
        ids = [match.get(i) for i in indices]
        row = dict(planes=edge["planes"], corners=edge["corners"], patch_ids=ids,
                   matched_distinct=None not in ids and ids[0] != ids[1],
                   nonparallel=None, endpoint_xy=None, depths_camera_z=None,
                   endpoint_delta_blocks=None, supported=None)
        if row["matched_distinct"]:
            normals = [t["patches"][p]["normal"] for p in ids]
            row["nonparallel"] = bool(np.dot(*normals) == 0)  # axis-aligned full blocks
            endpoints = np.mean([np.asarray(planes[i]["quad"])[corners]
                                 for i, corners in zip(indices, edge["corners"])], axis=0)
            row["endpoint_xy"] = endpoints.tolist()
            depths = [_endpoint_depths(t["charts"][p], endpoints) for p in ids]
            if all(d is not None for d in depths):
                differences = abs(depths[0]-depths[1])
                row.update(depths_camera_z=[d.tolist() for d in depths],
                           endpoint_delta_blocks=differences.tolist(),
                           supported=bool(np.all(differences <= DEPTH_TOLERANCE+1e-9)))
        edges.append(row)
    return dict(joins=len(edges), matched_distinct=sum(e["matched_distinct"] for e in edges),
                measured=sum(e["supported"] is not None for e in edges),
                supported=sum(e["supported"] is True for e in edges), edges=edges)


def score_case(t, outlines, surfaces, selected_outline):
    mask = opening_mask(t["id_map"])
    checked = outlines[selected_outline]["scores"]["matches"]
    return dict(openings={name:opening_coverage(mask, d["proposals"]) for name,d in outlines.items()},
                junctions={name:junctions(t, d["model"], checked) for name,d in surfaces.items()})


def summarize(rows):
    result = dict(frames=len(rows), openings={}, junctions={})
    if not rows:
        return result
    for method in rows[0]["openings"]:
        data = [row["openings"][method] for row in rows]
        count = sum(d["opening_pixels"] for d in data)
        filled = sum(d["filled_pixels"] for d in data)
        result["openings"][method] = dict(opening_pixels=count, filled_pixels=filled,
            frames_with_openings=sum(d["opening_pixels"] > 0 for d in data),
            frames_with_filled_openings=sum(d["filled_pixels"] > 0 for d in data),
            filled_fraction=filled/count if count else None)
    for method in rows[0]["junctions"]:
        data = [row["junctions"][method] for row in rows]
        totals = {key:sum(d[key] for d in data) for key in ("joins", "matched_distinct", "measured", "supported")}
        max_gaps = [max(e["endpoint_delta_blocks"]) for d in data for e in d["edges"]
                    if e["endpoint_delta_blocks"] is not None]
        totals.update(max_endpoint_delta_blocks=max(max_gaps, default=None),
                      median_larger_endpoint_delta_blocks=float(np.median(max_gaps)) if max_gaps else None,
                      supported_per_join=totals["supported"]/totals["joins"] if totals["joins"] else None,
                      supported_per_measured=totals["supported"]/totals["measured"] if totals["measured"] else None)
        result["junctions"][method] = totals
    return result


def evaluate_run(out):
    """Read native truth and previously generated predictions. Write just the supplement."""
    out = Path(out).resolve()
    method = read(out/"selection.json")["selected"]["outlines"]
    cases, by_group, releases = [], defaultdict(list), {}
    for case in read(out/"cases.json"):
        root = (out/case["source"]).resolve().parent.parent
        if root not in releases:
            releases[root] = {f["id"]:f for f in read(root/"index.json")["frames"]}
        frame = releases[root][case["id"]]
        if frame["image_sha256"] != case["sha256"]:
            raise ValueError(f"Source fixture differs for {case['id']}")
        t = truth(read(root/frame["pose"]), read(root/frame["labels"])["faces"])
        folder = out/case["folder"]
        row = score_case(t, read(folder/"outlines.json"), read(folder/"surfaces.json"), method)
        cases.append({**{k:case[k] for k in ("id", "floor", "layout", "cohort")}, **row})
        by_group[case["cohort"]].append(row)
    report = dict(schema=1, run_id=out.name, selected_outline=method, definitions=DEFINITIONS,
                  depth_tolerance_blocks=DEPTH_TOLERANCE, pixel_stride=DOWNSAMPLE, cases=cases,
                  cohorts={k:summarize(v) for k,v in by_group.items()}, all=summarize(cases))
    save(out/"topology.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A completed surface round with cases.json")
    evaluate_run(parser.parse_args().run)
