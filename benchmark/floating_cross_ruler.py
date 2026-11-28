"""Calibrate wall-unit scale from independent, source-verified faces and relative depth."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import benchmark
import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.relative_oracle_charts import fit_homography
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

benchmark.__path__.append(str(WORKTREES / "floating-extra-isolation/benchmark"))
from benchmark.floating_extra_isolation import GROUPS, PREFIX, combined, finder, handoffs, large_contraction, summarize

OUT = ROOT / "runs/cross-ruler"
REVIEW = ROOT / "benchmark/reviews/floating-cross-ruler-01"
ARCHIVE = WORKTREES / "floating-extra-isolation/runs/extra-isolation"


def height_at_center(H, plane, size):
    # At a plane's median screen point, d(pixel_y)/dv is proportional to
    # (vertical_column_y - pixel_y * vertical_column_z) / camera_depth.
    # Remove the pixel_y dependence before comparing faces elsewhere in the frame.
    runs = np.array(plane["support"]["runs"])
    widths = runs[:, 2] - runs[:, 1]
    sample = np.average(np.column_stack(((runs[:, 1]+runs[:, 2])/2, runs[:, 0]+.5)), axis=0, weights=widths)
    sample *= np.array(size) / plane["support"]["size"]
    uv = project(np.linalg.inv(H), sample)
    divisor = np.dot(H[2], [*uv, 1])
    scale = abs((H[1, 1] - size[1]/2 * H[2, 1]) / divisor)
    return float(scale), uv


def choose(charts, proposals, receipts, planes, size):
    anchors = []
    for audit in receipts:
        if audit["picked"] is None:
            continue
        face = audit["offers"][audit["picked"]]
        if face["correlation"] < .80 or (max(face["neighbors"], default=0) < .7 and face["correlation"] < .90):
            continue
        cid = audit["id"]
        quad = np.array(face["quad"])
        unit = fit_homography([0, 0], quad[[3, 2, 1, 0]])
        depth = planes[cid].get("relative_inverse_depth")
        if depth is None:
            continue
        height, _ = height_at_center(unit, planes[cid], size)
        if 4 <= height <= size[1]/3:
            anchors.append(dict(id=cid, depth=depth, height=height, atlas=face["atlas"], quad=quad.tolist()))
    decisions = []
    for chart, proposal in zip(charts, proposals, strict=True):
        cid = chart["id"]
        assert cid == proposal["id"]
        if len(chart["cells"]) < 256 or large_contraction(chart, proposal) or cid not in planes:
            continue
        plane = planes[cid]
        depth = plane.get("relative_inverse_depth")
        if depth is None or abs(plane["normal_camera"][1]) >= .67 or not chart.get("H"):
            continue
        H = np.array(chart["H"])
        old, uv = height_at_center(H, plane, size)
        estimates = []
        peers = [a for a in anchors if a["id"] != cid]
        for i, a in enumerate(peers):
            for b in peers[:i]:
                # The depth field has an arbitrary shift. Fit a slope AND intercept
                # with two independent faces; bound extrapolation to one anchor interval.
                dz = b["depth"] - a["depth"]
                if abs(dz) < .04:
                    continue
                slope = (b["height"] - a["height"]) / dz
                t = (depth - a["depth"]) / dz
                if slope <= 0 or not -1 <= t <= 2:
                    continue
                height = a["height"] + slope * (depth-a["depth"])
                if height > 0:
                    estimates.append(height)
        record = dict(id=cid, old=old, estimates=estimates, peers=[a["id"] for a in peers], take=False)
        if estimates and max(estimates) <= 1.5 * min(estimates) and old > 0:
            ratio = float(np.median(estimates) / old)
            if 2.5 <= ratio <= 8:
                # Preserve an observed chart intersection near the mask center.
                u, v = np.rint(uv)
                unit = H @ np.array([[ratio, 0, u], [0, ratio, v], [0, 0, 1]])
                record.update(take=True, ratio=ratio, H=unit.tolist())
        decisions.append(record)
    return anchors, decisions


def run(job):
    group, case = job
    name = case["id"]
    cv2.setNumThreads(1)
    src, _, packets, _ = handoffs(group)
    rel = Path("sky") / name / "charts.json"
    file = packets / rel
    proposed = finder(group) / "charts" / rel
    receipt = finder(group) / "offers" / (name+".json.gz")
    raw, new, audit = read(file), read(proposed), read(receipt)
    image = Path(case["variants"]["sky"]["image"])
    plane_path = src/"sky"/name/"source-handoff/surfaces.json"
    prior = read(ARCHIVE / group / "rows" / (name+".json"))
    assert digest(file) == new["source_packet_sha256"] == prior["old_packet_sha256"]
    assert digest(proposed) == prior["proposal_packet_sha256"] and digest(receipt) == prior["offers_sha256"]
    assert digest(image) == raw["source_sha256"] == audit["source_sha256"] == prior["source_sha256"]
    assert digest(plane_path) == raw["surfaces_sha256"]
    planes = {p["id"]:p for p in read(plane_path)["planes"]}
    with Image.open(image) as im:
        anchors, decisions = choose(raw["charts"], new["charts"], audit["charts"], planes, im.size)
        takes = {r["id"]:r for r in decisions if r["take"]}
        if takes:
            rgb = np.asarray(im.convert("RGB"))
    baseline = ARCHIVE/group/"large-contraction/assembly"
    dest = baseline
    if takes:
        charts = []
        for chart, offer in zip(raw["charts"], new["charts"], strict=True):
            cid = chart["id"]
            if cid in takes:
                alpha, (x,y,x1,y1) = _native_alpha(planes[cid]["support"], rgb.shape[1::-1])
                mask = np.zeros(rgb.shape[:2],bool)
                mask[y:y1,x:x1] = np.asarray(alpha)>0
                points = np.column_stack(np.nonzero(mask)[::-1])+.5
                measured = resample(chart, np.array(takes[cid]["H"]), mask, rgb, points)
                updated = preserve_and_rank(dict(chart,cells=[]), measured, rgb, mask)
                takes[cid].update(before=len(chart["cells"]), after=len(updated["cells"]))
                chart = dict(updated, confidence=chart["confidence"])
            elif large_contraction(chart, offer):
                chart = offer
            charts.append(chart)
        save(OUT/group/"charts"/rel, dict(raw,charts=charts))
        dest = OUT/group/"assembly"
        with patch.object(combined,"inputs",handoffs):
            combined.run((group,case,OUT/group/"charts",dest))
    row = summarize([case],{"control":baseline,"candidate":dest},PREFIX)["rows"][0]
    assert row["score"]["control"] == prior["score"]["large-contraction"]
    row.update(group=group, layout=case["layout"], split=case["split"], anchors=anchors, decisions=decisions,
               image=str(image), source_sha256=digest(image), surfaces_sha256=digest(plane_path),
               control_pack_sha256=prior["exports"]["large-contraction"]["pack_sha256"],
               reference_sha256=prior["exports"]["large-contraction"]["scoring_reference_sha256"],
               candidate_pack_sha256=read(dest/PREFIX/name/"score.json")["pack_sha256"],
               old_sha256=digest(file), offers_sha256=digest(receipt))
    save(OUT/group/"rows"/(name+".json"),row)
    return row


def aggregate(rows):
    arms = {}
    for arm in ("control","candidate"):
        metrics = {}
        for field in ("geometry","scores"):
            for variant in ("strict","facade"):
                c,m,e = [sum(r["score"][arm][field][variant][k] for r in rows) for k in ("correct","missed","extras")]
                metrics[field+"_"+variant] = dict(correct=c,missed=m,extras=e,f1=2*c/(2*c+m+e))
        arms[arm] = metrics
    return dict(conditions=len(rows), **arms, paired=dict(sum((Counter(r["paired"]["candidate"]) for r in rows),Counter())))


if __name__ == "__main__":
    jobs = [(g,c) for g in (*GROUPS,"holdout5") for c in handoffs(g)[3]]
    assert len(jobs)==92
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(run,jobs))
    save(REVIEW/"conditions.json.gz", rows)
    groups = {"all":rows, "past86":[r for r in rows if r["group"]!="holdout5"],
              "holdout5":[r for r in rows if r["group"]=="holdout5"]}
    for field in ("dataset","split"):
        groups.update({value:[r for r in rows if r[field]==value] for value in {r[field] for r in rows}})
    save(REVIEW/"summary.json",{k:aggregate(v) for k,v in groups.items()})
