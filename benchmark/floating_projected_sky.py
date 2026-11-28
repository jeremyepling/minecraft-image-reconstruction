"""Test rejecting exported cells whose projected footprint is mostly source sky."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_depth_refit import (EXTENDED, FLAT, PRIOR, SAMPLES, LOCATION, PANELS,
    ROOT, WORKTREES, combined, digest, paths, project, read, save, summarize, totals)
from engine.floating_sky import sky_support
from engine.sky_filter import sky_masks

PREVIOUS = WORKTREES / "floating-depth-refit"
OUT = ROOT / "runs/projected-sky"
REVIEW = ROOT / "benchmark/reviews/floating-projected-sky-01"


def filtered(chart, patches, masks):
    result = [[] for _ in masks]
    removed = [[] for _ in masks]
    groups = {}
    for cell in chart.get("export_cells", chart["cells"]):
        uv = (cell["u"], cell["v"])
        depth = cell.get("normal_offset", 0)
        fits = [h for sign, points, h in patches if depth == sign and uv in points]
        if len(fits) > 1:
            for kept in result:
                kept.append(cell)
            continue
        h = np.array(fits[0] if fits else chart["H"], dtype=float)
        groups.setdefault(tuple(h.ravel()), []).append(cell)
    for matrix, cells in groups.items():
        uv = np.array([[c["u"], c["v"]] for c in cells])
        xy = project(np.array(matrix).reshape(3, 3), uv[:,None,None,:] + SAMPLES)
        xy = np.floor(xy).astype(int)
        x, y = xy[...,0], xy[...,1]
        height, width = masks[0].shape
        inside = (x >= 0) & (y >= 0) & (x < width) & (y < height)
        for mask, kept, drops in zip(masks, result, removed):
            vote = mask[y.clip(0,height-1), x.clip(0,width-1)] & inside
            for cell, count in zip(cells, vote.sum(axis=(1,2))):
                if count > SAMPLES.shape[0]*SAMPLES.shape[1]/2:
                    drops.append([cell["u"], cell["v"], cell.get("normal_offset",0), int(count)])
                else:
                    kept.append(cell)
    # Retain the original cell order for material and overlap tie breaking.
    ordered = chart.get("export_cells", chart["cells"])
    selected = [{id(c) for c in kept} for kept in result]
    return [[c for c in ordered if id(c) in keep] for keep in selected], removed


def run(job):
    panel, case, row = job
    cv2.setNumThreads(1)
    name = case["id"]
    rel = Path("sky") / name / "charts.json"
    roots = [PREVIOUS/"runs/depth-refit"/panel/a for a in (EXTENDED,"source-refit")]
    roots.extend([PRIOR/panel/"source", FLAT/panel/"large-contraction"])
    base = next(root for root in roots if (root/"charts"/rel).exists())
    packet = read(base/"charts"/rel)
    image = Path(case["variants"]["sky"]["image"])
    assert digest(image) == packet["source_sha256"] == row["source_sha256"]
    receipt = read(base/"assembly"/LOCATION/name/"score.json")
    assert receipt["pack_sha256"] == row["pack_sha256"][EXTENDED]
    with Image.open(image) as rgb:
        border = sky_support(rgb)
        clear, tinted = sky_masks(rgb)
    arms = ("border", "border-and-cloud")
    charts = {arm:[] for arm in arms}
    drops = {arm:{} for arm in arms}
    for chart in packet["charts"]:
        patches = []
        for p in row["patches"] + row["extra_patches"]:
            if p["chart"] == chart["id"]:
                fit = p.get("source-refit", p)
                patches.append((p["depth"],set(map(tuple,fit["uv"])),fit["H"]))
        kept, removed = filtered(chart, patches, (border, border | clear | tinted))
        for arm, cells, rejected in zip(arms, kept, removed):
            charts[arm].append(dict(chart, export_cells=cells))
            if rejected:
                drops[arm][chart["id"]] = rejected
    outputs = {"prior": base/"assembly"}
    for arm in arms:
        outputs[arm] = base/"assembly"
        if drops[arm]:
            dest = OUT/panel/arm
            save(dest/"charts"/rel,dict(packet,charts=charts[arm]))
            with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
                combined.run((panel,case,dest/"charts",dest/"assembly"))
            outputs[arm] = dest/"assembly"
    result = summarize([case],outputs,str(LOCATION))["rows"][0]
    assert result["score"]["prior"] == row["score"][EXTENDED]
    result.update(panel=panel, drops=drops)
    return result


if __name__ == "__main__":
    archive = {(r["panel"],r["id"]):r for r in read(PREVIOUS/"benchmark/reviews/floating-depth-refit-01/conditions.json.gz") if r["active"]}
    jobs = [(p,c,archive[p,c["id"]]) for p in PANELS for c in paths(p)[0][3] if (p,c["id"]) in archive]
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(run,jobs))
    assert len(rows) == 84
    save(REVIEW/"conditions.json.gz",rows)
    save(REVIEW/"summary.json",{"active":totals(rows),"sky":totals([r for r in rows if r["panel"]=="sky"])})
