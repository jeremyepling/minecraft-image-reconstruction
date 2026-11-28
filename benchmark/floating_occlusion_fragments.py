"""Recover material-supported holes inside an established brick wall grid."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.floating_extra_budget import LOCATION, combined, paths
from benchmark.floating_bridges_report import summarize
from benchmark.floating_feedback_viewer import ISOLATION, SCALE
from benchmark.floating_parallel_depth_audit import faces_at_depth
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

RUN = ROOT / "runs/occlusion-fragments"
REVIEW = ROOT / "benchmark/reviews/floating-occlusion-fragments-01"
REVIEWED = "test-twinspire2-earth-high-right"
PREVIOUS_ROW = WORKTREES / "floating-extra-review/runs/visible-top-row/original/row-with-material"


def source_candidates(chart, plane, rgb):
    cells = {(c["u"], c["v"]): c for c in chart["cells"]}
    brick = {uv for uv, c in cells.items() if c["block_id"].split(":")[-1] == "bricks"}
    anchors = {uv for uv in brick if cells[uv].get("confidence", 0) >= .60}
    if len(anchors) < 3 or len(brick) < .50 * len(cells):
        return [], []
    alpha, (x, y, r, b) = _native_alpha(plane["support"], rgb.shape[1::-1])
    mask = np.asarray(alpha) > 0
    yy, xx = np.indices(mask.shape)
    uv = project(np.linalg.inv(chart["H"]), np.stack((xx + x + .5, yy + y + .5), axis=-1))
    index = np.floor(uv).astype(int)
    fraction = uv - index
    core = np.all((fraction >= .10) & (fraction <= .90), axis=-1)
    crop = rgb[y:b, x:r].astype(float)
    chroma = crop[..., :2] / np.maximum(1., crop.sum(axis=-1, keepdims=True))
    def tile(key):
        return (index[..., 0] == key[0]) & (index[..., 1] == key[1])
    seed = np.zeros(mask.shape, bool)
    for key in anchors:
        seed |= tile(key)
    seed &= core & mask
    if seed.sum() < 64:
        return [], []
    # Colored brick flecks separate the target from both pale bridge and pale mortar.
    red = chroma[..., 0] - chroma[..., 1]
    samples = chroma[seed & (red >= np.median(red[seed]))]
    color = np.median(samples, axis=0)
    if color[0] - color[1] < .035:
        return [], []
    agrees = (abs(chroma - color).max(axis=-1) <= .030) & (red >= .035)
    supported = core & mask & agrees
    keys = np.unique(index[supported], axis=0)
    lower = np.min(list(cells), axis=0)
    upper = np.max(list(cells), axis=0)
    additions, witnesses = [], []
    for i, j in keys:
        key = int(i), int(j)
        if key in cells or np.any(key < lower) or np.any(key > upper):
            continue
        if min(abs(i-u) + abs(j-v) for u, v in brick) > 2:
            continue
        pixels = tile(key) & supported
        count, components, stats, _ = cv2.connectedComponentsWithStats(pixels.astype("uint8"))
        if count < 2:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest, cv2.CC_STAT_AREA] < 8 or pixels.sum() < 12:
            continue
        rows, cols = np.nonzero(pixels)
        additions.append(dict(u=key[0], v=key[1], block_id="minecraft:bricks", fallback=False,
                              confidence=.60, mask_coverage=float(pixels.sum()/tile(key).sum())))
        witnesses.append((rows + y, cols + x))
    return additions, witnesses


def native_check(chart, plane, cells, witnesses, owners, labels):
    material = {b["id"]: b["block_id"] for b in labels["blocks"]}
    by_face = {f["id"]: material[f["block"]] for f in labels["faces"]}
    depth = faces_at_depth(labels)
    alpha, (x, y, r, b) = _native_alpha(plane["support"], owners.shape[::-1])
    ids, counts = np.unique(owners[y:b, x:r][np.asarray(alpha)>0], return_counts=True)
    planes = Counter()
    for fid, n in zip(ids, counts):
        if fid in depth:
            planes[depth[fid]] += int(n)
    principal = planes.most_common(1)[0][0]
    records = []
    for cell, (yy, xx) in zip(cells, witnesses):
        ids, counts = np.unique(owners[yy, xx], return_counts=True)
        brick_pixels = {str(fid):int(n) for fid,n in zip(ids,counts)
                       if depth.get(fid)==principal and by_face.get(fid)=="minecraft:bricks"}
        key = np.array([cell["u"], cell["v"]])
        corners = project(np.asarray(chart["H"]), key + np.array([[0,0],[1,0],[1,1],[0,1]]))
        lo = np.maximum(0, np.floor(corners.min(0)).astype(int))
        hi = np.minimum(owners.shape[::-1], np.ceil(corners.max(0)).astype(int))
        cy, cx = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
        uv = project(np.linalg.inv(chart["H"]), np.stack((cx+.5,cy+.5),axis=-1))
        inside = np.all(np.floor(uv).astype(int) == key, axis=-1)
        ids, counts = np.unique(owners[cy[inside],cx[inside]], return_counts=True)
        footprint = Counter()
        for fid, n in zip(ids, counts):
            category = ("target" if depth.get(fid)==principal and by_face.get(fid)=="minecraft:bricks"
                        else by_face.get(fid,"background"))
            footprint[category] += int(n)
        records.append(dict(u=cell["u"],v=cell["v"],witnesses=len(xx),footprint=footprint,
                            verified_target_pixels=sum(brick_pixels.values()),faces=brick_pixels))
    return dict(wall=chart["id"], plane=principal, cells=records)


def run(job):
    panel, case, base = job
    name=case["id"]
    src=paths(panel)[0][0]
    rel=Path("sky")/name/"charts.json"
    packet=read(base/"charts"/rel)
    source=Path(case["variants"]["sky"]["image"])
    assert digest(source)==case["variants"]["sky"]["source_sha256"]
    rgb=np.asarray(Image.open(source).convert("RGB"))
    shapes={p["id"]:p for p in read(src/"sky"/name/"source-handoff/surfaces.json")["planes"]}
    candidate=[]
    evidence=[]
    for chart in packet["charts"]:
        cells,witnesses=source_candidates(chart,shapes[chart["id"]],rgb)
        candidate.append(dict(chart,cells=chart["cells"]+cells))
        if cells:
            evidence.append((chart,cells,witnesses))
    out=RUN/panel
    save(out/"charts"/rel,dict(packet,charts=candidate))
    combined.run((panel,case,out/"charts",out/"assembly"))
    # All reference data enter only after ordinary export.
    frame=Path(case["frame"])
    with np.load(frame/"labels.npz") as labels:
        owners=labels["face_id"]
    labels=read(frame/"labels.json")
    native=[native_check(c,shapes[c["id"]],added,wit,owners,labels) for c,added,wit in evidence]
    arms=dict(previous=base/"assembly",fragments=out/"assembly")
    score=summarize([case],arms,str(LOCATION))["rows"][0]
    print(name,[(r["wall"],len(r["cells"])) for r in native],score["paired"]["fragments"],flush=True)
    return dict(panel=panel, name=name, source=digest(source), chart=digest(base/"charts"/rel),
                baseline=str(base), native=native, score=score)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    jobs=[]
    for panel in ("original","sky"):
        for case in paths(panel)[0][3]:
            name=case["id"]
            base=(PREVIOUS_ROW if name==REVIEWED else SCALE/panel) if panel=="original" else ISOLATION/panel/"large-contraction"
            if not (base/"charts/sky"/name/"charts.json").exists():
                continue
            jobs.append((panel,case,base))
    with ProcessPoolExecutor(args.workers) as pool:
        results=list(pool.map(run,jobs))
    fixed=[case for panel,case,_ in jobs if panel=="sky"]
    fixed_scores=summarize(fixed,dict(previous=ISOLATION/"sky/large-contraction/assembly",
                                      fragments=RUN/"sky/assembly"),str(LOCATION))["cohorts"]["all"]
    save(REVIEW/"paired.json.gz",dict(results=results,fixed=fixed_scores))
