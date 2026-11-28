"""Choose a wall unit by comparing native whole-sprite fits at the same screen location."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import benchmark
import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.full_block_lattice import UNIT, verify_sprite
from engine.geometry import project

# Archived harness and its immutable chart paths are maintained in the isolation worktree.
benchmark.__path__.append(str(WORKTREES / "floating-extra-isolation/benchmark"))
from benchmark.floating_extra_isolation import GROUPS, PREFIX, combined, finder, handoffs, summarize

OUT = ROOT / "runs/feedback-scale"
REVIEW = ROOT / "benchmark/reviews/floating-feedback-scale-01"
ARCHIVE = WORKTREES / "floating-extra-isolation/runs/extra-isolation"
REVIEWED = {"amphitheater-01", "hillside-chapel-1", "riverside-mill-1", "sawtooth2", "twinspire2"}


def choose(rgb, old, audit):
    if audit is None or audit["picked"] is None:
        return dict(take=False, reason="no accepted proposal")
    offer = audit["offers"][int(audit["picked"])]
    quad = np.asarray(offer["quad"])
    H = np.asarray(old["H"])
    # Express the new face in old chart coordinates so perspective does not change the scale ratio.
    old_units = project(np.linalg.inv(H), quad)
    area = abs(cv2.contourArea(old_units.astype("float32")))
    corr = offer["correlation"]
    second = max(offer["neighbors"], default=0.)
    evidence = dict(old_units_area=area, full_face=corr, adjacent_face=second)
    if area < 2 or second < .70:
        return dict(**evidence, take=False, reason="scale or adjacent face unsupported")
    # Sample five existing cells under the proposed face. Requiring an entire old
    # cell on screen and in the expanded chart avoids counting unseen fragments.
    samples = np.array([[.5,.5],[.25,.25],[.75,.25],[.75,.75],[.25,.75]])
    top = quad[0] + samples[:,0,None] * (quad[1] - quad[0])
    bottom = quad[3] + samples[:,0,None] * (quad[2] - quad[3])
    pixels = top + samples[:,1,None] * (bottom - top)
    keys = np.unique(np.floor(project(np.linalg.inv(H), pixels)).astype(int), axis=0)
    supported = {(c["u"], c["v"]) for c in old["cells"] if c.get("mask_coverage", 1) >= .95}
    scores = []
    for key in keys:
        if tuple(key) not in supported:
            continue
        corners = project(H, UNIT + key)
        if np.any(corners < 0) or np.any(corners >= rgb.shape[1::-1]):
            continue
        result = verify_sprite(rgb, corners)
        scores.append(result[2] if result else 0.)
    # A strong existing tile at this location is evidence against replacing the ruler.
    take = bool(scores and corr >= max(scores) + .08)
    return dict(**evidence, old_correlations=scores, take=take, reason="paired atlas test")


def run(job):
    group, case = job
    name = case["id"]
    rel = Path("sky") / name / "charts.json"
    src = handoffs(group)[2] / rel
    offered = finder(group) / "charts" / rel
    receipt = finder(group) / "offers" / (name + ".json.gz")
    old, new, audit = read(src), read(offered), read(receipt)
    image = Path(case["variants"]["sky"]["image"])
    assert digest(image) == old["source_sha256"] == audit["source_sha256"]
    assert digest(src) == new["source_packet_sha256"]
    rgb = np.asarray(Image.open(image).convert("RGB"))
    offers = {r["id"]: r for r in audit["charts"]}
    decisions = {c["id"]: choose(rgb,c,offers.get(c["id"])) for c in old["charts"]}
    picks = sorted(cid for cid, r in decisions.items() if r["take"])
    prior = read(ARCHIVE / group / "rows" / (name+".json"))
    assert prior["old_packet_sha256"] == digest(src) and prior["offers_sha256"] == digest(receipt)
    assert prior["proposal_packet_sha256"] == digest(offered)
    prev_root = ARCHIVE / group / "large-contraction/assembly"
    dest = prev_root
    if picks != prior["selected"]:
        dest = OUT / group / "assembly"
        charts = []
        for a,b in zip(old["charts"],new["charts"],strict=True):
            assert a["id"] == b["id"]
            charts.append(b if a["id"] in picks else a)
        save(OUT / group / "charts" / rel, dict(old, charts=charts))
        with patch.object(combined,"inputs",handoffs):
            combined.run((group,case,OUT / group / "charts",dest))
    roots = {"large-contraction":prev_root, "paired-atlas":dest}
    # Control for vetoing the earlier gate. These two substitution sets are nested
    # on every measured condition, so the union is already an exact assembled pack.
    union = sorted(set(prior["selected"]) | set(picks))
    if union == picks:
        roots["add-atlas"] = dest
    elif union == prior["selected"]:
        roots["add-atlas"] = prev_root
    else:
        folder = OUT / group / "add-atlas"
        charts = [b if a["id"] in union else a for a,b in zip(old["charts"],new["charts"],strict=True)]
        save(folder / "charts" / rel, dict(old,charts=charts))
        with patch.object(combined,"inputs",handoffs):
            combined.run((group,case,folder / "charts",folder / "assembly"))
        roots["add-atlas"] = folder / "assembly"
    row = summarize([case],roots,PREFIX)["rows"][0]
    assert row["score"]["large-contraction"] == prior["score"]["large-contraction"]
    row.update(panel=group, layout=case["layout"], picks=picks, previous=prior["selected"],
               decisions=decisions, source_sha256=digest(image), offers_sha256=digest(receipt),
               old_sha256=digest(src), new_sha256=digest(offered),
               pack_sha256=read(dest/PREFIX/name/"score.json")["pack_sha256"])
    save(OUT/group/"rows"/(name+".json"),row)
    return row


def aggregate(rows):
    out = {}
    for arm in ("large-contraction","paired-atlas","add-atlas"):
        out[arm] = {}
        for field in ("geometry","scores"):
            for metric in ("strict","facade"):
                c,m,e = [sum(r["score"][arm][field][metric][k] for r in rows) for k in ("correct","missed","extras")]
                out[arm][field+"_"+metric] = dict(correct=c,missed=m,extras=e,f1=2*c/(2*c+m+e))
    for arm in ("paired-atlas","add-atlas"):
        out[arm]["paired"] = {k:sum(r["paired"][arm][k] for r in rows)
                              for k in ("gained","lost","added","removed")}
    return out


if __name__ == "__main__":
    jobs = [(g,c) for g in (*GROUPS,"holdout5") for c in handoffs(g)[3]]
    assert len(jobs)==92
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(run,jobs))
    save(REVIEW/"conditions.json.gz",rows)
    partitions = {"all":rows, "reviewed":[r for r in rows if r["layout"] in REVIEWED],
                  "others":[r for r in rows if r["layout"] not in REVIEWED]}
    save(REVIEW/"summary.json",{k:dict(conditions=len(v),**aggregate(v)) for k,v in partitions.items()})
