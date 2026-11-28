"""Apply one proposed peer ruler to the source-only parts for a focused compatibility replay."""
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, read, save
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.expanded_wall_rulers import resample
from benchmark.floating_bridges_report import summarize
from benchmark.floating_verified_wall_expansion import BASE, PREFIX, WORKTREES, combined, data, full_face

RESULTS = Path("runs/peer-split-control")
PEER = WORKTREES/"floating-peer-ruler-refit/runs/peer-ruler-refit/atlas-first"
MIXED = WORKTREES/"floating-mixed-wall-refit/runs/mixed-refit"
FOCUS = (("holdout3","train-stable-yard-1-warm-low-left","mask-0"),
         ("holdout3","train-stable-yard-1-warm-high-right","mask-0"),
         ("sky","train-pillars-01-marble-low-left","mask-1"))


def run(panel,name,wall):
    source = data(panel)
    case = next(c for c in source[3] if c["id"]==name)
    rel = Path("sky")/name/"charts.json"
    base = BASE/panel/"large-contraction"
    peer = PEER/panel/"relaxed"
    split = MIXED/panel/"split"
    labels = MIXED/panel/"source"/name/"region-ids.png"
    if not labels.exists():
        print("No source split handoff",panel,name)
        return
    old = read(base/"charts"/rel)
    packet = read(peer/"charts"/rel)
    split_packet = read(split/"charts"/rel)
    image = Path(case["variants"]["sky"]["image"])
    assert digest(image) == old["source_sha256"] == packet["source_sha256"] == split_packet["source_sha256"]
    rgb = np.asarray(Image.open(image).convert("RGB"))
    with Image.open(labels) as picture:
        owners = np.asarray(picture)
    offered = next(c for c in packet["charts"] if c["id"]==wall)
    h = np.asarray(offered["H"])
    new = []
    verification = []
    for k in sorted(set(np.unique(owners))-{0}):
        mask = owners==k
        uv = np.column_stack(np.nonzero(mask)[::-1]) + .5
        blank = dict(offered,id=f"{wall}~r{k}",cells=[])
        fitted = resample(blank,h,mask,rgb,uv)
        if not fitted["cells"]:
            continue
        fitted = preserve_and_rank(blank,fitted,rgb,mask)
        cells = [(c["u"],c["v"]) for c in fitted["cells"]]
        evidence = [list(key) for key in cells if full_face(h,key,mask,rgb)["ok"]]
        separation = max((max(abs(u-x),abs(v-y)) for u,v in evidence for x,y in evidence),default=0)
        verification.append(dict(id=fitted["id"],visible_cells=len(cells),complete=sorted(evidence),
                                 two_regions=separation>=3))
        new.append(fitted)
    output=RESULTS/panel
    charts=[c for c in packet["charts"] if c["id"]!=wall]+new
    save(output/"charts"/rel,dict(packet,charts=charts))
    with patch.object(combined,"inputs",lambda _:source):
        combined.run((panel,case,output/"charts",output/"assembly"))
    roots={"baseline":base/"assembly","peer-atlas-forced":peer/"assembly",
           "split-prior":split/"assembly","split-with-peer-ruler":output/"assembly"}
    record=summarize([case],roots,str(PREFIX))["rows"][0]
    record.update(panel=panel,wall=wall,source_sha256=digest(image),partition_sha256=digest(labels),
                  peer_chart_sha256=digest(peer/"charts"/rel),verification=verification)
    print(panel,name,[(a,x["geometry"]["strict"]) for a,x in record["score"].items()],verification)
    return record


if __name__ == "__main__":
    cv2.setNumThreads(1)
    records=[run(*row) for row in FOCUS]
    save(RESULTS/"results.json",[row for row in records if row])
