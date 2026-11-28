"""Replay a source-only challenger through the existing horizontal-face assembler."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import cv2
import numpy as np

from benchmark.floating_patch_controls import inputs,OUT,WORKTREES,ROOT,read,save,paths,assembly,expanded_scores
from benchmark.floating_horizontal_bridges import solver,horizontal_charts
from benchmark.floating_patch_report import COHORTS,baseline,collect,write


def run(job):
    group,case,charts,dest=job
    cv2.setNumThreads(1)
    name=case["id"]
    source=inputs(group)[0]
    planes=read(source/"sky"/name/"source-handoff/surfaces.json")["planes"]
    image=Path(case["variants"]["sky"]["image"])
    rgb=np.asarray(Image.open(image).convert("RGB"))
    orient=solver.oriented
    def with_tiles(charts):
        walls,hint=orient(charts)
        return walls+horizontal_charts(walls,planes,rgb.shape[1::-1],rgb),hint
    def checked_read(path):
        data=read(path)
        if case["dataset"]=="visible-builds-v2" and path==Path(case["frame"])/"frame.json" and "framing" not in data:
            audit=read(ROOT/"training/framing-audit-v2.json")["frames"]
            entry=next(r for r in audit if r["id"]==name)
            assert entry["accepted"] and entry["rgb_sha256"]==case["rgb_sha256"]
            data=dict(data,framing=entry)
        return data
    mode=(charts,"charts.json",False,1,2.5,.5,False)
    with patch.object(solver,"oriented",with_tiles),patch.object(paths,"RUN",source),patch.object(
        assembly,"ASSEMBLY",dest),patch.dict(assembly.MODES,{"connected-atlas":mode}),patch.object(expanded_scores,"read",checked_read):
        assembly.evaluate((case,"sky","connected-atlas"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("chart_pattern",help="Absolute chart root with {group}, using --holdout-name for the sixth-view panel")
    p.add_argument("--holdout-name",default="new")
    a=p.parse_args()
    jobs=[]
    for g in COHORTS:
        charts=Path(a.chart_pattern.format(group=g if g!="holdout" else a.holdout_name))
        jobs.extend((g,c,charts,OUT/g/a.name/"assembly") for c in inputs(g)[3])
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(run,jobs))
    write(a.name,{g:collect(g,{"horizontal":baseline(g,True),a.name:OUT/g/a.name/"assembly"}) for g in COHORTS})
