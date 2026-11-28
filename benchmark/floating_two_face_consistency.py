"""Test agreement between separately located accepted source-image faces on the same mask."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
import argparse

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_one_face_control import (
    ROOT,WORKTREES,REVIEW,OUT,PANELS,UNIT,inputs,base,project,combined,extend,read,save,digest,summarize,_native_alpha)


def corrections(pick,offers,h):
    # Source proposals order corners from the opposite image-row direction to the chart.
    inverse=np.linalg.inv(h)
    candidates=[]
    for index,item in enumerate(offers):
        if index==pick or not item["accept"]:
            continue
        quad=np.asarray(item["quad"])[[3,2,1,0]]
        uv=project(inverse,quad)
        if not np.isfinite(uv).all():
            continue
        offset=np.rint((uv-UNIT).mean(0))
        span=np.linalg.norm(offset)
        if span<2 or np.max(np.linalg.norm(uv-UNIT-offset,axis=1))>.30:
            continue
        candidates.append((span,item["score"],index,offset,uv))
    if not candidates:
        return None
    _,_,index,off,found=max(candidates,key=lambda t:t[:2])
    local=np.vstack((UNIT,UNIT+off))
    measured=np.vstack((UNIT,found))
    t=np.eye(3)
    t[:2]=np.linalg.lstsq(np.column_stack((local,np.ones(8))),measured,rcond=None)[0].T
    return h@t,index


def run(job):
    group,case=job
    cv2.setNumThreads(1)
    src,_,whole,_=inputs(group)
    key={"holdout3":"new6","holdout4":"new7"}.get(group,group)
    parent=WORKTREES/"floating-face-proposals/runs/face-proposals"/key
    name=case["id"]
    rel=Path("sky")/name/"charts.json"
    raw=read(whole/rel)
    proposed=read(parent/"charts"/rel)
    assert proposed["source_packet_sha256"]==digest(whole/rel)
    source=Path(case["variants"]["sky"]["image"])
    rgb=np.asarray(Image.open(source).convert("RGB"))
    receipts=read(parent/"offers"/(name+".json.gz"))
    assert receipts["source_sha256"]==digest(source)
    evidence={r["id"]:r for r in receipts["charts"]}
    parts={p["id"]:p for p in read(src/"sky"/name/"source-handoff/surfaces.json")["planes"]}
    choices={"pair-required":[],"pair-refined":[]}
    accepted=[]
    for old,one in zip(raw["charts"],proposed["charts"],strict=True):
        pid=old["id"]
        assert one["id"]==pid
        row=evidence.get(pid,{})
        trial=corrections(row["picked"],row["offers"],np.asarray(one["H"])) if row.get("picked") is not None else None
        if trial is None:
            choices["pair-required"].append(old)
            choices["pair-refined"].append(one)
            continue
        h,index=trial
        alpha,(x,y,x1,y1)=_native_alpha(parts[pid]["support"],rgb.shape[1::-1])
        mask=np.zeros(rgb.shape[:2],bool)
        mask[y:y1,x:x1]=np.asarray(alpha)>0
        revised=extend(one,h,mask,rgb)
        choices["pair-required"].append(revised)
        choices["pair-refined"].append(revised)
        accepted.append(dict(id=pid,first=row["picked"],second=index))
    for arm,charts in choices.items():
        folder=OUT/group/arm
        save(folder/"charts"/rel,dict(raw,charts=charts,proposal_packet=digest(parent/"charts"/rel)))
        with patch.object(combined,"inputs",inputs):
            combined.run((group,case,folder/"charts",folder/"assembly"))
    save(OUT/group/"pair-audit"/(name+".json"),dict(id=name,accepted=accepted))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("groups",nargs="*",default=(*PANELS,"holdout3"))
    groups=p.parse_args().groups
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(run,[(g,c) for g in groups for c in inputs(g)[3]]))
    scores={}
    for g in groups:
        key={"holdout3":"new6","holdout4":"new7"}.get(g,g)
        source=WORKTREES/"floating-face-proposals/runs/face-proposals"/key
        roots={"horizontal":base(g),"source-one":source/"assembly",**{k:OUT/g/k/"assembly" for k in ("pair-required","pair-refined")}}
        scores[g]=summarize(inputs(g)[3],roots,"connected-atlas/sky")
        print(g,{k:round(v["geometry_strict"]["f1"]*100,2) for k,v in scores[g]["cohorts"]["all"].items()})
    save(REVIEW/("pairs-"+"-".join(groups)+".json.gz"),scores)
