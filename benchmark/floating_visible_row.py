"""Offer a single, visibly supported missing top row along a measured wall grid."""
from itertools import groupby
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from benchmark.floating_extra_feedback import grid, _native_alpha, LOCATION, paths, WORKTREES, read
from benchmark.floating_feedback_viewer import ISOLATION, SCALE
from benchmark.floating_extra_budget import combined
from benchmark.floating_bridges_report import summarize
from benchmark.expanded_wall_source_resample import TEXELS, atlas, rank_tile, project
from benchmark.floating_parallel_depth_audit import OUT, save

RUN=Path("runs/visible-top-row")
TESTS=[("original","test-twinspire2-earth-high-right","mask-2"),("holdout4","train-riverside-mill-1-warm-low-left","mask-0"),
       ("holdout2","train-shrine-02-ochre-low-left","mask-0-wall-0"),("sky","validation-hillside-chapel-1-warm-low-left","mask-0"),
       ("holdout3","train-sawtooth2-cool-high-right","mask-0-wall-1")]


def extend(chart,mask,rgb,min_correlation):
    h=np.asarray(chart["H"])
    cells=np.asarray([(c["u"],c["v"]) for c in chart["cells"]])
    center=np.median(cells,axis=0)+.5
    local=project(h,center+np.array([[0,0],[1,0],[0,1]]))
    axes=local[1:]-local[:1]
    vertical=int(np.argmax(abs(axes[:,1])/np.maximum(1e-9,np.linalg.norm(axes,axis=1))))
    up=-int(np.sign(axes[vertical,1]))
    if not up:
        return chart,[]
    proposed=[]
    for column in np.unique(cells[:,1-vertical]):
        subset=cells[cells[:,1-vertical]==column]
        key=subset[np.argmax(subset[:,vertical]*up)].copy()
        key[vertical]+=up
        xy=np.floor(project(h,TEXELS+key)).astype(int)
        x,y=xy[...,0],xy[...,1]
        inside=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
        x=x.clip(0,mask.shape[1]-1)
        y=y.clip(0,mask.shape[0]-1)
        visible=inside & mask[y,x]
        coverage=float(visible.mean())
        if coverage<.35:
            continue
        ranked=rank_tile(dict(rgb=rgb[y,x],mask=visible),atlas())
        if ranked["block_id"] and ranked["evidence"]["correlation"]>=min_correlation:
            proposed.append((int(column),dict(u=int(key[0]),v=int(key[1]),block_id=ranked["block_id"],
                            fallback=False,confidence=ranked["evidence"]["correlation"],mask_coverage=coverage)))
    accepted=[]
    for _,seq in groupby(enumerate(proposed),key=lambda pair:(pair[1][0]-pair[0],pair[1][1][("u","v")[vertical]])):
        strip=[row[1][1] for row in seq]
        if len(strip)>=3:
            accepted.extend(strip)
    return dict(chart,cells=chart["cells"]+accepted),accepted


def run():
    results={}
    for panel,name,wall in TESTS:
        (source,_,_,cases),_,_=paths(panel)
        case=next(c for c in cases if c["id"]==name)
        rel=Path("sky")/name/"charts.json"
        a=SCALE/panel if name=="test-twinspire2-earth-high-right" else ISOLATION/panel/"large-contraction"
        packet=read(a/"charts"/rel)
        chart=next(c for c in packet["charts"] if c["id"]==wall)
        with Image.open(case["variants"]["sky"]["image"]) as im:
            rgb=np.array(im.convert("RGB"))
        plane=next(p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
        alpha,(x,y,r,b)=_native_alpha(plane["support"],rgb.shape[1::-1])
        mask=np.zeros(rgb.shape[:2],bool)
        mask[y:b,x:r]=np.asarray(alpha)>0
        arms={"previous":a/"assembly"}
        additions={}
        for key,threshold in (("top-row",.65),("row-with-material", -1.)):
            new,cells=extend(chart,mask,rgb,threshold)
            out=RUN/panel/key
            save(out/"charts"/rel,dict(packet,charts=[new if c["id"]==wall else c for c in packet["charts"]]))
            with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
                combined.run((panel,case,out/"charts",out/"assembly"))
            arms[key]=out/"assembly"
            additions[key]=cells
        stats=summarize([case],arms,str(LOCATION))["rows"][0]
        results[name]=dict(cells=additions,stats=stats)
        print(name,{k:len(v) for k,v in additions.items()},stats["paired"],flush=True)
    save(OUT/"top-row.json.gz",results)


if __name__=="__main__":
    run()
