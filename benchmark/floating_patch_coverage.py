"""Count distinct visible block positions reachable at two reference-assisted sampling domains."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark.floating_patch_controls import inputs, native_centers, read, save
from benchmark.floating_patch_report import COHORTS, REVIEW
from engine.surface_perimeter import _native_alpha


def measure(job):
    group,case=job
    source,_,whole,_=inputs(group)
    name=case["id"]
    frame=Path(case["frame"])
    planes={p["id"]:p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"]}
    charts=read(whole/"sky"/name/"charts.json")["charts"]
    labels=read(frame/"labels.json")
    xyz={f["id"]:tuple(f["xyz_world"]) for f in labels["faces"]}
    with np.load(frame/"labels.npz") as arrays:
        owners=arrays["face_id"]
    masks,centers=set(),set()
    for chart in charts:
        if not chart["cells"] or chart["normal_camera"] is None or abs(chart["normal_camera"][1])>=.67:
            continue
        _,fids=native_centers(chart,owners)
        centers.update(xyz[i] for i in set(fids) if i)
        part=planes[chart["id"]]
        alpha,(x,y,x1,y1)=_native_alpha(part["support"],owners.shape[::-1])
        ids=np.unique(owners[y:y1,x:x1][np.asarray(alpha)>0])
        masks.update(xyz[i] for i in ids if i)
    return dict(group=group,id=name,targets=len(set(xyz.values())),sampled=len(centers),in_masks=len(masks),
                in_masks_without_center=len(masks-centers))


if __name__=="__main__":
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(measure,[(g,c) for g in (*COHORTS,"holdout2") for c in inputs(g)[3]]))
    save(REVIEW/"visible-coverage.json",rows)
    for group in (*COHORTS,"holdout2"):
        data=[r for r in rows if r["group"]==group]
        print(group,{k:sum(r[k] for r in data) for k in ("targets","sampled","in_masks","in_masks_without_center")})
