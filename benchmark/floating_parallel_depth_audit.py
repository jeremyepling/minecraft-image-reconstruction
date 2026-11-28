"""Measure same-facing, depth-separated visible planes inside frozen floating masks."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
from engine.surface_perimeter import _native_alpha
from benchmark.floating_extra_budget import paths, PANELS, read, WORKTREES
from benchmark.expanded_baseline import save

OUT = Path(__file__).resolve().parent/"reviews/floating-parts-feedback-02"
GROUPS = (*PANELS, "holdout3", "holdout4", "holdout5")


def faces_at_depth(labels):
    by_id = {}
    for face in labels["faces"]:
        normal = np.asarray(face["normal_world"])
        distances = np.asarray(face["corners_world"]) @ normal
        if np.ptp(distances) < 1e-5:
            by_id[face["id"]] = (face["direction"], round(float(distances[0]), 5))
    return by_id


def partition(mask, owners, by_id):
    ids, weights = np.unique(owners[mask], return_counts=True)
    counts = {}
    for face_id, weight in zip(ids, weights):
        if face_id in by_id:
            key = by_id[face_id]
            counts[key] = counts.get(key, 0) + int(weight)
    # Use the direction with the strongest secondary plane. Compare with the largest
    # plane of that same direction. Coplanar disjoint islands do not count.
    best = None
    for direction in ("north","south","east","west","up","down"):
        ordered = sorted(((w,d) for (n,d),w in counts.items() if n==direction), reverse=True)
        if not ordered:
            continue
        first, distance = ordered[0]
        second = next(((w,d) for w,d in ordered[1:] if abs(d-distance)>=.5), None)
        if second and (best is None or second[0] > best["secondary_pixels"]):
            best = dict(direction=direction,primary_distance=distance,secondary_distance=second[1],
                        primary_pixels=first,secondary_pixels=second[0])
    return best


def run(job):
    panel,case=job
    name=case["id"]
    src=paths(panel)[0][0]
    frame=Path(case["frame"])
    by_id=faces_at_depth(read(frame/"labels.json"))
    with np.load(frame/"labels.npz") as data:
        owners=data["face_id"]
    masks=read(src/"sky"/name/"source-handoff/surfaces.json")["planes"]
    results=[]
    for plane in masks:
        alpha,(x,y,r,b)=_native_alpha(plane["support"],owners.shape[::-1])
        mask=np.asarray(alpha)>0
        pixels=int(mask.sum())
        pair=partition(mask,owners[y:b,x:r],by_id)
        flag=bool(pair and pair["secondary_pixels"]>=max(64,.05*pixels))
        results.append(dict(id=plane["id"],pixels=pixels,pair=pair,flag=flag))
    return dict(panel=panel,id=name,layout=case["layout"],masks=results)


if __name__=="__main__":
    jobs=[(p,c) for p in GROUPS for c in paths(p)[0][3]]
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(run,jobs))
    save(OUT/"parallel-depth-audit.json.gz",rows)
    n=sum(len(r["masks"]) for r in rows)
    flagged=[(r,m) for r in rows for m in r["masks"] if m["flag"]]
    print(len(rows),n,len(flagged),len({(r["panel"],r["id"]) for r,m in flagged}))
