"""Retest visible upper-row proposals against the wall color and its observed upper seam."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, save
from benchmark.floating_visible_row import (
    ISOLATION, LOCATION, SCALE, TESTS, WORKTREES, _native_alpha, combined, paths, project,
    read, summarize,
)
from benchmark.floating_parallel_depth_audit import faces_at_depth

RUN = Path(__file__).resolve().parents[1] / "runs/floating-majority-edge"
REVIEW = Path(__file__).resolve().parent / "reviews/floating-majority-edge"
PRIOR = WORKTREES / "floating-extra-review/runs/visible-top-row"
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def pixels(h, key, shape):
    corners = project(h, UNIT + key)
    low = np.floor(corners.min(axis=0)).astype(int)
    high = np.ceil(corners.max(axis=0)).astype(int)
    y, x = np.mgrid[low[1]:high[1], low[0]:high[0]]
    uv = project(np.linalg.inv(h), np.stack((x+.5, y+.5), axis=-1)) - key
    interior = ((uv >= 0) & (uv < 1)).all(axis=-1)
    inside = interior & (x >= 0) & (y >= 0) & (x < shape[1]) & (y < shape[0])
    return x[inside], y[inside], int(interior.sum())


def axis(h, uv):
    points = project(h, uv + np.array([[0, 0], [1, 0], [0, 1]]))
    lines = points[1:] - points[0]
    a = int(np.argmax(abs(lines[:, 1]) / np.linalg.norm(lines, axis=1)))
    return a, -int(np.sign(lines[a, 1]))


def source_rule(chart, proposals, mask, image):
    h = np.asarray(chart["H"])
    if not proposals:
        return chart, chart, []
    cells = np.array([[p["u"], p["v"]] for p in chart["cells"]])
    a, up = axis(h, np.median(cells, axis=0))
    step = np.eye(2, dtype=int)[a] * up
    records = []
    for cell in proposals:
        key = np.array([cell["u"], cell["v"]])
        joint = key + .5 + step * .5
        # Compare colors across a continuous material edge. Each column uses the
        # visible preceding row and a strip above the proposal as its examples.
        t = np.linspace(-1.40, .55, 79)
        offs = np.linspace(-.38, .38, 17)
        uv = joint + t[:, None, None] * step + offs[None, :, None] * np.eye(2)[1-a]
        xy = np.floor(project(h, uv)).astype(int)
        if (xy < 0).any() or (xy >= image.shape[1::-1]).any():
            continue
        colors = image[xy[..., 1], xy[..., 0]].astype(float)
        lower = np.median(colors[(t >= -1.4) & (t <= -.8)], axis=(0,1))
        upper = np.median(colors[(t >= .1) & (t <= .5)], axis=(0,1))
        direction = upper-lower
        distinct = np.linalg.norm(direction) >= 28
        def side(p):
            return (p-(lower+upper)/2) @ direction < 0
        support = side(colors).mean(axis=1)
        candidates = np.flatnonzero((t>=-.85) & (t<=.15))
        # Choose the change point with the fewest color-side disagreements.
        best = min(candidates, key=lambda i: (1-support[:i]).sum()+support[i:].sum())
        coherent = distinct and support[:best].mean()>.72 and support[best:].mean()<.28
        rec = dict(key=key, cell=cell, joint=joint, seam=float(t[best]), coherent=bool(coherent),
                   lower=lower, upper=upper)
        records.append(rec)
    anchors = [r for r in records if r["coherent"]]
    new_h = h
    if len(anchors)>=3:
        xy = np.array([r["joint"] for r in anchors])
        if np.ptp(xy[:, a])==0 and np.ptp(xy[:, 1-a])>=2:
            slope, offset = np.polyfit(xy[:, 1-a], [r["seam"] for r in anchors], 1)
            residual = [abs(r["seam"]-slope*r["joint"][1-a]-offset) for r in anchors]
            hinge = cells[:, a].min() if up>0 else cells[:, a].max()+1
            span = xy[0, a]-hinge
            if np.median(residual)<.12 and abs(slope)<.12 and abs(span)>=2:
                # Fix height to the seam with the opposite wall edge as hinge.
                mid = np.median(xy[:, 1-a])
                scale = 1 + up*(slope*mid+offset)/span
                transform = np.eye(3)
                transform[a, a] = scale
                transform[a, 1-a] = up*slope
                transform[a, 2] = hinge*(1-scale)-up*slope*mid
                new_h = h @ transform
    charts = []
    for arm, mapping in (("majority", h), ("corrected", new_h)):
        accepted = []
        for r in records:
            x,y,n = pixels(mapping,r["key"],mask.shape)
            separated = ((image[y,x].astype(float)-(r["lower"]+r["upper"])/2) @
                         (r["upper"]-r["lower"])) < 0
            support = mask[y,x] & separated if r["coherent"] else np.zeros_like(x, bool)
            r[arm] = int(support.sum())/max(n,1)
            if r[arm]>.5:
                accepted.append(dict(r["cell"],mask_coverage=r[arm]))
        charts.append(dict(chart,H=mapping.tolist(),cells=chart["cells"]+accepted))
    return charts[0], charts[1], [dict(uv=r["key"].tolist(),seam=r["seam"],coherent=r["coherent"],
                         majority=r["majority"],corrected=r["corrected"]) for r in records]


def truth_data(case, chart, variants, mask, ids, labels, proposals):
    planes = faces_at_depth(labels)
    counts = Counter()
    for k,n in zip(*np.unique(ids[mask],return_counts=True)):
        if k in planes:
            counts[planes[k]] += int(n)
    plane = counts.most_common(1)[0][0]
    materials = {b["id"]: b["block_id"] for b in labels["blocks"]}
    faces = {f["id"]: f for f in labels["faces"]}
    a,up=axis(np.array(chart["H"]),np.median([[p["u"],p["v"]] for p in chart["cells"]],axis=0))
    down=np.eye(2,dtype=int)[a]*up
    original={(p["u"],p["v"]):p for p in chart["cells"]}
    report = {}
    for name,c in variants.items():
        h = np.array(c["H"])
        accepted={(p["u"],p["v"]) for p in c["cells"]}
        rows=[]
        for p in proposals:
            key=(p["u"],p["v"])
            neighbor=original.get(tuple(np.array(key)-down))
            if not neighbor:
                continue
            mat=neighbor["block_id"]
            x,y,total=pixels(h,key,ids.shape)
            right=sum(int(n) for fid,n in zip(*np.unique(ids[y,x],return_counts=True))
                      if planes.get(fid)==plane and materials[faces[fid]["block"]]==mat)
            rows.append(dict(uv=key,material=mat,correct_pixels=right,total=total,accepted=key in accepted))
        target=[f for f in faces.values() if planes.get(f["id"])==plane and all(f["corner_visible"]) and
                np.any(mask & (ids==f["id"])) and np.allclose(np.linalg.norm(np.roll(f["corners_world"],-1,axis=0)-np.array(f["corners_world"]),axis=1),1)]
        phase=np.concatenate([abs((uv:=project(np.linalg.inv(h),np.array(f["corners_px"]))) - uv.round()).ravel() for f in target]) if target else []
        report[name]=dict(rows=rows,corner_median_cells=float(np.median(phase)) if len(phase) else None)
    return report


def run(spec):
    panel,name,wall=spec
    (source,_,_,cases),_,_=paths(panel)
    case=next(c for c in cases if c["id"]==name)
    rel=Path("sky")/name/"charts.json"
    base=SCALE/panel if spec==TESTS[0] else ISOLATION/panel/"large-contraction"
    prior=PRIOR/panel/"row-with-material"
    packet=read(base/"charts"/rel)
    chart=next(c for c in packet["charts"] if c["id"]==wall)
    rev=next(c for c in read(prior/"charts"/rel)["charts"] if c["id"]==wall)
    oldkeys={(p["u"],p["v"]) for p in chart["cells"]}
    proposals=[p for p in rev["cells"] if (p["u"],p["v"]) not in oldkeys]
    imagepath=Path(case["variants"]["sky"]["image"])
    image=np.array(Image.open(imagepath).convert("RGB"))
    plane=next(p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
    alpha,(x,y,r,b)=_native_alpha(plane["support"],image.shape[1::-1])
    mask=np.zeros(image.shape[:2],bool)
    mask[y:b,x:r]=np.array(alpha)>0
    majority,seam,records=source_rule(chart,proposals,mask,image)
    arms={"previous":base/"assembly","prior":prior/"assembly"}
    for arm,new in (("majority",majority),("seam",seam)):
        out=RUN/panel/arm
        save(out/"charts"/rel,dict(packet,charts=[new if c["id"]==wall else c for c in packet["charts"]]))
        with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
            combined.run((panel,case,out/"charts",out/"assembly"))
        arms[arm]=out/"assembly"
    frame=Path(case["frame"])
    with np.load(frame/"labels.npz") as d:
        ids=d["face_id"]
    truth=truth_data(case,chart,dict(prior=rev,majority=majority,seam=seam),mask,ids,read(frame/"labels.json"),proposals)
    summary=summarize([case],arms,str(LOCATION))
    out=dict(id=name,wall=wall,panel=panel,image_sha256=digest(imagepath),previous_packet=digest(base/"charts"/rel),
             prior_packet=digest(prior/"charts"/rel),records=records,truth=truth,decoded=summary)
    save(REVIEW/(name+".json.gz"),out)
    print(name,len(proposals),len(majority["cells"])-len(chart["cells"]),len(seam["cells"])-len(chart["cells"]),flush=True)


if __name__=="__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(run,TESTS))


