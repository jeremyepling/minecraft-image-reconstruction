"""Check which Twinspire brick faces are covered by the reviewed B grid and its parent mask."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from engine.geometry import project
from engine.surface_perimeter import _native_alpha
from benchmark.floating_feedback_viewer import SCALE
from benchmark.floating_parallel_depth_audit import OUT, paths, read, save, faces_at_depth


def run():
    panel, name, wall = "original", "test-twinspire2-earth-high-right", "mask-2"
    (root,_,_,cases),_,_=paths(panel)
    frame=Path(next(c for c in cases if c["id"]==name)["frame"])
    labels=read(frame/"labels.json")
    by_id=faces_at_depth(labels)
    with np.load(frame/"labels.npz") as d:
        owners=d["face_id"]
    plane=next(p for p in read(root/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
    alpha,(x,y,r,b)=_native_alpha(plane["support"],owners.shape[::-1])
    mask=np.zeros_like(owners,bool)
    mask[y:b,x:r]=np.asarray(alpha)>0
    ids,count=np.unique(owners[mask],return_counts=True)
    votes=Counter()
    for fid,n in zip(ids,count):
        if fid in by_id:
            votes[by_id[fid]]+=int(n)
    principal=votes.most_common(1)[0][0]
    chart=next(c for c in read(SCALE/panel/"charts/sky"/name/"charts.json")["charts"] if c["id"]==wall)
    uv={(c["u"],c["v"]) for c in chart["cells"]}
    inv=np.linalg.inv(chart["H"])
    blocks={b["id"]:b for b in labels["blocks"]}
    rows=defaultdict(Counter)
    face_rows={}
    for f in labels["faces"]:
        fid=f["id"]
        if by_id.get(fid)!=principal:
            continue
        if blocks[f["block"]]["block_id"]!="minecraft:bricks":
            continue
        # Only the annotated wall's crop. Unrelated visible islands may share a plane.
        crop=owners[y:b,x:r]==fid
        if not crop.any():
            continue
        yy,xx=np.nonzero(crop)
        pix=np.column_stack((xx+x+.5,yy+y+.5))
        keys=np.floor(project(inv,pix)).astype(int)
        covered=np.array([tuple(k) in uv for k in keys])
        supported=mask[yy+y,xx+x]
        level=f["xyz_world"][1]
        rows[level].update(faces=1,pixels=len(xx),inside_mask=int(supported.sum()),under_grid=int(covered.sum()))
        face_rows[fid]=dict(level=level,pixels=len(xx),mask=int(supported.sum()),grid=int(covered.sum()))
    save(OUT/"twinspire-rows.json",dict(id=name,wall=wall,plane=principal,rows=rows,faces=face_rows))
    print(sorted(rows.items(),reverse=True))


if __name__=="__main__":
    run()
