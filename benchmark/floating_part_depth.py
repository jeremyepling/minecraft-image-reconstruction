"""Measure parallel relief inside one chart, keeping its projected grid and ordinary joins."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import NATIVE
from benchmark.floating_extra_budget import LOCATION, combined, paths
from benchmark.floating_bridges_report import summarize
from benchmark.relative_face_scoring import yaw_from_anchor_normal
from engine.geometry import bilinear, project
from engine.surface_fields import MODELS
from engine.surface_perimeter import _native_alpha

OUT = ROOT / "runs/part-depth"
REVIEW = ROOT / "benchmark/reviews/floating-part-depth-01"
PANELS = ("holdout4","holdout2","sky","original","fresh","supplemental","holdout","holdout3","holdout5")
SAMPLES = np.stack(np.meshgrid([.25,.5,.75],[.25,.5,.75]),-1).reshape(-1,2)


def fields(panel, case, source):
    group = Path(case["dataset"]) / case["id"]
    if panel == "sky":
        directory = combined.assembly.SHARED / "baseline/sky" / group / "handoffs"
    elif panel == "original":
        archived = WORKTREES / "full-expanded-controls/runs/expanded-baseline-01" / group / "handoffs"
        directory = archived if archived.is_dir() else NATIVE.parent / "fields" / group
    else:
        directory = source.parent / "fields" / group
    files = {k:directory/(k+".npz") for k in MODELS}
    sha = {k:digest(p) for k,p in files.items()}
    receipt = directory / "receipt.json"
    if not receipt.exists():
        receipt = directory.parent / "receipt.json"
    if receipt.exists():
        rec = read(receipt)
        assert rec["source_sha256"] == case["variants"]["sky"]["source_sha256"]
        assert all(rec["model_sha256"][k] == MODELS[k][2] for k in MODELS)
        if "fields_sha256" in rec:
            assert all(rec["fields_sha256"][k] == s for k,s in sha.items())
    values = {}
    for kind,path in files.items():
        with np.load(path) as data:
            values[kind] = data["value"]
    return values,sha


def components(active, uv):
    remaining = {tuple(uv[i]): i for i in np.flatnonzero(active)}
    while remaining:
        pos,i = remaining.popitem()
        piece,queue = [i],[pos]
        while queue:
            x,y = queue.pop()
            for neighbor in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if neighbor in remaining:
                    piece.append(remaining.pop(neighbor))
                    queue.append(neighbor)
        yield np.asarray(piece,int)


def image_offsets(chart, uv, pixels, valid, field, size):
    """Fit the common inverse-depth slope within contiguous, similarly directed cells."""
    normal = field["normal"]
    depth = field["depth"]
    n = np.median(bilinear(normal, pixels*normal.shape[1::-1]/size),axis=1)
    n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-8)
    d = np.median(bilinear(depth[:,:,None],pixels*depth.shape[1::-1]/size)[:,:,0],axis=1)
    xy = pixels[:,4]
    valid = valid & (n@np.asarray(chart["normal_camera"])>=.90) & np.isfinite(d)
    offset = np.zeros(len(uv),int)
    ev = dict(eligible=int(valid.sum()), reason="insufficient-support")
    if valid.sum()<12:
        return offset,ev
    lookup = {tuple(uv[i]):i for i in np.flatnonzero(valid)}
    links = np.asarray([(i,lookup[p]) for (x,y),i in lookup.items()
                        for p in ((x+1,y),(x,y+1)) if p in lookup],int)
    if len(links)<12:
        return offset,ev
    a,b = links.T
    dx,dd = xy[b]-xy[a],d[b]-d[a]
    kept = np.ones(len(dx),bool)
    for _ in range(3):
        slope,_,rank,_ = np.linalg.lstsq(dx[kept],dd[kept],rcond=None)
        if rank<2:
            return offset,ev
        residual = np.abs(dd-dx@slope)
        kept = residual<=np.quantile(residual,.7)
    residual = d-(xy-xy[valid].mean(0))@slope
    ordered = np.sort(residual[valid])
    centers = np.quantile(ordered,[.15,.85])
    for _ in range(12):
        high = ordered>centers.mean()
        if high.all() or not high.any():
            return offset,ev
        centers = np.array([np.median(ordered[~high]),np.median(ordered[high])])
    high = residual>centers.mean()
    population = [int((valid & (high==level)).sum()) for level in (False,True)]
    majority = bool(population[1]>population[0])
    gap = centers[1]-centers[0]
    spread = max(np.median(abs(ordered[ordered<=centers.mean()]-centers[0])),
                 np.median(abs(ordered[ordered>centers.mean()]-centers[1])))
    noise = np.median(abs(dd-dx@slope))
    ev.update(reason="ambiguous-bands", populations=population,gap=float(gap),spread=float(spread),noise=float(noise))
    if gap<=max(4*spread,5*noise,.025*abs(np.median(d[valid]))):
        return offset,ev
    sign = -1 if majority else 1
    marked = valid & (high!=majority)
    for piece in components(marked,uv):
        if len(piece)<4:
            continue
        # Adjacent cells across a real step must show the same ordinal discontinuity.
        member=np.zeros(len(uv),bool)
        member[piece]=True
        contact=(member[a] & ~marked[b]) | (member[b] & ~marked[a])
        if contact.sum()>=2 and np.median(abs(residual[a[contact]]-residual[b[contact]]))>=.35*gap:
            offset[piece]=sign
    ev["reason"] = "step" if offset.any() else "no-local-step"
    return offset,ev


def reference_offsets(chart, pixels, valid, owners, labels, camera):
    orient = yaw_from_anchor_normal(chart["normal_camera"],(0,0,1),camera)
    n = tuple(orient["best_world_normal"])
    offsets = np.zeros(len(valid),int)
    depth = np.full(len(valid),np.nan)
    if orient["yaw_quarters"] is None:
        return offsets, depth, None
    faces = labels["faces"]
    blocks = {b["id"]:b for b in labels["blocks"]}
    table = np.full(int(owners.max())+1,np.nan)
    for f in faces:
        if tuple(f["normal_world"])==n and blocks[f["block"]].get("geometry_kind","full_cube")=="full_cube":
            table[f["id"]] = np.dot(f["xyz_world"],n)
    ix,iy = np.floor(pixels).astype(int).transpose(2,0,1)
    inside=(ix>=0)&(ix<owners.shape[1])&(iy>=0)&(iy<owners.shape[0])
    point=table[owners[iy.clip(0,owners.shape[0]-1),ix.clip(0,owners.shape[1]-1)]]
    point[~inside]=np.nan
    for i in np.flatnonzero(valid):
        choices=Counter(point[i,np.isfinite(point[i])])
        if choices and choices.most_common(1)[0][1]>=5:
            depth[i]=choices.most_common(1)[0][0]
    eligible=np.isfinite(depth)
    if not eligible.any():
        return offsets, depth, None
    parent=int(Counter(depth[eligible]).most_common(1)[0][0])
    offsets[eligible]=np.clip(depth[eligible]-parent,-1,1).astype(int)
    return offsets, depth, parent


def boundary_order(uv, predicted, depth):
    # One whole-build translation can swap which depth band is treated as zero.
    # The signed step across a neighboring pair is invariant to this convention.
    lookup={tuple(pos):i for i,pos in enumerate(uv)}
    edges=Counter()
    for i in np.flatnonzero(predicted):
        x,y=uv[i]
        for pos in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            j=lookup.get(pos)
            if j is None or predicted[j]:
                continue
            if not np.isfinite(depth[[i,j]]).all():
                edges["unresolved"]+=1
                continue
            delta=int(depth[i]-depth[j])
            edges["comparable"]+=1
            edges["sign_agrees"]+=int(np.sign(delta)==predicted[i])
            edges["one_block_agrees"]+=int(delta==predicted[i])
    return dict(edges)


def measure_steps(uv, predicted, depth):
    """One scalar per existing source part, from visible, parallel neighbors across its boundary."""
    lookup={tuple(pos):i for i,pos in enumerate(uv)}
    out=np.zeros(len(uv),int)
    parts=[]
    for sign in (-1,1):
        for piece in components(predicted==sign,uv):
            votes=[]
            for i in piece:
                x,y=uv[i]
                for pos in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    j=lookup.get(pos)
                    if j is not None and not predicted[j] and np.isfinite(depth[[i,j]]).all():
                        votes.append(int(depth[i]-depth[j]))
            # Two independent adjacent-cell pairs are the minimum evidence. A tie
            # between distinct ordinal modes abstains from moving this part.
            ranked=Counter(votes).most_common(2)
            measured=0
            if ranked and ranked[0][1]>=2 and (len(ranked)==1 or ranked[0][1]>ranked[1][1]):
                measured=int(np.clip(ranked[0][0],-4,4))
            out[piece]=measured
            parts.append(dict(cells=len(piece),source_sign=sign,reference_step=measured,votes=dict(Counter(votes))))
    return out,parts


def run(job):
    panel,case=job
    cv2.setNumThreads(1)
    name=case["id"]
    rel=Path("sky")/name/"charts.json"
    baseline=WORKTREES/"floating-extra-isolation/runs/extra-isolation"/panel/"large-contraction"
    packet=read(baseline/"charts"/rel)
    source=paths(panel)[0][0]
    image=Path(case["variants"]["sky"]["image"])
    sha=digest(image)
    handoff=source/"sky"/name/"source-handoff"
    assert sha==packet["source_sha256"]==case["variants"]["sky"]["source_sha256"]
    combined.paths.check_handoff(packet,handoff,sha)
    with Image.open(image) as im:
        size=np.array(im.size)
    field,hashes=fields(panel,case,source)
    planes={p["id"]:p for p in read(handoff/"surfaces.json")["planes"]}
    frame=Path(case["frame"])
    with np.load(frame/"labels.npz") as raw:
        owners=raw["face_id"]
    labels=read(frame/"labels.json")
    assert owners.shape==tuple(size[::-1])
    counts=np.bincount(owners.ravel().astype(int))
    assert all(counts[f["id"]]==f["visible_pixels"] for f in labels["faces"])
    camera=read(frame/"camera.json")
    arms={k:[] for k in ("source","visible-reference","reference-source-step")}
    audit=[]
    for chart in packet["charts"]:
        cells=chart["cells"]
        if not cells or abs(chart["normal_camera"][1])>=.67:
            for charts in arms.values():
                charts.append(chart)
            continue
        uv=np.array([(c["u"],c["v"]) for c in cells])
        pixels=project(np.array(chart["H"]),uv[:,None,:]+SAMPLES)
        alpha,(x,y,r,b)=_native_alpha(planes[chart["id"]]["support"],size)
        ix,iy=np.floor(pixels).astype(int).transpose(2,0,1)
        ok=(ix>=x)&(ix<r)&(iy>=y)&(iy<b)
        native=np.asarray(alpha)>0
        ok &= native[(iy-y).clip(0,b-y-1),(ix-x).clip(0,r-x-1)]
        valid=ok.sum(1)>=5
        predicted,detail=image_offsets(chart,uv,pixels,valid,field,size)
        expected,depth,parent=reference_offsets(chart,pixels,valid,owners,labels,camera)
        measured,parts=measure_steps(uv,predicted,depth)
        for arm,values in zip(arms,(predicted,expected,measured)):
            copied=[dict(cell,normal_offset=int(v)) if v else cell for cell,v in zip(cells,values)]
            arms[arm].append(dict(chart,cells=copied))
        moved=predicted!=0
        record=dict(chart=chart["id"],cells=len(cells),base_plane=parent,source=detail,
                    boundary_order=boundary_order(uv,predicted,depth),
                    measured_source_parts=parts,
                    source_offsets=[[int(u),int(v),int(offset)] for (u,v),offset in zip(uv[moved],predicted[moved])],
                    reference_offsets=[[int(u),int(v),int(o)] for (u,v),o in zip(uv,expected) if o],
                    native_parallel=int(np.isfinite(depth).sum()),agrees=int((moved&(predicted==expected)).sum()),
                    wrong_depth=int((moved&np.isfinite(depth)&(predicted!=expected)).sum()),
                    no_parallel_face=int((moved&~np.isfinite(depth)).sum()))
        audit.append(record)
    roots={"large-contraction":baseline/"assembly"}
    if name in ("train-riverside-mill-1-warm-low-left","train-shrine-02-ochre-low-left"):
        noop=OUT/panel/"zero-offset"
        control=[dict(chart,cells=[dict(cell,normal_offset=0) for cell in chart["cells"]])
                 for chart in packet["charts"]]
        save(noop/"charts"/rel,dict(packet,charts=control))
        with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
            combined.run((panel,case,noop/"charts",noop/"assembly"))
        assert read(noop/"assembly"/LOCATION/name/"placements.json")==read(baseline/"assembly"/LOCATION/name/"placements.json")
    for arm,charts in arms.items():
        folder=OUT/panel/arm
        if charts==packet["charts"]:
            roots[arm]=baseline/"assembly"
        else:
            save(folder/"charts"/rel,dict(packet,charts=charts))
            with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
                combined.run((panel,case,folder/"charts",folder/"assembly"))
            roots[arm]=folder/"assembly"
    result=summarize([case],roots,str(LOCATION))["rows"][0]
    receipts={k:read(v/LOCATION/name/"score.json") for k,v in roots.items()}
    for arm,receipt in receipts.items():
        assert receipt["joins"]==receipts["large-contraction"]["joins"]
        assert receipt["reference_sha256"]==receipts["large-contraction"]["reference_sha256"]
    save(OUT/panel/"cases"/(name+".json.gz"),
         dict(**result,panel=panel,source_sha256=sha,field_sha256=hashes,chart_sha256=digest(baseline/"charts"/rel),
              reference_sha256=receipts["large-contraction"]["reference_sha256"],charts=audit,
              selected={k:v["selected_origins"] for k,v in receipts.items()},
              pack_sha256={k:v["pack_sha256"] for k,v in receipts.items()}))
    print(panel,name,{k:(v["geometry"]["strict"]["correct"],v["geometry"]["strict"]["extras"]) for k,v in result["score"].items()},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("panels",nargs="*",default=PANELS)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    jobs=[(g,c) for g in args.panels for c in paths(g)[0][3]]
    if args.resume:
        jobs=[(g,c) for g,c in jobs if not (OUT/g/"cases"/(c["id"]+".json.gz")).exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))


if __name__=="__main__":
    main()
