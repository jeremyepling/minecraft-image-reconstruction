"""Attach a counted but disconnected parallel wall fragment when image grids and repeated tiles agree."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_depth_refit import (ARMS, DEPTH_SAMPLES, EXTENDED, FLAT, LOCATION, PANELS,
    PRIOR, ROOT, WORKTREES, combined, digest, paths, read, save, summarize, totals)
from benchmark.relative_face_assembly import axis_vector, oriented
from benchmark.relative_face_scoring import _yaw
from engine.geometry import UNIT, bilinear, project

BASE = WORKTREES/"floating-depth-refit"
OUT = ROOT/"runs/column-recovery"
REVIEW = ROOT/"benchmark/reviews/floating-column-recovery-01"
METHODS = ("grid-attachment","correlation-only","repeated-texture-attachment","bounded-visible-cell")
TEXELS = np.stack(np.meshgrid((np.arange(12)+.5)/12,(np.arange(12)+.5)/12),-1)


def base_root(panel, name):
    rel = Path("charts/sky")/name/"charts.json"
    for root in (BASE/"runs/depth-refit"/panel/EXTENDED,BASE/"runs/depth-refit"/panel/ARMS[1],
                 PRIOR/panel/"source",FLAT/panel/"large-contraction"):
        if (root/rel).exists():
            return root
    raise FileNotFoundError(name)


def uv_cells(chart):
    return np.array([[c["u"],c["v"]] for c in chart["cells"]],int)


def candidate(parent, child):
    """Fit a shared integer phase over this observed fragment, checking both chart carriers."""
    cuv,puv = uv_cells(child),uv_cells(parent)
    if (parent["axes"]!=child["axes"] or parent["normal_axis"]!=child["normal_axis"]
            or min(len(cuv),len(puv))<4 or np.any(np.ptp(cuv,axis=0)<1)):
        return
    h,g = np.asarray(parent["H"]),np.asarray(child["H"])
    pos = project(np.linalg.solve(h,g),cuv+.5)
    shift = np.rint(np.median(pos-cuv-.5,axis=0)).astype(int)
    mapped = cuv+shift
    err = np.abs(pos-mapped-.5).max(1)
    radius = float(np.quantile(err,.9))
    if radius>.30 or err.max()>.45:
        return
    child_keys,parent_keys = set(map(tuple,mapped)),set(map(tuple,puv))
    if len(child_keys & parent_keys)>.10*len(child_keys):
        return
    touching = sum((x+dx,y+dy) in parent_keys for x,y in child_keys for dx,dy in ((0,1),(0,-1),(1,0),(-1,0)))
    if touching<2:
        return
    return dict(shift=shift.tolist(),p90=radius,max_error=float(err.max()),contacts=touching)


def signatures(chart, rgb):
    points = project(np.array(chart["H"]),uv_cells(chart)[:,None,None,:]+TEXELS)
    tiles = bilinear(rgb,points).astype("f")
    gray = tiles.mean(3)
    vertical = np.abs(np.diff(gray,axis=2)).mean((1,2))
    horizontal = np.abs(np.diff(gray,axis=1)).mean((1,2))
    ratio = vertical / np.maximum(horizontal,1.)
    templates = gray.reshape((len(gray),-1))
    templates -= templates.mean(1,keepdims=True)
    templates /= np.maximum(np.linalg.norm(templates,axis=1,keepdims=True),1e-5)
    return tiles.mean((1,2)), templates, ratio


def texture_depth(parent,child,shift,rgb,allow_grain=True):
    """Only a repeated appearance on a corresponding source row can transfer a known relief step."""
    puv,cuv = uv_cells(parent),uv_cells(child)+shift
    heights = next(i for i,a in enumerate(parent["axes"]) if abs(a)==2)
    pm,pt,pv = signatures(parent,rgb)
    cm,ct,cv = signatures(child,rgb)
    applied = {(c["u"],c["v"]):c.get("normal_offset",0) for c in parent.get("export_cells",parent["cells"])}
    rows, result = defaultdict(list),[]
    for i,uv in enumerate(cuv):
        row = int(uv[heights])
        candidates = np.flatnonzero(puv[:,heights]==row)
        hits = []
        for j in candidates:
            key = tuple(puv[j])
            if key not in applied:
                continue
            color = np.linalg.norm(pm[j]-cm[i])
            correlation = float(pt[j]@ct[i])
            # Strong vertical grain is allowed when the thin line falls in another texel.
            grain = cv[i]>=1.55 and pv[j]>=1.55 and color<15
            if color<40 and (correlation>=.50 or (allow_grain and grain)):
                hits.append((correlation-color/160+.15*(allow_grain and grain),int(applied[key]),j,
                             correlation,float(color),bool(grain)))
        if hits:
            _,offset,j,correlation,color,grain = max(hits)
            rows[row].append(offset)
            result.append(dict(child=i,parent=int(j),offset=offset,correlation=correlation,
                               color_distance=color,strong_grain=grain))
    depth = []
    for uv in cuv:
        votes = Counter(rows[int(uv[heights])]).most_common()
        depth.append(votes[0][0] if votes and votes[0][1]>.5*sum(n for _,n in votes) else 0)
    return len(result)>=4 and len(rows)>=2,depth,result


def reference_offsets(parent,child,shift,origin,receipt,owners,faces):
    """Keep only source proposals matching a frontmost face center under bounded normal translations."""
    fixed = receipt["score"]["fixed"] or receipt["score"]["relaxed_yaw"]
    yaw, translation = fixed["yaw_quarters"],np.asarray(fixed["translation"])
    u,v = map(axis_vector,parent["axes"])
    normal = axis_vector(parent["normal_axis"])
    world_normal = list(map(int,_yaw(normal,yaw)))
    start = np.array(origin)+np.minimum(0,u)+np.minimum(0,v)-np.maximum(0,normal)
    cuv = uv_cells(child)
    pixels = np.floor(project(np.asarray(child["H"]),cuv[:,None,:]+DEPTH_SAMPLES)).astype(int)
    ix,iy = pixels[...,0],pixels[...,1]
    valid = (ix>=0)&(ix<owners.shape[1])&(iy>=0)&(iy<owners.shape[0])
    ids = owners[iy.clip(0,owners.shape[0]-1),ix.clip(0,owners.shape[1]-1)]
    out = []
    for j,(a,b) in enumerate(cuv+shift):
        votes = Counter(int(k) for k in ids[j,valid[j]] if k and faces[int(k)]["normal_world"]==world_normal)
        best = votes.most_common(1)
        target = tuple(faces[best[0][0]]["xyz_world"]) if best and best[0][1]>=5 else None
        out.append(next((n for n in (0,-1,1,-2,2) if tuple(
            np.array(_yaw(start+a*u+b*v+n*normal,yaw))+translation)==target),None))
    return out


def run(job):
    panel,case,archived = job
    cv2.setNumThreads(1)
    name=case["id"]
    rel=Path("sky")/name/"charts.json"
    base=base_root(panel,name)
    packet=read(base/"charts"/rel)
    image=Path(case["variants"]["sky"]["image"])
    assert digest(image)==packet["source_sha256"]==archived["source_sha256"]
    receipt=read(base/"assembly"/LOCATION/name/"score.json")
    assert receipt["pack_sha256"]==archived["pack_sha256"][EXTENDED]
    selected=receipt["selected_origins"]
    charts,_=oriented(packet["charts"])
    choices=[]
    for child in charts:
        if child["id"] in selected:
            continue
        fits=[]
        for parent in charts:
            if parent["id"] in selected:
                fit=candidate(parent,child)
                if fit:
                    fits.append((fit["p90"],-fit["contacts"],parent["id"],parent,fit))
        if fits:
            *_,parent,fit=min(fits)
            choices.append((parent,child,fit))
    inserts={a:defaultdict(list) for a in METHODS}
    audit=[]
    if choices:
        rgb=np.array(Image.open(image).convert("RGB"))
        frame=Path(case["frame"])
        faces={f["id"]:f for f in read(frame/"labels.json")["faces"]}
        with np.load(frame/"labels.npz") as data:
            owners=data["face_id"]
        counts=np.bincount(owners.ravel().astype(int))
        assert owners.shape==rgb.shape[:2] and all(counts[k]==f["visible_pixels"] for k,f in faces.items())
        for parent,child,fit in choices:
            shift=np.array(fit["shift"])
            accepted,depth,hits=texture_depth(parent,child,shift,rgb)
            corr_ok,corr_depth,_=texture_depth(parent,child,shift,rgb,allow_grain=False)
            native=reference_offsets(parent,child,shift,selected[parent["id"]],receipt,owners,faces)
            for arm,offsets in zip(METHODS,([0]*len(depth),corr_depth if corr_ok else [None]*len(depth),
                                            depth if accepted else [None]*len(depth),native)):
                for cell,uv,n in zip(child["cells"],uv_cells(child)+shift,offsets):
                    if n is not None:
                        inserts[arm][parent["id"]].append(dict(cell,u=int(uv[0]),v=int(uv[1]),normal_offset=n))
            audit.append(dict(parent=parent["id"],child=child["id"],H=child["H"],uv=uv_cells(child).tolist(),
                              fit=fit,accepted=accepted,source_steps=depth,witnesses=hits,native_steps=native))
    roots={"prior":base/"assembly"}
    for arm,added in inserts.items():
        if not added:
            roots[arm]=roots["prior"]
            continue
        revised=[dict(c,export_cells=c.get("export_cells",c["cells"])+added[c["id"]])
                 if c["id"] in added else c for c in packet["charts"]]
        folder=OUT/panel/arm
        save(folder/"charts"/rel,dict(packet,charts=revised))
        with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
            combined.run((panel,case,folder/"charts",folder/"assembly"))
        roots[arm]=folder/"assembly"
    row=summarize([case],roots,str(LOCATION))["rows"][0]
    assert row["score"]["prior"]==archived["score"][EXTENDED]
    row.update(panel=panel,layout=archived["layout"],active=archived["active"],source_sha256=archived["source_sha256"],
               source_chart_sha256=digest(base/"charts"/rel),reference_sha256=receipt["reference_sha256"],attachments=audit,
               pack_sha256={k:read(p/LOCATION/name/"score.json")["pack_sha256"] for k,p in roots.items()})
    save(OUT/panel/"cases"/(name+".json.gz"),row)
    print(panel,name,len(choices),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    archive={(r["panel"],r["id"]):r for r in read(BASE/"benchmark/reviews/floating-depth-refit-01/conditions.json.gz")}
    jobs=[(p,c,archive[p,c["id"]]) for p in PANELS for c in paths(p)[0][3]]
    if args.resume:
        jobs=[j for j in jobs if not (OUT/j[0]/"cases"/(j[1]["id"]+".json.gz")).exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))
    rows=[read(f) for p in PANELS for f in sorted((OUT/p/"cases").glob("*.json.gz"))]
    assert len(rows)==92
    save(REVIEW/"conditions.json.gz",rows)
    save(REVIEW/"summary.json",{k:totals(g) for k,g in dict(
        active=[r for r in rows if r["active"]],sky=[r for r in rows if r["panel"]=="sky"]).items()})


if __name__=="__main__":
    main()
