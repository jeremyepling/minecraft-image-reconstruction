"""Diagnose surface ownership and local-grid measurement on already proposed visible wall cells."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import cases, NATIVE
from benchmark.expanded_wall_source_resample import TEXELS
from benchmark.floating_horizontal_bridges import FRESH, OLD
from benchmark.camera_labels import screen_matrix
from benchmark.full_block_lattice import atlas
from benchmark.relative_oracle_charts import fit_homography, UNIT
from engine.atlas_count import rank_tile
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

OUT=ROOT/"runs/patch-controls"
NEW=WORKTREES/"floating-patch-validation/runs/floating-patch-validation"
SKY=WORKTREES/"floating-horizontal-bridges/runs/six-sky"
ARMS=("seed-planes","split","split-refit")


def inputs(cohort):
    if cohort=="original":
        base=NATIVE.parent/"source"
        return base,base/"atlas/connected",OLD/"source_material/charts",list(cases())
    if cohort=="sky":
        base=WORKTREES/"floating-visible-builds/runs/floating-cloud-gate-01"
        return base,base/"garden-anchor-extension/connected",SKY/"charts",read(assembly.SHARED/"fixtures.json")["cases"]
    base=NEW if cohort=="holdout" else FRESH
    manifest="supplemental-manifest.json" if cohort=="supplemental" else "manifest.json"
    return base/"ordinary",base/"ordinary/atlas/connected",base/"source_material/charts",read(base/manifest)["cases"]


def native_centers(chart,owners):
    cells=chart["cells"]
    if not cells:
        return [],[]
    uv=np.array([(c["u"]+.5,c["v"]+.5) for c in cells])
    xy=np.floor(project(np.array(chart["H"]),uv)).astype(int)
    valid=(xy>=0).all(axis=1)&(xy<owners.shape[::-1]).all(axis=1)
    fids=np.zeros(len(cells),int)
    fids[valid]=owners[xy[valid,1],xy[valid,0]]
    return xy,fids


def refit(part,faces,owners,rgb,camera):
    up=np.array([0,1,0])
    right=np.cross(up,faces[0]["normal_world"])
    basis=np.stack([right,up])
    measured=[]
    eye=np.asarray(camera["eye_world"])
    matrix=screen_matrix(camera)
    for face in faces:
        xyz=np.array(face["corners_world"])
        q=(xyz-eye)@matrix[:,:3].T+matrix[:,3]
        px=np.array(face["corners_px"])
        assert np.max(abs(q[:,:2]/q[:,2:]-px))<1e-4
        uv=(xyz@basis.T).astype(int)
        low=uv.min(0)
        vertices=dict(zip(map(tuple,uv-low),px))
        corners=np.array([vertices[tuple(a)] for a in UNIT])
        measured.append((face,low,corners))
    origin=measured[0][1]
    main=max(measured,key=lambda item:item[0]["visible_pixels"])
    h=fit_homography(main[1]-origin,main[2])
    cells=[]
    for face,low,corners in measured:
        uv=low-origin
        assert np.max(abs(project(h,uv+UNIT)-corners))<1e-4
        xy=np.floor(project(h,uv+TEXELS)).astype(int)
        x,y=xy[...,0],xy[...,1]
        ih,iw=owners.shape
        inside=(x>=0)&(x<iw)&(y>=0)&(y<ih)
        x,y=x.clip(0,iw-1),y.clip(0,ih-1)
        ranked=rank_tile(dict(rgb=rgb[y,x],mask=inside&(owners[y,x]==face["id"])),atlas())
        block=ranked["block_id"]
        cells.append(dict(u=int(uv[0]),v=int(uv[1]),block_id=block or "smooth_stone",fallback=not block,
                          confidence=max(0.,ranked["evidence"]["correlation"]) if block else 0.))
    return dict(part,H=h.tolist(),cells=cells)


def run(job):
    cohort,case,regrow=job
    cv2.setNumThreads(1)
    source,ordinary,whole,_=inputs(cohort)
    frame=Path(case["frame"])
    name=case["id"]
    rel=Path("sky")/name/"charts.json"
    raw,packet=read(ordinary/rel),read(whole/rel)
    image=Path(case["variants"]["sky"]["image"])
    sha=digest(image)
    rgb=np.asarray(Image.open(image).convert("RGB"))
    for item in (raw,packet):
        paths.check_handoff(item,source/"sky"/name/"source-handoff",sha)
    planes={p["id"]:p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"]}
    with np.load(frame/"labels.npz") as arrays:
        owners=arrays["face_id"]
    assert rgb.shape[:2]==owners.shape
    native=read(frame/"labels.json")
    faces={f["id"]:f for f in native["faces"]}
    counts=np.bincount(owners.ravel().astype(int))
    assert all(counts[i]==f["visible_pixels"]>0 for i,f in faces.items())
    camera=read(frame/"camera.json")
    blocks={b["id"]:b for b in native["blocks"]}
    by_id={c["id"]:c for c in raw["charts"]}
    arms=("regrow-supported",) if regrow else ARMS
    out={arm:[] for arm in arms}
    diagnostics=[]
    for chart in packet["charts"]:
        seed=by_id[chart["id"]]
        if not seed["cells"] or not chart["cells"] or abs(chart["normal_camera"][1])>=.67:
            for arm in arms:
                out[arm].append(chart)
            continue
        assert np.array_equal(chart["H"],seed["H"])
        _,seed_fids=native_centers(seed,owners)
        seed_planes={faces[f]["surface_id"] for f in seed_fids if f}
        xy,fids=native_centers(chart,owners)
        grouped=defaultdict(list)
        for c,f in zip(chart["cells"],fids):
            if f:
                grouped[faces[f]["surface_id"]].append((c,int(f)))
        if regrow:
            alpha,(x,y,x1,y1)=_native_alpha(planes[chart["id"]]["support"],owners.shape[::-1])
            owned_pixels=np.zeros_like(owners)
            owned_pixels[y:y1,x:x1]=np.where(np.asarray(alpha)>0,owners[y:y1,x:x1],0)
            by_surface=defaultdict(list)
            for fid in np.unique(owned_pixels):
                if fid:
                    f=faces[fid]
                    if f["normal_world"][1]==0 and blocks[f["block"]].get("geometry_kind","full_cube")=="full_cube":
                        by_surface[f["surface_id"]].append(f)
            for sid in sorted(grouped.keys()|by_surface.keys()):
                part=dict(chart,id=f"{chart['id']}-surface-{sid}",cells=[c for c,_ in grouped[sid]])
                out["regrow-supported"].append(refit(part,by_surface[sid],owned_pixels,rgb,camera) if sid in by_surface else part)
            continue
        keep=[c for c,f in zip(chart["cells"],fids) if f and faces[f]["surface_id"] in seed_planes]
        out["seed-planes"].append(dict(chart,cells=keep))
        record=dict(id=chart["id"],centers=len(xy),background=int((fids==0).sum()),unique_faces=len(set(fids)-{0}),
                    surfaces=len(grouped),seed_surfaces=sorted(seed_planes),kept=len(keep))
        diagnostics.append(record)
        for sid,rows in grouped.items():
            part=dict(chart,id=f"{chart['id']}-surface-{sid}",cells=[c for c,_ in rows])
            out["split"].append(part)
            unique=[faces[f] for f in dict.fromkeys(f for _,f in rows)]
            cube_wall=all(f["normal_world"][1]==0 and blocks[f["block"]].get("geometry_kind","full_cube")=="full_cube" for f in unique)
            out["split-refit"].append(refit(part,unique,owners,rgb,camera) if cube_wall else part)
        if cohort=="sky" and "chapel" in name and chart["id"]=="mask-0":
            pic=Image.fromarray(rgb)
            pen=ImageDraw.Draw(pic)
            for (x,y),f in zip(xy,fids):
                color="#40ff9b" if f and faces[f]["surface_id"] in seed_planes else "#ff7b43"
                pen.ellipse((x-2,y-2,x+2,y+2),fill=color)
            path=ROOT/"benchmark/reviews/floating-patch-controls-01/chapel-centers.jpg"
            path.parent.mkdir(parents=True,exist_ok=True)
            pic.save(path,quality=90)
    def checked_read(path):
        data=read(path)
        if case["dataset"]=="visible-builds-v2" and path==frame/"frame.json" and "framing" not in data:
            proof=next(r for r in read(ROOT/"training/framing-audit-v2.json")["frames"] if r["id"]==name)
            assert proof["accepted"] and proof["rgb_sha256"]==case["rgb_sha256"]
            data=dict(data,framing=proof)
        return data
    for arm in arms:
        folder=OUT/cohort/arm
        save(folder/"charts"/rel,dict(packet,charts=out[arm],diagnostic=arm))
        mode=(folder/"charts","charts.json",False,1,2.5,.5,False)
        with patch.object(paths,"RUN",source),patch.object(assembly,"ASSEMBLY",folder/"assembly"),patch.dict(
                assembly.MODES,{"connected-atlas":mode}),patch.object(expanded_scores,"read",checked_read):
            assembly.evaluate((case,"sky","connected-atlas"))
    if not regrow:
        save(OUT/cohort/"audit"/(name+".json"),dict(id=name,charts=diagnostics))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cohorts",nargs="+")
    cli.add_argument("--workers",type=int,default=4)
    cli.add_argument("--regrow",action="store_true")
    args=cli.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,[(group,c,args.regrow) for group in args.cohorts for c in inputs(group)[3]]))
