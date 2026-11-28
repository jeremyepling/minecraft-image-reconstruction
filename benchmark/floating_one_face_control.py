"""Supply one visibly complete unit face, then rebuild cells with ordinary mask and atlas consumers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_patch_combined as combined
from benchmark.expanded_baseline import ROOT,WORKTREES,read,save,digest
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.floating_patch_controls import inputs as archived_inputs, paths
from benchmark.floating_patch_report import baseline as archived_baseline
from benchmark.floating_bridges_report import summarize
from benchmark.camera_labels import screen_matrix
from benchmark.relative_oracle_charts import UNIT,fit_homography
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

OUT=ROOT/"runs/one-face"
REVIEW=ROOT/"benchmark/reviews/floating-one-face-01"
PANELS=("original","fresh","supplemental","sky","holdout","holdout2")
ARMS=("one-view","one-chart","one-chart-own-plane")


def inputs(group):
    if group in ("holdout3","holdout4"):
        root=WORKTREES/"floating-face-audit/runs"/("floating-"+group)
        return root/"ordinary",root/"ordinary/atlas/connected",root/"source_material/charts",read(root/"manifest.json")["cases"]
    return archived_inputs(group)


def base(group):
    if group in ("holdout3","holdout4"):
        return WORKTREES/"floating-face-audit/runs"/("floating-"+group)/"horizontal-cells/assembly"
    return archived_baseline(group,True)


def visible_quad(face,mask,owners):
    """All strictly interior native pixel centers must be visible and inside the chart mask."""
    quad=np.array(face["corners_px"])
    if (quad<0).any() or (quad>=owners.shape[::-1]).any():
        return False
    edges=np.roll(quad,-1,axis=0)-quad
    if np.min(np.linalg.norm(edges,axis=1))<4:
        return False
    x,y=np.floor(quad.min(0)).astype(int)
    x1,y1=np.ceil(quad.max(0)).astype(int)
    yy,xx=np.mgrid[y:y1,x:x1]
    inside=np.ones(xx.shape,bool)
    sign=np.sign(edges[0,0]*edges[1,1]-edges[0,1]*edges[1,0])
    for (a,b),(dx,dy) in zip(quad,edges):
        inside &= sign*(dx*(yy+.5-b)-dy*(xx+.5-a))>1e-6
    return inside.sum()>=16 and np.all(mask[y:y1,x:x1][inside]) and np.all(owners[y:y1,x:x1][inside]==face["id"])


def face_points(face,camera):
    up=np.array([0,1,0])
    basis=np.stack([np.cross(up,face["normal_world"]),up])
    xyz=np.asarray(face["corners_world"])
    px=np.asarray(face["corners_px"])
    m=screen_matrix(camera)
    q=(xyz-camera["eye_world"])@m[:,:3].T+m[:,3]
    assert np.max(abs(q[:,:2]/q[:,2:]-px))<1e-4
    uv=xyz@basis.T
    corners=dict(zip(map(tuple,uv-uv.min(0)),px))
    assert set(corners)==set(map(tuple,UNIT))
    return uv.min(0),np.array([corners[tuple(p)] for p in UNIT])


def face_grid(face,camera):
    _,ordered=face_points(face,camera)
    h=fit_homography([0,0],ordered)
    assert np.max(abs(project(h,UNIT)-ordered))<1e-4
    return h


def extend(chart,h,mask,rgb):
    points=np.column_stack(np.nonzero(mask)[::-1])+.5
    # Points exactly on the fitted vanishing line have no finite grid coordinate.
    inv=np.linalg.inv(h)
    w=points@inv[2,:2]+inv[2,2]
    points=points[np.abs(w)>1e-10]
    blank=dict(chart,cells=[])
    sampled=resample(blank,h,mask,rgb,points)
    return preserve_and_rank(blank,sampled,rgb,mask)


def run(job):
    group,case,precision,stabilize=job
    cv2.setNumThreads(1)
    source,ordinary,whole,_=inputs(group)
    frame=Path(case["frame"])
    rel=Path("sky")/case["id"]/"charts.json"
    current,raw=read(whole/rel),read(ordinary/rel)
    image=Path(case["variants"]["sky"]["image"])
    rgb=np.asarray(Image.open(image).convert("RGB"))
    handoff=source/"sky"/case["id"]/"source-handoff"
    for p in (current,raw):
        paths.check_handoff(p,handoff,digest(image))
    parts={p["id"]:p for p in read(handoff/"surfaces.json")["planes"]}
    original={c["id"]:c for c in raw["charts"]}
    with np.load(frame/"labels.npz") as data:
        owners=data["face_id"]
    assert owners.shape==rgb.shape[:2]
    native=read(frame/"labels.json")
    blocks={b["id"]:b for b in native["blocks"]}
    walls={f["id"]:f for f in native["faces"] if f["normal_world"][1]==0 and blocks[f["block"]].get("geometry_kind","full_cube")=="full_cube"}
    counts=np.bincount(owners.ravel().astype(int))
    for fid,f in walls.items():
        assert counts[fid]==f["visible_pixels"]>0
    camera=read(frame/"camera.json")
    lookup=np.zeros(len(counts),int)
    for f in native["faces"]:
        lookup[f["id"]]=f["surface_id"]
    surface_image=lookup[owners]
    fitted,restricted,audit={},{},[]
    paired={}
    for chart in current["charts"]:
        key=chart["id"]
        seed=original[key]
        if not seed["cells"] or not chart["cells"] or abs(chart["normal_camera"][1])>=.67:
            continue
        alpha,(x,y,x1,y1)=_native_alpha(parts[key]["support"],rgb.shape[1::-1])
        mask=np.zeros(owners.shape,bool)
        mask[y:y1,x:x1]=np.asarray(alpha)>0
        seed_uv=[[c["u"]+.5,c["v"]+.5] for c in seed["cells"]]
        seed_xy=project(np.asarray(seed["H"]),np.asarray(seed_uv))
        ids,covered=np.unique(owners[mask],return_counts=True)
        candidates=[]
        for fid,area in zip(ids,covered):
            if fid in walls and area==counts[fid]:
                f=walls[fid]
                center=np.mean(f["corners_px"],axis=0)
                distance=float(np.min(np.linalg.norm(seed_xy-center,axis=1)))
                candidates.append((distance,-int(area),int(fid)))
        face=next((walls[fid] for _,_,fid in sorted(candidates) if visible_quad(walls[fid],mask,owners)),None)
        if face is None:
            audit.append(dict(chart=key,face=None))
            continue
        h=face_grid(face,camera)
        if stabilize:
            _,src=face_points(face,camera)
            observed=np.floor(src)+.5
            prior=np.asarray(chart["H"])
            rectified=project(np.linalg.inv(prior),observed)
            # Fit local axes, spacing, and origin, keeping the source chart's vanishing line.
            transform=np.eye(3)
            transform[:2]=np.linalg.lstsq(np.column_stack((UNIT,np.ones(4))),rectified,rcond=None)[0].T
            fitted[key]=extend(chart,prior@transform,mask,rgb)
            restricted[key]=fitted[key]
        elif precision:
            origin,src=face_points(face,camera)
            rounded=np.floor(src)+.5
            rounded_h=fit_homography([0,0],rounded)
            fitted[key]=extend(chart,rounded_h,mask,rgb)
            # A second reference face supplies the offset in whole units as well as pixel corners.
            farthest=sorted((t for t in candidates if walls[t[2]]["surface_id"]==face["surface_id"] and t[2]!=face["id"]),
                key=lambda t:-np.linalg.norm(np.mean(walls[t[2]]["corners_px"],axis=0)-src.mean(0)))
            other=next((walls[fid] for _,_,fid in farthest if visible_quad(walls[fid],mask,owners)),None)
            paired[key]=other["id"] if other else None
            if other:
                offset,second=face_points(other,camera)
                local=np.vstack([UNIT,UNIT+(offset-origin)])
                target=np.vstack([rounded,np.floor(second)+.5])
                two_h,_=cv2.findHomography(local,target,0)
                restricted[key]=extend(chart,two_h,mask,rgb)
            else:
                restricted[key]=fitted[key]
        else:
            fitted[key]=extend(chart,h,mask,rgb)
            restricted[key]=extend(chart,h,mask&(surface_image==face["surface_id"]),rgb)
        audit.append(dict(chart=key,face=face["id"],corners=face["corners_px"],original=len(chart["cells"]),
                          proposed=len(fitted[key]["cells"]),restricted=len(restricted[key]["cells"])))
    biggest=max((c for c in current["charts"] if c["id"] in fitted),key=lambda c:len(c["cells"]),default={})
    for arm in (("prior-perspective",) if stabilize else ("pixel-center","two-pixel-center") if precision else ARMS):
        replacements=restricted if arm in ("one-chart-own-plane","two-pixel-center") else fitted
        charts=[replacements.get(c["id"],c) if arm!="one-view" or biggest.get("id")==c["id"] else c for c in current["charts"]]
        root=OUT/group/arm
        save(root/"charts"/rel,dict(current,charts=charts,diagnostic=arm))
        with patch.object(combined,"inputs",inputs):
            combined.run((group,case,root/"charts",root/"assembly"))
    audit_folder="perspective-audit" if stabilize else "precision-audit" if precision else "audit"
    save(OUT/group/audit_folder/(case["id"]+".json"),dict(id=case["id"],single=biggest.get("id"),anchors=audit,second_faces=paired))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("groups",nargs="*",default=PANELS)
    p.add_argument("--precision",action="store_true")
    p.add_argument("--stabilize",action="store_true")
    args=p.parse_args()
    groups=args.groups
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(run,[(g,c,args.precision,args.stabilize) for g in groups for c in inputs(g)[3]]))
    scores={}
    for g in groups:
        arms=("prior-perspective",) if args.stabilize else ("pixel-center","two-pixel-center") if args.precision else ARMS
        roots={"horizontal":base(g),**{a:OUT/g/a/"assembly" for a in arms}}
        scores[g]=summarize(inputs(g)[3],roots,"connected-atlas/sky")
        print(g,{a:round(100*v["geometry_strict"]["f1"],2) for a,v in scores[g]["cohorts"]["all"].items()})
    label="stabilize-" if args.stabilize else "precision-" if args.precision else "oracle-"
    save(REVIEW/(label+"-".join(groups)+".json.gz"),scores)
