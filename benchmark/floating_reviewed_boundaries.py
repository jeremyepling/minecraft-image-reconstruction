"""Isolate reference surface boundaries and grid origin on the six reviewed walls."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import save
from benchmark.floating_bridges_report import summarize
from benchmark.floating_extra_budget import combined, LOCATION, WORKTREES
from benchmark.floating_one_face_control import extend
from benchmark.floating_reviewed_rulers import CASES, OUT, paths, project, read


def phase_offset(corners):
    # All true block corners have integer coordinates on a perfectly aligned grid.
    # Hold pitch and perspective fixed. Find the translation minimizing mean circular L1.
    phase = np.linspace(-.5, .5, 201)
    uv = corners.reshape(-1,2)
    d = uv[:,:,None] - phase
    return phase[np.argmin(np.mean(abs(d-np.round(d)), axis=0),axis=1)]


def run():
    measurements = read(OUT/"measurements.json")
    results = {}
    for panel,name,wall,_,_ in CASES:
        info = measurements[name]
        (source,_,_,cases),_,_ = paths(panel)
        case = next(c for c in cases if c["id"]==name)
        root = WORKTREES/"floating-extra-isolation/runs/extra-isolation"/panel/"large-contraction"
        rel = Path("sky")/name/"charts.json"
        raw = read(root/"charts"/rel)
        chart = next(c for c in raw["charts"] if c["id"]==wall)
        h = np.asarray(chart["H"])
        frame = Path(case["frame"])
        labels = read(frame/"labels.json")
        principal = [f for f in labels["faces"] if f["surface_id"]==info["principal_surface_id"]]
        with np.load(frame/"labels.npz") as arr:
            mask_ref = np.isin(arr["face_id"],[f["id"] for f in principal])
        from engine.surface_perimeter import _native_alpha
        plane = next(p for p in read(source/"sky"/name/"source-handoff/surfaces.json")["planes"] if p["id"]==wall)
        alpha,(x,y,r,b) = _native_alpha(plane["support"],mask_ref.shape[::-1])
        mask = np.zeros_like(mask_ref)
        mask[y:b,x:r] = np.asarray(alpha)>0
        visible = mask & mask_ref
        with Image.open(case["variants"]["sky"]["image"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        with np.load(frame/"labels.npz") as arr:
            counts = np.bincount(arr["face_id"][mask].astype(int))
        usable = [f for f in principal if f["id"]<len(counts) and counts[f["id"]] >= .5*f["visible_pixels"]
                  and np.allclose(np.linalg.norm(np.roll(f["corners_world"],-1,axis=0)-np.array(f["corners_world"]),axis=1),1.)
                  and all(f["corner_visible"])]
        corners = np.array([project(np.linalg.inv(h),np.array(f["corners_px"])) for f in usable])
        offset = phase_offset(corners)
        delta = np.eye(3)
        delta[:2,2] = offset
        arms = {"original":root/"assembly"}
        target = Path("runs/feedback-boundary")/panel
        for key,hm,support in [("resample",h,mask), ("origin",h@delta,mask),
                                ("boundary",h,visible), ("both",h@delta,visible)]:
            candidate = extend(chart,hm,support,rgb)
            candidate["confidence"] = chart["confidence"]
            packet = dict(raw, charts=[candidate if c["id"]==wall else c for c in raw["charts"]])
            folder = target/key
            save(folder/"charts"/rel,packet)
            with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
                combined.run((panel,case,folder/"charts",folder/"assembly"))
            arms[key] = folder/"assembly"
        scores = summarize([case],arms,str(LOCATION))["rows"][0]
        results[name] = dict(offset=offset.tolist(),corners=len(usable)*4,scores=scores)
        print(name,{k:(v["geometry"]["strict"]["correct"],v["geometry"]["strict"]["extras"]) for k,v in scores["score"].items()},flush=True)
    save(OUT/"boundary-and-origin.json.gz",results)


if __name__=="__main__":
    run()
