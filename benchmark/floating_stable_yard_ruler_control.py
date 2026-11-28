"""Isolate known wall-ruler and ownership inputs on the two Stable Yard 1 views."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_extra_budget import LOCATION, WORKTREES, combined, paths
from benchmark.expanded_baseline import digest, read, save
from benchmark.floating_bridges_report import summarize
from benchmark.floating_one_face_control import extend, face_grid, visible_quad
from benchmark.expanded_wall_rulers import local_rulers
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

DEST = Path("runs/verified-expansion-review")
PANEL = "holdout3"
NAMES = ["train-stable-yard-1-warm-low-left","train-stable-yard-1-warm-high-right"]


def run(case, dest, local_pitch):
    name = case["id"]
    relation = Path("sky")/name/"charts.json"
    inputs = paths(PANEL)[0]
    handoff = inputs[0]/"sky"/name/"source-handoff"
    base = WORKTREES/"floating-extra-isolation/runs/extra-isolation"/PANEL/"large-contraction"
    packet = read(base/"charts"/relation)
    wall = next(c for c in packet["charts"] if c["id"]=="mask-0")
    plane = next(p for p in read(handoff/"surfaces.json")["planes"] if p["id"]=="mask-0")
    image = Path(case["variants"]["sky"]["image"])
    assert digest(image) == packet["source_sha256"]
    with Image.open(image) as source:
        rgb = np.asarray(source.convert("RGB"))
    alpha,(x,y,x1,y1) = _native_alpha(plane["support"],rgb.shape[1::-1])
    mask = np.zeros(rgb.shape[:2],bool)
    mask[y:y1,x:x1] = np.asarray(alpha)>0
    frame = Path(case["frame"])
    labels = read(frame/"labels.json")
    camera = read(frame/"camera.json")
    with np.load(frame/"labels.npz") as data:
        owners = data["face_id"]
    faces = {f["id"]:f for f in labels["faces"] if f["normal_world"][1]==0}
    valid,counts = np.unique(owners[mask],return_counts=True)
    found = {int(f):int(c) for f,c in zip(valid,counts) if f in faces}
    areas = Counter()
    for k,c in found.items():
        areas[faces[k]["surface_id"]] += c
    principal = areas.most_common(1)[0][0]
    reference_mask = np.isin(owners,[k for k,f in faces.items() if f["surface_id"]==principal]) & mask
    seeds = next(c for c in read(inputs[1]/relation)["charts"] if c["id"]=="mask-0")
    anchors = project(np.asarray(seeds["H"]),np.asarray([[c["u"]+.5,c["v"]+.5] for c in seeds["cells"]]))
    possible = [f for k,f in faces.items() if f["surface_id"]==principal and found.get(k)==f["visible_pixels"]
                and visible_quad(f,mask,owners)]
    face = min(possible,key=lambda f:np.linalg.norm(anchors - np.mean(f["corners_px"],axis=0),axis=1).min())
    fitted = face_grid(face,camera)
    roots = {"baseline":base/"assembly"}
    counts = {"baseline":len(wall["cells"])}
    treatments = {"exact-ruler": (fitted, mask), "exact-ruler-and-plane": (fitted, reference_mask)}
    calibration = None
    if local_pitch:
        (_, spacing, phase), calibration = local_rulers(wall, fitted)
        treatments = {"exact-local-pitch": (spacing, mask),
                      "exact-local-pitch-and-phase": (phase, mask), "exact-ruler": (fitted, mask)}
    for arm,(geometry,support) in treatments.items():
        new = extend(wall,geometry,support,rgb)
        out = dest/name/arm
        save(out/"charts"/relation,dict(packet,charts=[new if c["id"]=="mask-0" else c for c in packet["charts"]]))
        with patch.object(combined,"inputs",lambda _:inputs):
            combined.run((PANEL,case,out/"charts",out/"assembly"))
        counts[arm] = len(new["cells"])
        roots[arm] = out/"assembly"
    summary=summarize([case],roots,str(LOCATION))["rows"][0]
    summary.update(chart_size=counts, anchor=face["id"],principal=principal,
                   source_sha256=digest(image),reference_sha256=digest(frame/"labels.npz"),
                   source_mask_pixels=int(mask.sum()),principal_pixels=int(reference_mask.sum()))
    if calibration is not None:
        summary["local_calibration"] = calibration
    print(name,counts,summary["source_mask_pixels"],summary["principal_pixels"],face["id"])
    print([(arm,s["geometry"]["strict"],s["scores"]["strict"]["correct"]) for arm,s in summary["score"].items()])
    return summary


if __name__=="__main__":
    cv2.setNumThreads(1)
    cases={case["id"]:case for case in paths(PANEL)[0][3]}
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--local-pitch", action="store_true", help="Preserve the source perspective and test exact pitch and phase.")
    args = parser.parse_args()
    dest = Path("runs/peer-mixed-review") if args.local_pitch else DEST
    receipt = "pitch-only-control.json" if args.local_pitch else "stable-yard-oracle.json"
    save(dest/receipt,[run(cases[name],dest,args.local_pitch) for name in NAMES])
