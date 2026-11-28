"""Isolate reference wall directions while preserving each atlas proposal's local pitch."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, read, save
from benchmark.expanded_wall_rulers import jacobian
from benchmark.floating_bridges_report import summarize
from benchmark.floating_extra_budget import LOCATION, WORKTREES, combined, paths
from benchmark.floating_one_face_control import extend, face_grid
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

PEER = WORKTREES/"floating-peer-ruler-refit/runs/peer-ruler-refit/atlas-first/holdout3/relaxed"
DEST = Path("runs/stable-yard-direction-control")


def reorient(proposal, reference, snap=False):
    """Change only directions and perspective, retaining both derivative lengths at the fit origin."""
    center = project(proposal,np.zeros(2))
    uv = project(np.linalg.inv(reference),center)
    original = jacobian(proposal,np.zeros(2))
    if snap:
        uv = uv.round()
    target = jacobian(reference,uv)
    original_lengths = np.linalg.norm(original,axis=0)
    new_lengths = np.linalg.norm(target,axis=0)
    cosine = (target/new_lengths).T @ (original/original_lengths)
    order = [0,1] if np.trace(abs(cosine)) > np.trace(abs(cosine[:,::-1])) else [1,0]
    direction = np.eye(2)[:,order] * np.sign(cosine[order,np.arange(2)])
    convert = np.eye(3)
    convert[:2,:2] = direction * (original_lengths/new_lengths[order])
    convert[:2,2] = uv
    return reference @ convert


def run(measurement, case):
    name = case["id"]
    rel = Path("sky")/name/"charts.json"
    inputs = paths("holdout3")[0]
    input_root = inputs[0]/"sky"/name/"source-handoff"
    image = Path(case["variants"]["sky"]["image"])
    packet = read(PEER/"charts"/rel)
    assert digest(image)==packet["source_sha256"]==measurement["source_sha256"]
    with Image.open(image) as im:
        rgb = np.asarray(im.convert("RGB"))
    wall = next(c for c in packet["charts"] if c["id"]=="mask-0")
    plane = next(p for p in read(input_root/"surfaces.json")["planes"] if p["id"]=="mask-0")
    alpha,(x,y,x1,y1) = _native_alpha(plane["support"],rgb.shape[1::-1])
    mask = np.zeros(rgb.shape[:2],bool)
    mask[y:y1,x:x1] = np.asarray(alpha)>0
    frame = Path(case["frame"])
    face = next(f for f in read(frame/"labels.json")["faces"] if f["id"]==measurement["anchor"])
    reference = face_grid(face,read(frame/"camera.json"))
    roots={"forced-atlas":PEER/"assembly"}
    for arm,snap in (("exact-directions",False),("exact-directions-and-local-alignment",True)):
        h=reorient(np.asarray(wall["H"]),reference,snap)
        updated=extend(wall,h,mask,rgb)
        out=DEST/arm
        save(out/"charts"/rel,dict(packet,charts=[updated if c["id"]=="mask-0" else c for c in packet["charts"]]))
        with patch.object(combined,"inputs",lambda _:inputs):
            combined.run(("holdout3",case,out/"charts",out/"assembly"))
        roots[arm]=out/"assembly"
    result=summarize([case],roots,str(LOCATION))["rows"][0]
    print(name,[(arm,entry["geometry"]["strict"],entry["scores"]["strict"]["correct"]) for arm,entry in result["score"].items()])
    return result


if __name__=="__main__":
    inputs=paths("holdout3")[0]
    cases={c["id"]:c for c in inputs[3]}
    values=read(Path("benchmark/reviews/floating-verified-expansion-review-01/stable-yard-ruler-control.json"))
    save(DEST/"results.json",[run(row,cases[row["id"]]) for row in values])
