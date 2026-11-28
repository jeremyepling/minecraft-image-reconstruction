"""Transfer the physical brick period between parallel faces meeting separate corners."""
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, read, save, digest
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import TEXELS, atlas, preserve_and_rank
from benchmark.floating_extra_budget import paths, combined, LOCATION
from benchmark.floating_bridges_report import summarize
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

BASE = WORKTREES / "floating-occlusion-fragments/runs/occlusion-fragments"
OUT = ROOT / "runs/parallel-scale"
REVIEW = ROOT / "benchmark/reviews/floating-parallel-scale-01"


def matches(chart, material):
    return sum(c["block_id"].split(":")[-1] == material and c.get("confidence",0) >= .5
               for c in chart["cells"])


def owned_mask(plane, rgb):
    alpha,(x,y,r,b) = _native_alpha(plane["support"],rgb.shape[1::-1])
    mask = np.zeros(rgb.shape[:2],bool)
    mask[y:b,x:r] = np.array(alpha)>0
    return mask


def texture_score(h, points, mask, rgb, patterns):
    keys = np.unique(np.floor(project(np.linalg.inv(h),points)).astype(int),axis=0)
    xy = np.floor(project(h,keys[:,None,None,:]+TEXELS)).astype(int)
    x,y = xy[...,0],xy[...,1]
    inside = (x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
    x,y = x.clip(0,mask.shape[1]-1), y.clip(0,mask.shape[0]-1)
    usable = (inside & mask[y,x]).mean((1,2))>=.97
    if usable.sum()<4:
        return -1.
    tiles = rgb[y[usable],x[usable]].mean(-1)
    tiles -= tiles.mean((1,2),keepdims=True)
    tiles /= np.maximum(tiles.std((1,2),keepdims=True),1.e-6)
    # One orientation for the entire photographed wall.
    return float(np.median(np.einsum("nhw,khw->nk",tiles,patterns)/256,axis=0).max())


def proposals(chart, charts, plane, rgb):
    h = np.asarray(chart["H"])
    # This corner representation uses u=0 on the connected vertical seam.
    orth = [c for c in charts if c is not chart and matches(c,"bricks")>=4 and
            np.allclose(np.asarray(c["H"])[:,1:]/c["H"][2][2],h[:,1:]/h[2,2],atol=1.e-5)]
    donors = [c for c in charts if c is not chart and matches(c,"bricks")>=4 and
              matches(c,"bricks")>=.4*len(c["cells"]) and
              np.dot(c["normal_camera"],chart["normal_camera"])>.99]
    if not orth or not donors or matches(chart,"bricks")>=.4*len(chart["cells"]):
        return
    donor = max(donors,key=lambda c:matches(c,"bricks"))
    left = np.asarray(donor["H"])
    scale = np.dot(h[:,1],left[:,1])/np.dot(left[:,1],left[:,1])
    if np.linalg.norm(left[:,1]*scale-h[:,1]) > .025*np.linalg.norm(h[:,1]):
        return
    transferred = h.copy()
    transferred[:,0] = left[:,0]*scale
    # The mask quad identifies each facade end. Round only the vertical row labels.
    # Search widths around the parallel transfer and let the screenshot choose the pitch.
    quad = np.asarray(plane["quad"])
    uv = project(np.linalg.inv(transferred), quad)
    near,far = np.argsort(np.abs(uv[:,0]))[:2], np.argsort(np.abs(uv[:,0]))[2:]
    if np.max(np.abs(uv[near,0]))>.3 or np.ptp(uv[far,0])>1.1:
        return
    width = int(round(np.mean(uv[far,0])))
    if width < 3:
        return
    rows = np.rint(uv[far,1])
    if np.max(np.abs(rows-uv[far,1]))>.45:
        return
    mask = owned_mask(plane,rgb)
    points = np.column_stack(np.nonzero(mask)[::-1])+.5
    a = atlas()
    patterns = a.patterns[a.labels.index("bricks")]
    arms = [("unchanged",h),("transferred",transferred)]
    for n in range(max(2,width-2),width+3):
        matrix, target = [],[]
        for (x,y),v in zip(quad[far],rows):
            fixed = h[:,2]+v*h[:,1]
            matrix.extend([[n,0,-x*n],[0,n,-y*n]])
            target.extend([x*fixed[2]-fixed[0],y*fixed[2]-fixed[1]])
        fitted = h.copy()
        fitted[:,0] = np.linalg.lstsq(matrix,target,rcond=None)[0]
        arms.append((str(n),fitted))
    scores = {key:texture_score(warp,points,mask,rgb,patterns) for key,warp in arms}
    chosen,warp = max(arms,key=lambda p:scores[p[0]])
    if chosen=="unchanged" or scores[chosen] < .48 or scores[chosen]-scores["unchanged"]<.15:
        return
    updated = resample(chart,warp,mask,rgb,points)
    updated = preserve_and_rank(dict(cells=[]),updated,rgb,mask)
    return dict(updated,confidence=chart["confidence"]),dict(donor=donor["id"],anchor=orth[0]["id"],
        recipient=chart["id"],transferred_far_uv=uv[far].tolist(),scores=scores,chosen=chosen,
        previous_cells=len(chart["cells"]),corrected_cells=len(updated["cells"]))


def run(job):
    panel,case=job
    name=case["id"]
    src,_,_,_=paths(panel)[0]
    chartfile=BASE/panel/"charts/sky"/name/"charts.json"
    packet=read(chartfile)
    source=Path(case["variants"]["sky"]["image"])
    assert digest(source)==case["variants"]["sky"]["source_sha256"]
    rgb=np.asarray(Image.open(source).convert("RGB"))
    planes={p["id"]:p for p in read(src/"sky"/name/"source-handoff/surfaces.json")["planes"]}
    changed,output=[],[]
    for c in packet["charts"]:
        proposal=proposals(c,packet["charts"],planes[c["id"]],rgb)
        if proposal:
            output.append(proposal[0])
            changed.append(proposal[1])
        else:
            output.append(c)
    dest=OUT/panel
    save(dest/"charts/sky"/name/"charts.json",dict(packet,charts=output))
    combined.run((panel,case,dest/"charts",dest/"assembly"))
    roots=dict(previous=BASE/panel/"assembly",parallel=dest/"assembly")
    score=summarize([case],roots,str(LOCATION))
    record=dict(id=name,panel=panel,source=digest(source),baseline=digest(chartfile),changed=changed,
                score=score["cohorts"]["all"],paired=score["rows"][0]["paired"]["parallel"])
    save(REVIEW/panel/(name+".json"),record)
    print(name,changed,record["paired"],flush=True)
    return record


if __name__=="__main__":
    import argparse
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--panel",choices=["sky","original"],default="original")
    panel=args.parse_args().panel
    jobs=[(panel,c) for c in paths(panel)[0][3] if (BASE/panel/"charts/sky"/c["id"]/"charts.json").is_file()]
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(run,jobs))
