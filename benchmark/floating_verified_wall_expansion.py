"""Verify expansion of a selected wall in source pixels, preserving its original join geometry."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_wall_source_resample import TEXELS, atlas, rank_tile
from benchmark.floating_distant_seams import data, combined, digest, read, save, project
from benchmark.floating_one_face_control import ROOT, WORKTREES, _native_alpha, UNIT
from benchmark.floating_bridges_report import summarize

OUT = ROOT / "runs/verified-wall-expansion"
REVIEW = ROOT / "benchmark/reviews/floating-verified-wall-expansion-01"
FLAGS = WORKTREES/"floating-depth-refit/benchmark/reviews/floating-depth-refit-01/conditions.json.gz"
BASE = WORKTREES/"floating-extra-isolation/runs/extra-isolation"
ARMS = ("anchor", "regional", "peer-guard")
PREFIX = Path("connected-atlas/sky")


def full_face(h, uv, mask, rgb):
    """Test native interior coverage and sprite registration versus quarter-shifted tiles."""
    uv = np.asarray(uv)
    quad = project(h, uv + UNIT)
    if not np.isfinite(quad).all() or (quad < 0).any() or (quad >= mask.shape[::-1]).any():
        return dict(ok=False, reason="clipped")
    edges = np.roll(quad, -1, axis=0) - quad
    if np.min(np.linalg.norm(edges, axis=1)) < 6:
        return dict(ok=False, reason="minified")
    x,y = np.floor(quad.min(0)).astype(int)
    x1,y1 = np.ceil(quad.max(0)).astype(int)
    yy,xx = np.mgrid[y:y1,x:x1]
    crosses = [dx*(yy+.5-b)-dy*(xx+.5-a) for (a,b),(dx,dy) in zip(quad,edges)]
    inside = np.all(np.array(crosses)>0,axis=0) | np.all(np.array(crosses)<0,axis=0)
    if inside.sum()<36 or not mask[y:y1,x:x1][inside].all():
        return dict(ok=False, reason="partial")
    scores = {}
    for shift in ((0,0),(.25,0),(-.25,0),(0,.25),(0,-.25)):
        xy = np.floor(project(h, uv + TEXELS + shift)).astype(int)
        xi,yi = xy[...,0],xy[...,1]
        if (xi<0).any() or (yi<0).any() or (xi>=mask.shape[1]).any() or (yi>=mask.shape[0]).any():
            continue
        valid = mask[yi,xi]
        if valid.mean()<.88:
            continue
        match = rank_tile(dict(rgb=rgb[yi,xi],mask=valid),atlas())
        scores[shift] = (match["evidence"]["correlation"], match["block_id"])
    if (0,0) not in scores or len(scores)<3:
        return dict(ok=False, reason="sparse phase controls")
    corr, material = scores.pop((0,0))
    advantage = corr - np.median([s[0] for s in scores.values()])
    return dict(ok=bool(corr>=.65 and advantage>=.07), correlation=corr, phase_advantage=float(advantage),
                material=material)


def flood(available, starts):
    seen = set(starts) & available
    pending = list(seen)
    while pending:
        u,v = pending.pop()
        for point in ((u+1,v),(u-1,v),(u,v+1),(u,v-1)):
            if point in available and point not in seen:
                seen.add(point)
                pending.append(point)
    return seen


def choose(chart, seed, mask, rgb):
    cells = {(c["u"],c["v"]) for c in chart["cells"]}
    seed_uv = [(c["u"],c["v"]) for c in seed["cells"]]
    # Only challenge a large extrapolation of a small local chart, on the same ruler.
    if (not seed_uv or len(cells)<256 or len(cells)<16*len(seed_uv)
            or not np.allclose(chart["H"],seed["H"]) or abs(chart["normal_camera"][1])>=.67):
        return {}, {}
    h = np.array(chart["H"])
    seeds = sorted(cells.intersection(seed_uv))
    verified = {p:full_face(h,p,mask,rgb) for p in seeds}
    anchors = {p for p,r in verified.items() if r["ok"]}
    restricted = set(seeds)
    regions = defaultdict(list)
    for p in sorted(cells):
        regions[p[0]//6,p[1]//6].append(p)
    supported = {(p[0]//6,p[1]//6) for p in anchors}
    checks = []
    for (u,v),points in sorted(regions.items()):
        if not anchors or (u,v) in supported:
            continue
        # Independent locations are chosen by geometry before inspecting texture evidence.
        samples = sorted(points,key=lambda p:(p[0]-(6*u+2.5))**2+(p[1]-(6*v+2.5))**2)
        probes = []
        for p in samples:
            if all(max(abs(p[0]-a),abs(p[1]-b))>=3 for a,b in (*seeds,*(s["uv"] for s in probes))):
                probes.append(dict(uv=p,**full_face(h,p,mask,rgb)))
            if len(probes)==3:
                break
        if sum(r["ok"] for r in probes)>=2:
            supported.add((u,v))
        checks.append(dict(region=(u,v),probes=probes))
    region_cells = {p for p in cells if (p[0]//6,p[1]//6) in supported}
    regional = restricted | flood(region_cells,anchors)
    return dict(anchor=cells if anchors else restricted,regional=regional), dict(
        seeds=[dict(uv=p,**r) for p,r in verified.items()], seed_count=len(seed_uv), cells=len(cells),
        regions=checks, supported_regions=sorted(supported), regional=len(regional), anchor=len(cells if anchors else restricted))


def peer_pitches(chart, charts):
    """Compare vertical pixel pitches at the same projected position on independent wall charts."""
    def vertical_pitch(h, uv):
        center = project(h,uv)
        steps = project(h,uv+np.eye(2))-center
        lengths = np.linalg.norm(steps,axis=1)
        verticality = abs(steps[:,1])/np.maximum(lengths,1e-7)
        axis = np.argmax(verticality)
        if verticality[axis]<.8 or verticality[axis]-min(verticality)<.2:
            return None
        return lengths[axis]

    ratios = defaultdict(list)
    source = np.asarray(chart["H"])
    for peer in charts:
        # Slices split from the same original mask are a single independent witness.
        origin = peer["id"].split("-wall-")[0]
        if (origin==chart["id"].split("-wall-")[0] or not peer["cells"]
                or not peer.get("normal_camera") or abs(peer["normal_camera"][1])>=.67):
            continue
        other = np.asarray(peer["H"])
        uv = np.median([(c["u"]+.5,c["v"]+.5) for c in peer["cells"]],axis=0)
        here = project(np.linalg.inv(source), project(other,uv))
        a,b = vertical_pitch(source,here),vertical_pitch(other,uv)
        if a and b:
            ratios[origin].append(float(b/a))
    return {name:float(np.median(values)) for name,values in ratios.items()}


def run(job):
    group,case = job
    cv2.setNumThreads(1)
    name = case["id"]
    source,seeds_dir,_,_ = data(group)
    baseline = BASE/group/"large-contraction"
    rel = Path("sky")/name/"charts.json"
    packet = read(baseline/"charts"/rel)
    seeds_path = seeds_dir/rel
    seeds = {c["id"]:c for c in read(seeds_path)["charts"]}
    image = Path(case["variants"]["sky"]["image"])
    assert packet["source_sha256"]==digest(image)
    rgb = np.asarray(Image.open(image).convert("RGB"))
    surfaces = source/"sky"/name/"source-handoff/surfaces.json"
    planes = {p["id"]:p for p in read(surfaces)["planes"]}
    audit,variant = {},{arm:[] for arm in ARMS}
    for chart in packet["charts"]:
        cid = chart["id"]
        choices = {}
        if len(chart["cells"])>=256 and cid in seeds:
            alpha,(x,y,x1,y1) = _native_alpha(planes[cid]["support"],rgb.shape[1::-1])
            mask = np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1] = np.asarray(alpha)>0
            choices,findings = choose(chart,seeds[cid],mask,rgb)
            if choices:
                ratios = peer_pitches(chart,packet["charts"])
                anchors = [s for s in findings["seeds"] if s["ok"]]
                local_regions = {(s["uv"][0]//6,s["uv"][1]//6) for s in anchors}
                far_support = len(findings["supported_regions"])>len(local_regions)
                # Inconclusive distant texture alone is common on valid walls. Require a
                # contradictory unit height from at least two independent source masks.
                contradicted = (len(ratios)>=2 and min(ratios.values())>=2 and not far_support)
                choices["peer-guard"] = choices["regional"] if contradicted else {
                    (c["u"],c["v"]) for c in chart["cells"]}
                findings.update(peer_ratios=ratios,peer_guard=contradicted)
                audit[cid] = findings
        for arm in ARMS:
            keep = choices.get(arm)
            variant[arm].append(chart if keep is None or len(keep)==len(chart["cells"]) else
                                dict(chart,export_uv=sorted(keep)))
    hashes = {}
    for arm in ARMS:
        dst = OUT/group/arm
        changed = any("export_uv" in c for c in variant[arm])
        # Identical handoffs reuse the decoded pack. Changed exports run the complete assembler.
        if changed:
            save(dst/"charts"/rel,dict(packet,charts=variant[arm]))
            with patch.object(combined,"inputs",data):
                combined.run((group,case,dst/"charts",dst/"assembly"))
        root = dst/"assembly" if changed else baseline/"assembly"
        hashes[arm] = dict(changed=changed,pack=digest(root/PREFIX/name/"reconstruction.zip"),
                           charts=digest((dst if changed else baseline)/"charts"/rel))
    save(OUT/group/"audit"/(name+".json.gz"),dict(panel=group,id=name,charts=audit,arms=hashes,
         source=digest(image),mask=digest(surfaces),seeds=digest(seeds_path),base=digest(baseline/"charts"/rel)))
    print(group,name,[(c,v["cells"],v["anchor"],v["regional"]) for c,v in audit.items()],flush=True)


def jobs():
    flags = read(FLAGS)
    panels = dict.fromkeys(r["panel"] for r in flags)
    cases = {(g,c["id"]):c for g in panels for c in data(g)[3]}
    return [(r["panel"],cases[r["panel"],r["id"]]) for r in flags if r["active"]]


def report(work):
    rows = []
    for g,c in work:
        n = c["id"]
        audit = read(OUT/g/"audit"/(n+".json.gz"))
        base = BASE/g/"large-contraction/assembly"
        roots = dict(base=base)
        for arm,r in audit["arms"].items():
            roots[arm] = OUT/g/arm/"assembly" if r["changed"] else base
        row = summarize([c],roots,PREFIX)["rows"][0]
        baseline = read(base/PREFIX/n/"score.json")
        for root in roots.values():
            new = read(root/PREFIX/n/"score.json")
            for key in ("source_sha256","reference_sha256","selected_origins","joins","yaw"):
                assert new[key]==baseline[key],(g,n,key)
        rows.append(dict(**row,audit=audit))
    save(REVIEW/"conditions.json.gz",rows)
    totals = aggregate(rows)
    save(REVIEW/"summary.json",totals)
    save(REVIEW/"cohorts.json",{k:aggregate([r for r in rows if r["dataset"]==k or r["audit"]["panel"]==k])
                                 for k in ("sky",*sorted({r["dataset"] for r in rows}))})
    print(totals,flush=True)


def aggregate(rows):
    totals = {}
    for arm in ("base",*ARMS):
        values = {}
        for field in ("geometry","scores"):
            for kind in ("strict","facade"):
                count = Counter()
                for r in rows:
                    count.update({k:r["score"][arm][field][kind][k] for k in ("correct","missed","extras")})
                count["f1"] = 2*count["correct"]/(2*count["correct"]+count["missed"]+count["extras"])
                values[field+"_"+kind]=dict(count)
        values["paired"] = dict(sum((Counter(r["paired"][arm]) for r in rows),Counter()))
        totals[arm] = values
    return totals


def sheet(work):
    rows = read(REVIEW/"conditions.json.gz")
    cases = {(g,c["id"]):c for g,c in work}
    examples = [r for r in rows if r["audit"]["arms"]["peer-guard"]["changed"]]
    image = Image.new("RGB",(1200,272*len(examples)),"white")
    draw = ImageDraw.Draw(image)
    for i,row in enumerate(examples):
        panel,name = row["audit"]["panel"],row["id"]
        case = cases[panel,name]
        source = Image.open(case["variants"]["sky"]["image"]).convert("RGB")
        rel = Path("sky")/name/"charts.json"
        packet = read(OUT/panel/"peer-guard/charts"/rel)
        selected = next(c for c in packet["charts"] if "export_uv" in c)
        points = project(np.asarray(selected["H"]),np.array([(c["u"]+.5,c["v"]+.5) for c in selected["cells"]]))
        lo = np.maximum(0,points.min(0)-32)
        hi = np.minimum(source.size,points.max(0)+32)
        box = tuple(map(int,(*lo,*hi)))
        images = [source]+[Image.open(folder/PREFIX/name/"source-overlay.jpg") for folder in (
            BASE/panel/"large-contraction/assembly", OUT/panel/"peer-guard/assembly")]
        for j,frame in enumerate(images):
            frame = frame.crop(box)
            frame.thumbnail((394,207),Image.Resampling.LANCZOS)
            image.paste(frame,(400*j+(400-frame.width)//2,272*i+38))
        draw.text((5,272*i+4),name.replace("train-",""),fill="black",font=ImageFont.truetype("arial.ttf",18))
        for j,key in enumerate(("source","base","peer-guard")):
            text="Source image" if j==0 else ("Whole chart" if j==1 else "Peer guard")+"   "+"/".join(
                str(row["score"][key]["geometry"]["strict"][k]) for k in ("correct","missed","extras"))
            draw.text((400*j+5,272*i+244),text,fill="black",font=ImageFont.truetype("arial.ttf",16))
    image.save(REVIEW/"wall-expansion.jpg",quality=91)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter",default="")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--report",action="store_true")
    args = parser.parse_args()
    work = [(g,c) for g,c in jobs() if args.filter in c["id"]]
    if args.report:
        report(work)
        sheet(work)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run,work))
