"""Search replacement wall rulers suggested by independent image-based wall pitches."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_verified_wall_expansion import (BASE, ROOT, WORKTREES, FLAGS, PREFIX, combined,
    data, digest, full_face, jobs, project, read, save, summarize, _native_alpha)
from benchmark.floating_one_face_control import extend
from benchmark.floating_one_face_scores import best

PREVIOUS = WORKTREES/"floating-verified-wall-expansion/benchmark/reviews/floating-verified-wall-expansion-01/conditions.json.gz"
OUT = ROOT/"runs/peer-ruler-refit"
REVIEW = ROOT/"benchmark/reviews/floating-peer-ruler-refit-01"
ARMS = ("bypass","relaxed","verified")
RES = 8


def perspective(h, pivot, strength):
    """Change perspective strength about the patch center, preserving the local Jacobian."""
    t = np.eye(3)
    t[:2,2] = pivot
    centered = h @ t
    centered /= centered[2,2]
    gradient = centered[2,:2].copy()
    centered[2,:2] *= strength
    centered[:2,:2] += (strength-1)*centered[:2,2,None]*gradient
    return centered


def profiles(h, mask, rgb, split):
    xy = np.column_stack(np.nonzero(mask)[::-1])+.5
    uv = project(np.linalg.inv(h),xy)
    lo = np.floor(uv.min(0))-1
    hi = np.ceil(uv.max(0))+1
    shape = (hi-lo)*RES
    if min(shape)<8 or max(shape)>1600:
        return None
    yy,xx = np.indices(shape[::-1].astype(int))
    coords = np.stack((xx,yy),-1)/RES+lo
    points = project(h,coords)
    remap = (points-.5).astype('f')
    pixels = cv2.remap(rgb,remap,None,cv2.INTER_LINEAR).astype(float)
    covered = cv2.remap(mask.astype('B'),remap,None,cv2.INTER_NEAREST).astype(bool)
    # Fit on the left 40% by source X, validate on the right 40%, leave a gap.
    partitions = [covered & (points[...,0]<split[0]),covered & (points[...,0]>split[1])]
    out = []
    for axis in (0,1):
        colors = pixels if axis==0 else pixels.transpose(1,0,2)
        grad = np.linalg.norm(colors[:,2:]-colors[:,:-2],axis=2)
        regions = []
        for region in partitions:
            reg = region if axis==0 else region.T
            good = reg[:,:-2]&reg[:,2:]&reg[:,1:-1]
            weights = good.sum(0)
            strength = np.minimum(grad,64.)
            values = (strength*good).sum(0)/np.maximum(weights,1)
            regions.append((values,weights))
        out.append((lo[axis]+np.arange(1,len(grad[0])+1)/RES,regions))
    return out


def line_score(profile, pitch, phases, region):
    domain,regions = profile
    values,weights = regions[region]
    starts = np.arange(np.floor(domain[0]/pitch)-1,np.ceil(domain[-1]/pitch)+1)*pitch
    grid = starts+phases[:,None]
    support = np.min([np.interp(grid+d*pitch,domain,weights,left=0,right=0) for d in (0.,-.25,.25)],axis=0)
    active = support>=12
    weight = np.sqrt(support)*active
    edge,lf,rt = [np.interp(grid+d*pitch,domain,values) for d in (0.,-.25,.25)]
    advantage = (edge-(lf+rt)/2)/(4+(edge+lf+rt)/3)
    score = (weight*np.clip(advantage,-1,1)).sum(1)/np.maximum(weight.sum(1),1)
    return np.where(active.sum(1)>=3,score,-1),active.sum(1)


def candidates(profile, ratios):
    low,high = min(ratios),max(ratios)
    pitches = np.unique(np.round(np.r_[np.arange(.5,2.05,.1),np.arange(.75*low,1.25*high+.051,.10),
                                      ratios,np.median(ratios)],3))
    out = []
    for pitch in pitches:
        phases = np.arange(24)*pitch/24
        fit,_ = line_score(profile,pitch,phases,0)
        win = int(fit.argmax())
        if .75*low<=pitch<=1.25*high:
            out.append((float(fit[win]),float(pitch),float(phases[win])))
    return max(out)


def harmonic_margin(profile,pitch,phase):
    score,lines = line_score(profile,pitch,np.array([phase]),1)
    rivals = []
    for factor in (.5,2.):
        alt=pitch*factor
        phases=np.arange(24)*alt/24
        fit,_=line_score(profile,alt,phases,0)
        far,_=line_score(profile,alt,phases[[fit.argmax()]],1)
        rivals.append(float(far[0]))
    return dict(validation=float(score[0]),lines=int(lines[0]),harmonics=rivals,margin=float(score[0]-max(rivals)))


def texture(h,mask,rgb,split):
    xy=np.column_stack(np.nonzero(mask)[::-1])+.5
    uv=np.unique(np.floor(project(np.linalg.inv(h),xy)).astype(int),axis=0)
    centers=project(h,uv+.5)
    sides=[]
    for keep in (centers[:,0]<split[0],centers[:,0]>split[1]):
        points=uv[keep]
        # Spatially distributed cell centers, selected without looking at RGB or truth.
        order=np.lexsort((points[:,1],points[:,0]))
        probes=points[order[np.linspace(0,len(order)-1,min(32,len(order))).astype(int)]] if len(order) else []
        sides.append([dict(uv=p.tolist(),**full_face(h,p,mask,rgb)) for p in probes])
    return sides


def search(chart,ratios,mask,rgb):
    h=np.asarray(chart["H"])
    xy=np.column_stack(np.nonzero(mask)[::-1])+.5
    split=np.quantile(xy[:,0],[.4,.6])
    pivot=np.median(project(np.linalg.inv(h),xy),axis=0)
    shapes=[]
    for strength in (0.,.5,1.,1.5):
        corrected=perspective(h,pivot,strength)
        prof=profiles(corrected,mask,rgb,split)
        if prof is None:
            continue
        axes=[candidates(p,ratios) for p in prof]
        # Cross-axis fit selection uses only the fitting partition.
        value=min(a[0] for a in axes)+.3*sum(a[0] for a in axes)
        shapes.append((value,strength,corrected,prof,axes))
    if not shapes:
        return None,dict(accepted=False,reason="no supported rectification")
    _,strength,h,prof,axes=max(shapes,key=lambda s:s[0])
    transform=np.eye(3)
    tested=[]
    for i,(fit,pitch,phase) in enumerate(axes):
        transform[i,i]=pitch
        transform[i,2]=phase
        tested.append(dict(fit=fit,pitch=pitch,phase=phase,**harmonic_margin(prof[i],pitch,phase)))
    winner=h@transform
    sides=texture(winner,mask,rgb,split)
    hits=[sum(p["ok"] for p in side) for side in sides]
    failures=[]
    if not all(a["validation"]>=.06 for a in tested):
        failures.append("far seams on both axes")
    if not all(a["margin"]>=.015 for a in tested):
        failures.append("held-out half/double challenge")
    if min(hits)<1:
        failures.append("whole atlas face in each region")
    audit=dict(accepted=not failures,failures=failures,axes=tested,atlas_hits=hits,atlas=sides,
               split_x=split.tolist(),perspective=strength,homography=winner.tolist(),
               shape_fits=[dict(strength=s[1],fit=s[0],axes=s[4]) for s in shapes])
    return winner,audit


def run(job):
    group,case,prior=job
    name=case["id"]
    baseline=BASE/group/"large-contraction"
    rel=Path("sky")/name/"charts.json"
    packet=read(baseline/"charts"/rel)
    image=Path(case["variants"]["sky"]["image"])
    assert digest(image)==prior["source"]==packet["source_sha256"] and digest(baseline/"charts"/rel)==prior["base"]
    rgb=np.asarray(Image.open(image).convert("RGB"))
    surf=data(group)[0]/"sky"/name/"source-handoff/surfaces.json"
    assert digest(surf)==prior["mask"]
    planes={p["id"]:p for p in read(surf)["planes"]}
    models={a:[] for a in ARMS}
    audits={}
    for old in packet["charts"]:
        cid=old["id"]
        proof=prior["charts"].get(cid,{})
        proposed=None
        audit={}
        if proof.get("peer_guard"):
            alpha,(x,y,x1,y1)=_native_alpha(planes[cid]["support"],rgb.shape[1::-1])
            mask=np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1]=np.asarray(alpha)>0
            winner,audit=search(old,list(proof["peer_ratios"].values()),mask,rgb)
            if winner is not None:
                proposed=extend(old,winner,mask,rgb)
                audit.update(before=len(old["cells"]),after=len(proposed["cells"]),peer_ratios=proof["peer_ratios"])
            audits[cid]=audit
        for arm in ARMS:
            use=proposed is not None and (arm=="relaxed" or (arm=="verified" and audit["accepted"]))
            models[arm].append(proposed if use else old)
    arms={}
    for arm in ARMS:
        dst=OUT/group/arm
        changed=models[arm]!=packet["charts"]
        if changed or (arm=="bypass" and audits):
            save(dst/"charts"/rel,dict(packet,charts=models[arm]))
            with patch.object(combined,"inputs",data):
                combined.run((group,case,dst/"charts",dst/"assembly"))
            root=dst/"assembly"
        else:
            root=baseline/"assembly"
        arms[arm]=dict(changed=changed,root=str(root),pack=digest(root/PREFIX/name/"reconstruction.zip"))
    save(OUT/group/"audit"/(name+".json.gz"),dict(id=name,group=group,source=prior["source"],
        mask=prior["mask"],baseline_chart=prior["base"],charts=audits,arms=arms))
    if audits:
        print(group,name,[(c,a.get("atlas_hits"),a.get("failures"),a.get("after")) for c,a in audits.items()],flush=True)


def work():
    # Read only source audit fields from the previous experiment; never pass score rows to the search.
    receipt={(r["audit"]["panel"],r["id"]):r["audit"] for r in read(PREVIOUS)}
    return [(g,c,receipt[g,c["id"]]) for g,c in jobs()]


def report(items):
    rows=[]
    for g,c,_ in items:
        n=c["id"]
        audit=read(OUT/g/"audit"/(n+".json.gz"))
        roots=dict(base=BASE/g/"large-contraction/assembly")
        roots.update({a:Path(v["root"]) for a,v in audit["arms"].items()})
        row=summarize([c],roots,PREFIX)["rows"][0]
        assert row["score"]["base"]==row["score"]["bypass"]
        for a,root in roots.items():
            score=read(root/PREFIX/n/"score.json")
            base=read(roots["base"]/PREFIX/n/"score.json")
            assert score["reference_sha256"]==base["reference_sha256"]
            if a=="bypass":
                assert read(root/PREFIX/n/"placements.json")==read(roots["base"]/PREFIX/n/"placements.json")
            if a in audit["arms"]:
                audit["arms"][a].update(selected=score["selected_origins"],yaw=score["yaw"],joins=score["joins"])
        rows.append(dict(**row,audit=audit))
    save(REVIEW/"conditions.json.gz",rows)
    totals={}
    for cohort in ("all","sky",*sorted({r["dataset"] for r in rows})):
        subset=[r for r in rows if cohort in ("all",r["audit"]["group"],r["dataset"])]
        if not subset:
            continue
        options=best([r["score"] for r in subset],["base","relaxed"])
        sums={}
        for arm in ("base",*ARMS,"reference-chooser"):
            metrics={}
            for f in ("geometry","scores"):
                for k in ("strict","facade"):
                    total=Counter()
                    for r,o in zip(subset,options):
                        v=r["score"][o if arm=="reference-chooser" else arm][f][k]
                        total.update({field:v[field] for field in ("correct","missed","extras")})
                    total["f1"]=2*total["correct"]/(2*total["correct"]+total["missed"]+total["extras"])
                    metrics[f+"_"+k]=dict(total)
            sums[arm]=metrics
        totals[cohort]=sums
    save(REVIEW/"summary.json",totals)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filter",default="")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--report",action="store_true")
    a=p.parse_args()
    items=[j for j in work() if a.filter in j[1]["id"]]
    if a.report:
        report(items)
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            list(pool.map(run,items))
