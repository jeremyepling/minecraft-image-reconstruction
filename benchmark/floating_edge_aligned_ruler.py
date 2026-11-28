"""Use lines from the observed wall to estimate its vanishing points before testing whole tiles."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_peer_atlas_search import search
from benchmark.floating_peer_recovery import (BASE, PRIOR, PREFIX, ROOT, WORKTREES, combined, data, digest,
    extend, full_face, materials, project, read, save, summarize, work, _native_alpha)
from benchmark.floating_peer_material_region import propagate
from benchmark.floating_one_face_scores import best
from engine.geometry import UNIT

OUT=ROOT/"runs/edge-aligned-ruler"
REVIEW=ROOT/"benchmark/reviews/floating-edge-aligned-ruler-01"
RECOVERY=WORKTREES/"floating-peer-refit-recovery/runs/peer-refit-recovery"
ARMS=("line-only","refit","refit-material","verified","fine","fine-material","fine-verified")


def gather(rgb,mask,old):
    y,x=np.nonzero(mask)
    lo=np.maximum(0,[x.min()-4,y.min()-4])
    hi=np.minimum(mask.shape[::-1],[x.max()+5,y.max()+5])
    crop=rgb[lo[1]:hi[1],lo[0]:hi[0]]
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY))[0]
    partitions=[[],[]]
    if lines is None:
        return partitions
    support=cv2.dilate(mask.astype('B'),np.ones((9,9),'B')).astype(bool)
    interior=cv2.erode(mask.astype('B'),np.ones((13,13),'B')).astype(bool)
    for line in lines[:,0]:
        ends=line.reshape(2,2)+lo+.5
        step=ends[1]-ends[0]
        length=np.linalg.norm(step)
        if length<max(22.,.05*(hi-lo).max()):
            continue
        center=ends.mean(0)
        points=(ends[0]+np.linspace(.08,.92,11)[:,None]*step).astype(int)
        xx,yy=points.T
        if support[yy,xx].mean()<.8:
            continue
        # Long colored beams and silhouette sections receive priority over repeated short marks.
        rim=float((~interior[yy,xx]).mean())
        uv=project(np.linalg.inv(old),center)
        axes=project(old,uv+np.eye(2))-center
        cosine=abs(axes@step)/(np.linalg.norm(axes,axis=1)*length)
        axis=int(np.argmax(cosine))
        if cosine[axis]<np.cos(np.radians(25)):
            continue
        weight=float(min(length,180.)*(1+.6*rim))
        partitions[axis].append(dict(ends=ends.tolist(),weight=weight,rim=rim,length=float(length)))
    return partitions


def vanishing(segments,center,scale):
    seg=sorted(segments,key=lambda s:-s["weight"])[:64]
    if len(seg)<3:
        return None,dict(accepted=False,segments=len(seg))
    pairs=(np.array([s["ends"] for s in seg])-center)/scale
    mid=pairs.mean(1)
    direction=pairs[:,1]-pairs[:,0]
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    norm=np.stack((-direction[:,1],direction[:,0]),axis=1)
    lines=np.column_stack((norm,-(norm*mid).sum(1)))
    hypotheses=[np.r_[d,0.] for d in direction]
    hypotheses.extend(np.cross(lines[i],lines[j]) for i in range(len(lines)) for j in range(i))
    vp=np.array(hypotheses)
    vp=vp[np.linalg.norm(vp,axis=1)>1.e-10]
    vp/=np.linalg.norm(vp,axis=1)[:,None]
    weight=np.array([s["weight"] for s in seg])
    def errors(v):
        toward=v[None,:,:2]-mid[:,None,:]*v[None,:,2:]
        return np.abs(np.einsum("ij,ikj->ik",norm,toward))/np.maximum(np.linalg.norm(toward,axis=2),1.e-8)
    e=errors(vp)
    robust=np.maximum(0,1-(e/np.sin(np.radians(2.)))**2)
    i=int(np.argmax(weight@robust))
    winner=vp[i]
    keep=e[:,i]<np.sin(np.radians(2.))
    # Require distinct parallels with both line-family width and spatial span.
    offsets=mid@norm[np.argmax(weight*keep)]
    spread=float(np.ptp(offsets[keep])*scale) if keep.any() else 0.
    accepted=bool(keep.sum()>=3 and spread>=20 and weight[keep].sum()>=.37*weight.sum())
    if accepted:
        for _ in range(2):
            err=errors(winner[None])[:,0]
            v=weight*np.maximum(0,1-(err/np.sin(np.radians(2.)))**2)
            _,_,vh=np.linalg.svd(lines*np.sqrt(v)[:,None],full_matrices=False)
            winner=vh[-1]
    result=np.r_[winner[:2]*scale+center*winner[2],winner[2]]
    return (result if accepted else None),dict(accepted=accepted,lines=seg,inliers=keep.tolist(),spread=spread,
                    fraction=float(weight[keep].sum()/weight.sum()),vp=result.tolist())


def aligned(h,vanish,anchor):
    trans=np.eye(3)
    trans[:2,2]=anchor
    b=h@trans
    b/=b[2,2]
    xy=b[:2,2]
    jac=b[:2,:2]-xy[:,None]*b[2,:2]
    for i,point in enumerate(vanish):
        if point is None:
            continue
        d=point[:2]-xy*point[2]
        unit=np.sign(jac[:,i]@d)*np.linalg.norm(jac[:,i])/np.linalg.norm(d)
        b[:,i]=unit*point
    trans[:2,2]=-anchor
    return b@trans


def middle(offers,mask,rgb,split):
    # Fitting uses left 40% by source X. This selection uses the middle 20%.
    # The independent rightmost 40% is read only after the candidate is chosen.
    sorted_offers=sorted(offers,key=lambda p:(p[5]["ok"],p[0]),reverse=True)
    options=[]
    for proposal in sorted_offers[:8]:
        ranking,cv,h,pitches,start,p=proposal
        xy=np.column_stack(np.nonzero(mask)[::-1])+.5
        band=xy[(xy[:,0]>=split[0])&(xy[:,0]<=split[1])]
        uv=np.unique(np.floor(project(np.linalg.inv(h),band)).astype(int),axis=0)
        center=project(h,uv+.5)
        uv=uv[(center[:,0]>=split[0])&(center[:,0]<=split[1])]
        uv=uv[(np.max(abs(uv),axis=1)>=2)]
        probes=uv[np.linspace(0,len(uv)-1,min(16,len(uv))).astype(int)] if len(uv) else []
        matches=[full_face(h,pos,mask,rgb) for pos in probes]
        hits=[m for m in matches if m["ok"]]
        strongest=max((m.get("correlation",0) for m in matches),default=0)
        score=ranking+.22*min(2,len(hits))+.2*strongest
        options.append((score,proposal,dict(near=p,mid_hits=hits,probes=len(matches),pitch=list(pitches),
                                          max_mid_corr=float(strongest))))
    _,choice,record=max(options,key=lambda s:s[0])
    return choice,dict(winner=record,options=[s[2] for s in options])


def correct_material(chart,old,peers,mask,rgb):
    variants,mat=materials(chart,old,peers,mask,rgb)
    proof=full_face(np.asarray(chart["H"]),np.zeros(2),mask,rgb)
    secondary=[tuple(x["uv"]) for x in mat["cells"] if proof["ok"] and
               x["vote"]==proof["material"].removeprefix("minecraft:")]
    return propagate(variants["overlap"],mask,rgb,secondary)[0]


def run(job):
    g,c,prior=job
    n=c["id"]
    rel=Path("sky")/n/"charts.json"
    flat=BASE/g/"large-contraction"
    old=read(flat/"charts"/rel)
    receipt=read(PRIOR/g/"audit"/(n+".json.gz"))
    changed=receipt["arms"]["relaxed"]["changed"]
    peer=PRIOR/g/"relaxed" if changed else flat
    rec=RECOVERY/g/"corroborated"
    if not (rec/"charts"/rel).exists():
        rec=peer
    roots=dict(original=flat/"assembly",peer=peer/"assembly",recovery=rec/"assembly")
    findings={}
    if changed:
        packet=read(peer/"charts"/rel)
        recovered=read(rec/"charts"/rel)
        image=Path(c["variants"]["sky"]["image"])
        rgb=np.asarray(Image.open(image).convert("RGB"))
        surf=data(g)[0]/"sky"/n/"source-handoff/surfaces.json"
        assert digest(image)==prior["source"] and digest(surf)==prior["mask"]
        planes={v["id"]:v for v in read(surf)["planes"]}
        packets={a:[] for a in ARMS}
        for ch,base,keep in zip(packet["charts"],old["charts"],recovered["charts"],strict=True):
            cid=ch["id"]
            if cid not in receipt["charts"]:
                for a in ARMS:
                    packets[a].append(ch)
                continue
            alpha,(x,y,x1,y1)=_native_alpha(planes[cid]["support"],rgb.shape[1::-1])
            mask=np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1]=np.asarray(alpha)>0
            h=np.asarray(ch["H"])
            initial=np.asarray(base["H"])
            yx=np.column_stack(np.nonzero(mask))
            center=np.median(yx,axis=0)[::-1]
            scale=np.ptp(yx,axis=0).max()
            segments=gather(rgb,mask,h)
            pairs=[vanishing(parts,center,scale) for parts in segments]
            vps=[item[0] for item in pairs]
            # Preserve native center and local scale of the actual verified origin tile.
            line_h=aligned(h,vps,np.array([.5,.5]))
            line_chart=extend(ch,line_h,mask,rgb)
            shape=aligned(initial,vps,project(np.linalg.inv(initial),center))
            ratios=list(receipt["charts"][cid]["peer_ratios"].values())
            answer,audit=search(dict(base,H=shape.tolist()),ratios,mask,rgb,middle)
            target=extend(ch,answer,mask,rgb) if answer is not None else line_chart
            peers=[v for v in old["charts"] if v["id"]!=cid]
            colored=correct_material(target,base,peers,mask,rgb)
            accepted=bool(all(v is not None for v in vps) and audit["accepted"])
            arms={"line-only":line_chart,"refit":target,"refit-material":colored,"verified":colored if accepted else keep}
            from benchmark.floating_edge_repeat_fit import repeat_fit
            fine=target
            refined={"accepted":False}
            if answer is not None and audit.get("near",{}).get("ok"):
                improved,refined=repeat_fit(answer,audit["near"]["material"],mask,rgb)
                fine=extend(ch,improved,mask,rgb)
            fine_colored=correct_material(fine,base,peers,mask,rgb)
            arms.update(fine=fine,**{"fine-material":fine_colored,"fine-verified":fine_colored if
                        all(v is not None for v in vps) and refined["accepted"] else keep})
            for a,v in arms.items():
                packets[a].append(v)
            findings[cid]=dict(lines=[item[1] for item in pairs],skew_h=line_h.tolist(),ruler=audit,refined=refined,
                               counts={a:len(v["cells"]) for a,v in arms.items()},accepted=accepted)
        for a in ARMS:
            dst=OUT/g/a
            save(dst/"charts"/rel,dict(packet,charts=packets[a]))
            with patch.object(combined,"inputs",data):
                combined.run((g,c,dst/"charts",dst/"assembly"))
            roots[a]=dst/"assembly"
    else:
        roots.update({a:roots["recovery"] for a in ARMS})
    r=summarize([c],roots,PREFIX)["rows"][0]
    r.update(panel=g,source=prior["source"],mask=prior["mask"],charts=findings,packs={a:digest(root/PREFIX/n/"reconstruction.zip") for a,root in roots.items()},
             selected={a:read(root/PREFIX/n/"score.json")["selected_origins"] for a,root in roots.items()})
    save(OUT/g/"rows"/(n+".json.gz"),r)
    if changed:
        print(n,[(a,r["score"][a]["geometry"]["strict"]["correct"],r["score"][a]["geometry"]["strict"]["extras"],
                  r["score"][a]["scores"]["strict"]["correct"]) for a in roots],flush=True)


def report():
    rows=[read(OUT/g/"rows"/(c["id"]+".json.gz")) for g,c,_ in work()]
    save(REVIEW/"conditions.json.gz",rows)
    sums={}
    for group in ("all","sky",*sorted({r["dataset"] for r in rows})):
        subset=[r for r in rows if group in ("all",r["panel"],r["dataset"])]
        chooser=best([r["score"] for r in subset],["recovery",*ARMS])
        methods={}
        for arm in ("original","peer","recovery",*ARMS,"reference-chooser"):
            metrics={}
            for f in ("geometry","scores"):
                for k in ("strict","facade"):
                    cnt=Counter()
                    for r,a in zip(subset,chooser):
                        v=r["score"][a if arm=="reference-chooser" else arm][f][k]
                        cnt.update({i:v[i] for i in ("correct","missed","extras")})
                    cnt["f1"]=2*cnt["correct"]/(2*cnt["correct"]+cnt["missed"]+cnt["extras"])
                    metrics[f+"_"+k]=dict(cnt)
            methods[arm]=metrics
        sums[group]=methods
    save(REVIEW/"summary.json",sums)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report",action="store_true")
    a=p.parse_args()
    if a.report:
        report()
    else:
        with ProcessPoolExecutor(5) as pool:
            list(pool.map(run,work()))
