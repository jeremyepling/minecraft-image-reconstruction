"""Cross-check revised wall cells against source-local material and boundary observations."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_peer_ruler_refit import (BASE, PREFIX, ROOT, WORKTREES, combined, data,
    digest, extend, line_score, profiles, project, read, save, summarize, work, _native_alpha)
from benchmark.floating_verified_wall_expansion import full_face
from benchmark.floating_one_face_scores import best
from engine.geometry import UNIT

PRIOR = WORKTREES/"floating-peer-ruler-refit/runs/peer-ruler-refit/atlas-first"
OUT = ROOT/"runs/peer-refit-recovery"
REVIEW = ROOT/"benchmark/reviews/floating-peer-refit-recovery-01"
ARMS = ("overlap","witness","anchor-region","region-overlap","corroborated","partial","phase","phase-material")
SITES = np.stack(np.meshgrid((np.arange(9)+.5)/9,(np.arange(9)+.5)/9),-1)


def observed(rgb,mask,h,uv):
    xy=project(h,uv+SITES)
    ix,iy=np.floor(xy).astype(int).transpose(2,0,1)
    valid=(ix>=0)&(iy>=0)&(ix<mask.shape[1])&(iy<mask.shape[0])
    ix=ix.clip(0,mask.shape[1]-1)
    iy=iy.clip(0,mask.shape[0]-1)
    return xy,rgb[iy,ix].astype(float),valid & mask[iy,ix]


def key(cell):
    return cell["u"],cell["v"]


def label(cell):
    return cell["block_id"].removeprefix("minecraft:")


def descriptor(pixels):
    return np.quantile(pixels,[.15,.5,.85],axis=0)


def error(a,b):
    # Restrict exposure variation. Retain hues and within-tile contrast.
    scale=np.clip(np.sum(a[1]*b[1])/max(1,np.sum(b[1]**2)),.82,1.22)
    return np.mean(abs(a-scale*b))+.12*abs(scale-1)*np.mean(b)


def materials(chart,old,peers,mask,rgb):
    h=np.asarray(chart["H"])
    original={key(c):c for c in old["cells"]}
    inv=np.linalg.inv(old["H"])
    witnesses=[]
    for peer in [old,*peers]:
        ph=np.asarray(peer["H"])
        for cell in peer["cells"]:
            if cell.get("fallback") or cell["confidence"]<.59:
                continue
            uv=np.array(key(cell))
            xy,pix,visible=observed(rgb,mask,ph,uv)
            # A source neighbor contributes only when it overlaps the same observed mask.
            if visible.mean()<.92:
                continue
            proof=full_face(ph,uv,mask,rgb)
            if proof["ok"]:
                witnesses.append((label({"block_id":proof["material"]}),descriptor(pix[visible]),
                                  xy[4,4],proof["correlation"]))
    evidence=[]
    alternatives={a:[] for a in ("overlap","witness")}
    for cell in chart["cells"]:
        uv=np.array(key(cell))
        xy,pix,visible=observed(rgb,mask,h,uv)
        votes=Counter()
        normalized=descriptor(pix[visible])
        old_uv=np.floor(project(inv,xy[visible])).astype(int)
        for z,color in zip(old_uv,pix[visible]):
            source=original.get(tuple(z))
            if source is not None and not source.get("fallback") and source["confidence"]>=.30:
                if np.mean(abs(color-normalized[1]))<=23:
                    votes[label(source)]+=source["confidence"]**2
        ranked=votes.most_common(2)
        consensus=None
        if ranked and sum(votes.values())>=4:
            winner,weight=ranked[0]
            if weight>=.45*sum(votes.values()) and (len(ranked)==1 or weight>=1.5*ranked[1][1]):
                consensus=winner
        chosen=consensus
        # Neighbor evidence must match photometry, lie nearby at the fitted scale and have
        # independent repeat support. Reject ties between plausible source material classes.
        local={}
        pitch=np.mean(np.linalg.norm(project(h,uv+.5+np.eye(2))-xy[4,4],axis=1))
        for material,ref,center,confidence in witnesses:
            distance=np.linalg.norm(center-xy[4,4])/max(1,pitch)
            if distance<=4:
                loss=error(normalized,ref)+.4*distance
                local.setdefault(material,[]).append(float(loss))
        scored=sorted((float(np.mean(sorted(v)[:2])),k) for k,v in local.items() if len(v)>=2)
        if scored and scored[0][0]<8 and (len(scored)==1 or scored[1][0]-scored[0][0]>=1.5):
            chosen=scored[0][1]
        for arm,new_id in (("overlap",consensus),("witness",chosen)):
            alternatives[arm].append(cell if new_id is None or new_id==label(cell)
                                      else dict(cell,block_id="minecraft:"+new_id,fallback=False))
        evidence.append(dict(uv=key(cell),old=label(cell),vote=consensus,witness=chosen))
    return {a:dict(chart,cells=c) for a,c in alternatives.items()},dict(witness_count=len(witnesses),cells=evidence)


def recover_partial(chart,old,mask,rgb):
    """Test visibly clipped unit cells with a connected neighbor and old covered image evidence."""
    cells={key(c):c for c in chart["cells"]}
    old_cells={key(c):c for c in old["cells"]}
    inv=np.linalg.inv(old["H"])
    h=np.asarray(chart["H"])
    offered=set()
    for u,v in cells:
        offered.update(((u+1,v),(u-1,v),(u,v+1),(u,v-1)))
    new=[]
    for uv in sorted(offered-cells.keys()):
        xy,pixels,visible=observed(rgb,mask,h,np.asarray(uv))
        if not .40<=visible.mean()<.85:
            continue
        # Demand a single observed component crossing two or more edges of this tile;
        # both neighboring source cells must agree on the surface material.
        _,cc=cv2.connectedComponents(visible.astype("B"),connectivity=4)
        parts=np.bincount(cc.ravel())
        if len(parts)<2:
            continue
        component=cc==np.argmax(parts[1:])+1
        if component.sum()<.9*visible.sum():
            continue
        u,v=uv
        neighbors=[cells[q] for q,edge in (((u-1,v),component[:,0]),((u+1,v),component[:,-1]),
                   ((u,v-1),component[0]),((u,v+1),component[-1])) if q in cells and edge.sum()>=3]
        votes=Counter(label(c) for c in neighbors)
        if not votes or votes.most_common(1)[0][1]<2:
            continue
        material=votes.most_common(1)[0][0]
        old_samples={tuple(p) for p in np.floor(project(inv,xy[component])).astype(int)}
        corroboration=[old_cells[p] for p in old_samples if p in old_cells and label(old_cells[p])==material]
        if len(corroboration)<2:
            continue
        new.append(dict(u=u,v=v,block_id="minecraft:"+material,confidence=float(np.mean([c["confidence"] for c in corroboration])),
                        mask_coverage=float(visible.mean()),fallback=False))
    return dict(chart,cells=chart["cells"]+new),new


def tweak(chart,mask,rgb):
    h=np.array(chart["H"])
    xy=np.column_stack(np.nonzero(mask)[::-1])+.5
    split=np.quantile(xy[:,0],[.4,.6])
    prof=profiles(h,mask,rgb,split)
    phases=np.arange(-3,4)/8
    selected=[]
    audit=[]
    for p in prof:
        fit,nfit=line_score(p,1.,phases,0)
        far,nfar=line_score(p,1.,phases,1)
        # Require consistent seam improvement in both disjoint source partitions,
        # evaluated at the same candidate phase. Keep the inherited phase on sparse evidence.
        margin=np.minimum(fit-fit[3],far-far[3])
        best_idx=int(np.argmax(margin-.02*abs(phases)))
        index=best_idx if margin[best_idx]>=.04 else 3
        selected.append(float(phases[index]))
        audit.append(dict(phases=phases.tolist(),fit=fit.tolist(),far=far.tolist(),chosen=float(phases[index]),
                          min_gain=float(margin[best_idx]),fit_lines=nfit.tolist(),far_lines=nfar.tolist()))
    transform=np.eye(3)
    transform[:2,2]=selected
    return (extend(chart,h@transform,mask,rgb) if any(selected) else chart),audit


def run(job):
    g,c,prior=job
    n=c["id"]
    base=BASE/g/"large-contraction"
    rel=Path("sky")/n/"charts.json"
    original=read(base/"charts"/rel)
    prev=read(PRIOR/g/"audit"/(n+".json.gz"))
    changed=prev["arms"]["relaxed"]["changed"]
    source=PRIOR/g/"relaxed" if changed else base
    p=read(source/"charts"/rel)
    assert p["source_sha256"]==original["source_sha256"]==prior["source"]
    packets={a:[] for a in ARMS}
    decisions={}
    if changed:
        rgb=np.array(Image.open(c["variants"]["sky"]["image"]).convert("RGB"))
        source_path=data(g)[0]/"sky"/n/"source-handoff/surfaces.json"
        assert digest(source_path)==prior["mask"]
        planes={a["id"]:a for a in read(source_path)["planes"]}
        for chart,old in zip(p["charts"],original["charts"],strict=True):
            cid=chart["id"]
            assert cid==old["id"]
            if cid not in prev["charts"]:
                for arm in ARMS:
                    packets[arm].append(chart)
                continue
            alpha,(x,y,x1,y1)=_native_alpha(planes[cid]["support"],rgb.shape[1::-1])
            mask=np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1]=np.asarray(alpha)>0
            peers=[v for v in original["charts"] if v["id"]!=cid]
            recolor,mat=materials(chart,old,peers,mask,rgb)
            from benchmark.floating_peer_material_region import propagate
            region,region_proof=propagate(chart,mask,rgb)
            region_overlap,_=propagate(recolor["overlap"],mask,rgb)
            secondary=[tuple(c["uv"]) for c in mat["cells"] if region_proof.get("accepted") and
                       c["vote"]==region_proof["proof"]["material"].removeprefix("minecraft:")]
            corroborated,spread=propagate(recolor["overlap"],mask,rgb,secondary)
            partial,added=recover_partial(corroborated,old,mask,rgb)
            shifted,details=tweak(chart,mask,rgb)
            both=materials(shifted,old,peers,mask,rgb)[0]["witness"] if shifted is not chart else recolor["witness"]
            for arm,entry in dict(**recolor,phase=shifted,**{"phase-material":both,"anchor-region":region,"region-overlap":region_overlap,"corroborated":corroborated,"partial":partial}).items():
                packets[arm].append(entry)
            decisions[cid]=dict(material=mat,phase=details,region=region_proof,secondary=secondary,corroborated=spread,added=added)
    roots={"original":base/"assembly","refit":source/"assembly"}
    for arm in ARMS:
        dest=OUT/g/arm
        if changed and packets[arm]!=p["charts"]:
            save(dest/"charts"/rel,dict(p,charts=packets[arm]))
            with patch.object(combined,"inputs",data):
                combined.run((g,c,dest/"charts",dest/"assembly"))
            roots[arm]=dest/"assembly"
        else:
            roots[arm]=source/"assembly"
    row=summarize([c],roots,PREFIX)["rows"][0]
    ref=dict(read(roots["refit"]/PREFIX/n/"placements.json"))
    packed={}
    for a,root in roots.items():
        receipt=read(root/PREFIX/n/"score.json")
        packed[a]=dict(pack=receipt["pack_sha256"],chart=receipt["chart_sha256"],translation=row["score"][a]["translation"])
        if a in ("overlap","witness","anchor-region","region-overlap","corroborated"):
            repl=read(root/PREFIX/n/"placements.json")
            for field in ("local","aligned"):
                assert [v for v,_ in repl[field]]==[v for v,_ in ref[field]]
    row.update(panel=g,source=prior["source"],mask=prior["mask"],decisions=decisions,arms=packed)
    save(OUT/g/"rows"/(n+".json.gz"),row)
    if decisions:
        print(n,[(a,row["score"][a]["geometry"]["strict"]["correct"],row["score"][a]["geometry"]["strict"]["extras"],
                  row["score"][a]["scores"]["strict"]["correct"]) for a in roots],flush=True)


def report():
    rows=[read(OUT/g/"rows"/(c["id"]+".json.gz")) for g,c,_ in work()]
    save(REVIEW/"conditions.json.gz",rows)
    sums={}
    for cohort in ("all","sky",*sorted({r["dataset"] for r in rows})):
        subset=[r for r in rows if cohort in ("all",r["panel"],r["dataset"])]
        choices=best([r["score"] for r in subset],["refit",*ARMS])
        cohort_scores={}
        for arm in ("original","refit",*ARMS,"geometry-chooser"):
            values={}
            for f in ("geometry","scores"):
                for k in ("strict","facade"):
                    total=Counter()
                    for r,o in zip(subset,choices):
                        m=r["score"][o if arm=="geometry-chooser" else arm][f][k]
                        total.update({i:m[i] for i in ("correct","missed","extras")})
                    total["f1"]=2*total["correct"]/(2*total["correct"]+total["missed"]+total["extras"])
                    values[f+"_"+k]=dict(total)
            cohort_scores[arm]=values
        sums[cohort]=cohort_scores
    save(REVIEW/"summary.json",sums)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report",action="store_true")
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--filter",default="")
    args=p.parse_args()
    if args.report:
        report()
    else:
        items=[j for j in work() if args.filter in j[1]["id"]]
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run,items))
