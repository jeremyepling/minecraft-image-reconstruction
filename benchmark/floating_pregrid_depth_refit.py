"""Fit rulers inside source depth regions before cell enumeration on the adopted expanded benchmark."""
import argparse
from itertools import combinations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_depth_before_atlas as depth
from benchmark.floating_package_rechecks import (PACKAGE,PREFIX,ROOT,WORKTREES,border_filter,
    combined,data,digest,jobs,native_mask,project,read,save,sky_support,summarize)
from benchmark.floating_one_face_scores import best
from benchmark.mask_blocks import decoded_pack

PIN=WORKTREES/"floating-visible-builds/benchmark/reviews/floating-expanded-benchmark-01"
RECHECK=WORKTREES/"floating-package-rechecks/runs/package-rechecks"
OUT=ROOT/"runs/pregrid-depth-refit"
REVIEW=ROOT/"benchmark/reviews/floating-pregrid-depth-refit-01"
ARMS=("noop","split-only","complete","partial","fresh-only","connected","rim-corner")
FINAL=("combined","nonshrinking")


def workload():
    manifest=read(PIN/"manifest.json")
    assert digest(PIN/manifest["archive"]["path"])==manifest["archive"]["sha256"]
    pinned={(r["panel"],r["id"]):r for r in manifest["conditions"]}
    known=read(ROOT/"benchmark/reviews/floating-depth-before-atlas-01/conditions.json.gz")
    reviewed={(r["panel"],r["id"]):list(r["audit"]) for r in known if r["audit"]}
    cases=jobs()
    transfer=set()
    # Freeze both angles of one whole layout per extra panel before scoring.
    for p in ("holdout","holdout2","holdout5"):
        ids=sorted(c["id"] for g,c in cases if g==p)
        family=ids[0].rsplit("-",2)[0]
        transfer.update((p,n) for n in ids if n.rsplit("-",2)[0]==family)
    assert len(pinned)==len(cases)==84 and len(reviewed)==3 and len(transfer)==6
    return [(g,c,pinned[g,c["id"]],reviewed.get((g,c["id"]),[]),(g,c["id"]) in transfer) for g,c in cases]


def fields(panel,case,src,size):
    folder=depth.prior.NATIVE.parent/"fields"/case["dataset"]/case["id"]
    archived=WORKTREES/"full-expanded-controls/runs/expanded-baseline-01"/case["dataset"]/case["id"]/"handoffs"
    if panel!="original" or archived.exists() or (folder/"receipt.json").exists():
        return depth.source_fields(panel,case,src,size)
    # The historical v2 producer wrote per-case npz arrays, while its source RGB
    # and model fingerprints live in the source handoff and paired crop receipt.
    manifest=read(src/"sky"/case["id"]/"source-handoff/build-manifest.json")
    paired=read(depth.prior.NATIVE.parent/"pairs"/case["dataset"]/(case["id"]+".json"))
    assert manifest["image_sha256"]==case["rgb_sha256"]==paired["source_sha256"]
    values={}
    sha={}
    for kind in ("normal","depth"):
        assert manifest["model_sha256"][kind]==depth.MODELS[kind][2]
        path=folder/(kind+".npz")
        with np.load(path) as raw:
            values[kind]=cv2.resize(raw["value"].astype("f"),size)
        sha[kind]=digest(path)
    return depth._unit(values["normal"]),values["depth"],sha


def part_fits(chart,region,rgb,normal,whole,vanishing):
    frames={}
    if normal@chart["normal_camera"]>.92:
        frames["source quadrilateral"]=depth.prior.rectify(whole["quad"],whole["evidence"].get("aspect"),rgb,region)
    frames["source line tangents"]=depth.tangent_frame(rgb,region,normal,vanishing)
    options={}
    brief={}
    for kind,frame in frames.items():
        if frame is None:
            continue
        # Strip all old grid cells and H before the new pitch, phase and coverage search.
        carrier={k:v for k,v in chart.items() if k not in ("H","export_uv","export_cells")}
        carrier.update(cells=[],normal_camera=normal.tolist())
        found,measure=depth.prior.region_ruler(carrier,region,rgb,None,None,rectified=frame)
        brief[kind]=dict(reason=measure.get("reason"),maximum=measure.get("max_hits"))
        for arm,value in found.items():
            evidence=measure["evidence"][arm]
            hits=evidence["holdout"]
            rank=(len(hits),np.mean([h["corr"] for h in hits]))
            if arm not in options or rank>options[arm][0]:
                options[arm]=(rank,value,dict(frame=kind,holdout=hits,pitch=evidence["pitch"]))
    return options,brief


def replay(g,c,charts_dir,dest,membership):
    """Keep physical perpendicular shared-edge joins; prevent same-parent coplanar reconstruction."""
    solver=combined.solver
    old=solver.coplanar_joins
    excluded=[]
    def guarded(charts,tol):
        result=[]
        for join in old(charts,tol):
            a,b=[charts[join[k]]["id"] for k in ("i","j")]
            if a in membership and membership[a]==membership.get(b):
                excluded.append([a,b])
            else:
                result.append(join)
        return result
    with patch.object(solver,"coplanar_joins",guarded),patch.object(combined,"inputs",data):
        combined.run((g,c,charts_dir,dest))
    return excluded


def corner_proof(selected,joins,fresh,membership):
    """A new chart must close a three-surface, nonparallel corner in the actual winning group."""
    edges={frozenset(j["chart_ids"]):j["edges"] for j in joins
           if j["kind"]=="edge" and set(j["chart_ids"])<=set(selected)}
    proofs=[]
    for cid in fresh & set(selected):
        for a,b in combinations(sorted(set(selected)-{cid}),2):
            if len({membership.get(i,i) for i in (cid,a,b)})!=3:
                continue
            counts=[edges.get(frozenset(uv),0) for uv in ((cid,a),(cid,b),(a,b))]
            if min(counts)>=1 and max(counts[:2])>=2 and counts[2]>=2:
                proofs.append(dict(fresh=cid,other=[a,b],shared_edges=counts))
    return proofs


def run(job):
    g,c,pin,reviewed,transfer=job
    cv2.setNumThreads(1)
    n=c["id"]
    rel=Path("sky")/n/"charts.json"
    src=data(g)[0]
    package=PACKAGE/g
    verify_files=pin["files"]
    charts_file=OUT/g/"noop/charts"/rel
    baseline_pack=OUT/g/"noop/input"/n/"reconstruction.zip"
    charts_file.parent.mkdir(parents=True,exist_ok=True)
    baseline_pack.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(PIN/"exports.zip") as z:
        for key,target in (("charts",charts_file),("pack",baseline_pack)):
            target.write_bytes(z.read(verify_files[key]["entry"]))
            assert digest(target)==verify_files[key]["sha256"]
    reference=package if pin["resolved_arm"]=="base" else RECHECK/g/pin["resolved_arm"]
    assert digest(reference/"assembly"/PREFIX/n/"reconstruction.zip")==verify_files["pack"]["sha256"]
    roots=dict(adopted=reference/"assembly")
    replay(g,c,OUT/g/"noop/charts",OUT/g/"noop/assembly",{})
    roots["noop"]=OUT/g/"noop/assembly"
    assert decoded_pack(roots["noop"]/PREFIX/n/"reconstruction.zip")==decoded_pack(baseline_pack)
    assert read(roots["noop"]/PREFIX/n/"placements.json")==read(roots["adopted"]/PREFIX/n/"placements.json")
    assert read(roots["noop"]/PREFIX/n/"score.json")["score"]==read(roots["adopted"]/PREFIX/n/"score.json")["score"]
    fit_case(g,c,pin,reviewed,transfer,src,package,charts_file,roots)


def prepare_charts(g,c,pin,reviewed,src,package,min_area=5000):
    n=c['id']
    rel=Path('sky')/n/'charts.json'
    audit={}
    rgb_path=Path(c["variants"]["sky"]["image"])
    assert digest(rgb_path)==pin["source"]["sha256"]
    img=Image.open(rgb_path).convert("RGB")
    rgb=np.array(img);sky=sky_support(img)
    packet=read(package/"charts"/rel)
    assert digest(package/"charts"/rel)==pin["selected_package_chart_sha256"]
    plane_file=src/"sky"/n/"source-handoff/surfaces.json"
    assert digest(plane_file)==pin["surfaces_sha256"]
    planes={v["id"]:v for v in read(plane_file)["planes"]}
    counts={v["id"]:v for v in read(plane_file.with_name("counts.json"))["surfaces"]}
    source_ids=set(v["id"] for v in packet["charts"])
    masks={k:native_mask(p,img.size) for k,p in planes.items() if k in source_ids
           and p.get("normal_camera") is not None and abs(p["normal_camera"][1])<.67 and k in counts}
    if not reviewed:
        target=max(masks,key=lambda k:masks[k].sum(),default=None)
        targets=[target] if target and masks[target].sum()>=min_area else []
    else:
        targets=reviewed
    if targets:
        normal,dist,hashes=fields(g,c,src,img.size)
        vanishing=depth.prior.garden.pencils(rgb,depth.prior.garden.line_segments(rgb))
        candidates={a:[] for a in ("split-only","complete","partial","fresh-only")}
        membership={}
        source_verified=set()
        partial_verified=set()
        region_charts=[]
        for ch in packet["charts"]:
            cid=ch["id"]
            if cid not in targets:
                for charts in candidates.values():
                    charts.append(ch)
                continue
            mask=masks[cid]
            coarse,meta=depth.partition(rgb,mask,normal,dist)
            ids,parts,boundaries=depth.refine_adjacent(rgb,mask,normal,dist,coarse,meta)
            path=OUT/g/"source"/n/(cid+"-ids.png");path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(ids).save(path)
            uv=np.array([(cell["u"]+.5,cell["v"]+.5) for cell in ch["cells"]]).reshape(-1,2)
            center=project(np.asarray(ch["H"]),uv).astype(int)
            idx=ids[center[:,1],center[:,0]]
            details={}
            for part in parts:
                k=part["id"]
                name=f"{cid}~p{k}"
                membership[name]=cid
                inherited=[cell for cell,assignment in zip(ch["cells"],idx) if assignment==k]
                sub=dict(ch,id=name,cells=inherited)
                options,brief=part_fits(sub,ids==k,rgb,np.array(part["normal_camera"]),counts[cid],vanishing)
                details[k]=dict(pixels=int((ids==k).sum()),inherited=len(inherited),evidence=brief,grids={})
                if inherited:
                    candidates["split-only"].append(sub)
                for a,kind in (("complete","refit"),("partial","visible-patches")):
                    if kind in options:
                        _,found,ev=options[kind]
                        candidates[a].append(found)
                        if a=="partial":
                            partial_verified.add(name)
                        if a=="complete":
                            candidates["fresh-only"].append(found)
                            source_verified.add(name)
                        details[k]["grids"][a]=dict(**ev,H=found["H"],cells=len(found["cells"]))
                        region_charts.append((cid,k,a,found))
                    elif inherited:
                        candidates[a].append(sub)
            residual=[cell for cell,i in zip(ch["cells"],idx) if i==0]
            if residual:
                for a in ("split-only","complete","partial"):
                    candidates[a].append(dict(ch,id=cid+"~p0",cells=residual))
            audit[cid]=dict(mask_sha256=digest(path),fields=hashes,parts=details,
                            boundary_count=len(boundaries),residual=len(residual))
        return packet,sky,audit,candidates,membership,source_verified,partial_verified,region_charts
    return None


def fit_case(g,c,pin,reviewed,transfer,src,package,charts_file,roots):
    n=c["id"]
    rel=Path("sky")/n/"charts.json"
    audit={}
    proof=[]
    prepared=prepare_charts(g,c,pin,reviewed,src,package)
    if prepared:
        packet,sky,audit,candidates,membership,source_verified,partial_verified,region_charts=prepared
        decisions={}
        for arm,charts in candidates.items():
            filtered,dropped=border_filter(charts,sky)
            dst=OUT/g/arm
            save(dst/"charts"/rel,dict(packet,charts=filtered))
            blocked=replay(g,c,dst/"charts",dst/"assembly",membership)
            roots[arm]=dst/"assembly"
            result=read(roots[arm]/PREFIX/n/"score.json")
            decisions[arm]=dict(selected=result["selected_origins"],joins=result["joins"],
                                component_sizes=result["component_sizes"],blocked_coplanar=blocked,drops=dropped)
        selected=set(decisions["complete"]["selected"])
        old_selected=set(read(roots["adopted"]/PREFIX/n/"score.json")["selected_origins"])
        # Independent 3D registration is essential: the measured new pieces must
        # belong to the actual selected component without losing its other surfaces.
        normalize=lambda ids:{membership.get(i,i) for i in ids}
        accept=bool(source_verified and source_verified<=selected and
                    normalize(old_selected)<=normalize(selected) and
                    all(any(i in j["chart_ids"] for j in decisions["complete"]["joins"]) for i in source_verified))
        roots["connected"]=roots["complete"] if accept else roots["noop"]
        record=decisions["partial"]
        proof=corner_proof(record["selected"],record["joins"],partial_verified,membership)
        roots["rim-corner"]=roots["partial"] if proof else roots["noop"]
        # Native ownership is read only after options, export and source decision.
        for cid,k,arm,ch in region_charts:
            audit[cid]["parts"][k]["grids"][arm]["native"]=depth.native_grid_score(c,ch)
        audit["decision"]=dict(arms=decisions,verified=sorted(source_verified),accept=accept,rim_corner=proof)
    for a in ARMS:
        roots.setdefault(a,roots["noop"])
    row=summarize([c],roots,PREFIX)["rows"][0]
    row.update(panel=g,audit=audit,reviewed=reviewed,transfer=transfer,source_sha256=pin["source"]["sha256"],
               reference_sha256={k:v["sha256"] for k,v in pin["reference"].items()},
               roots={a:str(p) for a,p in roots.items()},packs={a:digest(p/PREFIX/n/"reconstruction.zip") for a,p in roots.items()})
    save(OUT/g/"rows"/(n+".json.gz"),row)
    if reviewed or transfer or (audit and proof):
        print(g,n,[(a,row["score"][a]["geometry"]["strict"]["correct"],row["score"][a]["geometry"]["strict"]["extras"],
                    row["score"][a]["scores"]["strict"]["correct"]) for a in roots],flush=True)


def finalize(row,case):
    # These are mutually ordered alternatives over already assembled whole exports.
    # Source selection reads physical-join receipts and actual local pack size only.
    decision=row["audit"].get("decision",{})
    accepted=decision.get("accept",False)
    triangle=bool(decision.get("rim_corner"))
    n=row["id"]
    local=lambda arm:decoded_pack(Path(row["roots"][arm])/PREFIX/n/"reconstruction.zip")
    no_shrink=not accepted or len(local("connected"))>=len(local("noop"))
    for arm,condition in (("combined",accepted),("nonshrinking",accepted and no_shrink)):
        use="rim-corner" if triangle else "connected" if condition else "noop"
        for key in ("roots","packs","score","paired"):
            row[key][arm]=row[key][use]
    # Gross exact changes are evaluated in the ordinary arm-specific registration.
    labels=read(Path(case["frame"])/"labels.json")
    blocks={b["id"]:b["block_id"] for b in labels["blocks"]}
    truth={tuple(f["xyz_world"]):blocks[f["block"]] for f in labels["faces"]}
    cache={}
    for arm,root in row["roots"].items():
        if root not in cache:
            placed=read(Path(root)/PREFIX/n/"placements.json")["aligned"]
            cache[root]={tuple(p) for p,b in placed if truth.get(tuple(p))==b}
    orig=cache[row["roots"]["noop"]]
    row["exact_paired"]={arm:dict(gained=len(cache[root]-orig),lost=len(orig-cache[root])) for arm,root in row["roots"].items()}
    row["source_choices"]=dict(connected=accepted,rim=triangle,nonshrinking=no_shrink)
    return row


def report(work):
    rows=[finalize(read(OUT/g/"rows"/(c["id"]+".json.gz")),c) for g,c,*_ in work]
    save(REVIEW/"conditions.json.gz",rows)
    summary={}
    for group in ("all","sky","transfer",*sorted({r["dataset"] for r in rows})):
        subset=[r for r in rows if group in ("all",r["panel"],r["dataset"]) or (group=="transfer" and r["transfer"])]
        chooser=best([r["score"] for r in subset],[*ARMS,*FINAL])
        arms={}
        for arm in (*ARMS,*FINAL,"whole-export-chooser"):
            choices=chooser if arm=="whole-export-chooser" else [arm]*len(subset)
            metrics={}
            for f in ("geometry","scores"):
                for k in ("strict","facade"):
                    t=Counter()
                    for r,a in zip(subset,choices):
                        t.update({i:r["score"][a][f][k][i] for i in ("correct","missed","extras")})
                    t["f1"]=2*t["correct"]/(2*t["correct"]+t["missed"]+t["extras"])
                    metrics[f+"_"+k]=dict(t)
            metrics["paired"]={key:sum(r["paired"][a][key] for r,a in zip(subset,choices))
                               for key in ("gained","lost","added","removed")}
            metrics["exact_paired"]={k:sum(r["exact_paired"][a][k] for r,a in zip(subset,choices)) for k in ("gained","lost")}
            arms[arm]=metrics
        summary[group]=arms
    save(REVIEW/"summary.json",summary)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report",action="store_true")
    p.add_argument("--filter",default="")
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    work=workload()
    if args.report:
        report(work)
    else:
        items=[j for j in work if args.filter in j[1]["id"] and
               not(args.resume and (OUT/j[0]/"rows"/(j[1]["id"]+".json.gz")).exists() and
                   "rim-corner" in read(OUT/j[0]/"rows"/(j[1]["id"]+".json.gz"))["score"])]
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run,items))
