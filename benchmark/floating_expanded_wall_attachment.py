"""Retest observed detached-wall attachments on the adopted expanded split-border export."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.floating_attachment_evidence import candidate,texture_depth,uv_cells
from benchmark.floating_bridges_report import read_arm
from benchmark.floating_one_face_scores import best
from benchmark.floating_package_rechecks import PACKAGE, PREFIX, ROOT, WORKTREES, combined,data,digest,jobs,read,save
from benchmark.relative_face_assembly import axis_vector,oriented
from benchmark.mask_blocks import decoded_pack

RECHECK=WORKTREES/"floating-package-rechecks"
ARCHIVE=RECHECK/"benchmark/reviews/floating-package-rechecks-01/conditions.json.gz"
PREVIOUS=WORKTREES/"floating-column-recovery/benchmark/reviews/floating-column-recovery-01/conditions.json.gz"
OUT=ROOT/"runs/expanded-wall-attachment"
REVIEW=ROOT/"benchmark/reviews/floating-expanded-wall-attachment-01"
ARMS=("tangent","texture","majority")


def baseline(g,n,archived):
    """Resolve aliases exactly as the source split-border replay did, by its recorded pack digest."""
    expected=archived["audit"]["pack_sha256"]["split-border"]
    root=RECHECK/"runs/package-rechecks"/g
    for folder in (root/"split-border",root/"border",root/"split",PACKAGE/g):
        path=folder/"assembly"/PREFIX/n
        if not (path/"score.json").exists():
            continue
        receipt=read(path/"score.json")
        if receipt["pack_sha256"]==expected:
            chart=folder/"charts/sky"/n/"charts.json"
            assert digest(chart)==receipt["chart_sha256"]
            assert digest(path/"reconstruction.zip")==expected
            return folder,receipt
    raise FileNotFoundError((g,n,expected))


def permitted(chart):
    cells=chart["cells"]
    assert "export_cells" not in chart
    if "export_uv" in chart:
        allowed=set(map(tuple,chart["export_uv"]))
        cells=[c for c in cells if (c["u"],c["v"]) in allowed]
        assert len(cells)==len(allowed)
    return dict(chart,cells=cells)


def options(packet,selected,rgb):
    walls,_=oriented(packet["charts"])
    # All chart geometry remains intact for the orientation/join solve. The attachment
    # evidence and newly exported faces see exactly the existing export membership.
    kept={c["id"]:permitted(c) for c in walls}
    inserts={arm:[] for arm in ARMS}
    audit=[]
    for original in walls:
        cid=original["id"]
        child=kept[cid]
        if cid in selected or len(child["cells"])<4:
            continue
        fits=[]
        for parent in walls:
            pid=parent["id"]
            if pid not in selected or len(kept[pid]["cells"])<4:
                continue
            fit=candidate(kept[pid],child)
            if fit:
                fits.append((fit["p90"],-fit["contacts"],pid,fit))
        if not fits:
            continue
        _,_,pid,fit=min(fits)
        parent=kept[pid]
        shift=np.asarray(fit["shift"])
        accepted,steps,hits=texture_depth(parent,child,shift,rgb)
        majority=accepted and len(hits)*2>=len(child["cells"]) and fit["contacts"]>=3
        u,v=map(axis_vector,parent["axes"])
        normal=axis_vector(parent["normal_axis"])
        start=np.asarray(selected[pid])+np.minimum(0,u)+np.minimum(0,v)-np.maximum(0,normal)
        raw_parent=next(c for c in walls if c["id"]==pid)
        rejected=set(map(tuple,uv_cells(raw_parent)))-set(map(tuple,uv_cells(parent)))
        skips=0
        for cell,(a,b),step in zip(child["cells"],uv_cells(child)+shift,steps):
            if (a,b) in rejected:
                skips+=1
                continue
            for arm,active,dz in (("tangent",True,0),("texture",accepted,step),("majority",majority,step)):
                if active:
                    pos=tuple((start+a*u+b*v+dz*normal).tolist())
                    inserts[arm].append((pos,cell["block_id"],not cell.get("fallback"),pid,cid))
        audit.append(dict(parent=pid,child=cid,fit=fit,accepted=accepted,majority=majority,steps=steps,
                          kept=len(child["cells"]),blocked_child=len(original["cells"])-len(child["cells"]),
                          blocked_parent_uv=skips,hits=hits))
    return inserts,audit


def replay(g,c,folder,arm,dest,additions,original_origins):
    assembler=combined.assembly.assemble
    def with_attachments(charts,**kwargs):
        result=assembler(charts,**kwargs)
        if result["groups"]:
            selected=result["groups"][0]
            assert selected["origins"]==original_origins
            for pos,material,assigned,parent,child in additions:
                # Existing selected blocks always retain their ownership and material.
                if pos not in selected["blocks"]:
                    selected["blocks"][pos]=material
                    if assigned:
                        selected["assigned"].add(pos)
        else:
            assert not additions and not original_origins
        return result
    with patch.object(combined,"inputs",data),patch.object(combined.assembly,"assemble",with_attachments):
        combined.run((g,c,folder/"charts",dest))


def compare(before,after,truth):
    old,new=before.keys(),after.keys()
    correct=truth.keys()
    old_exact={p for p in old & correct if before[p]==truth[p]}
    new_exact={p for p in new & correct if after[p]==truth[p]}
    return dict(gained=len((new-old)&correct),lost=len((old-new)&correct),added=len((new-old)-correct),
                removed=len((old-new)-correct),exact_gained=len(new_exact-old_exact),exact_lost=len(old_exact-new_exact))


def run(job):
    g,c,row,prior=job
    n=c["id"]
    folder,rec=baseline(g,n,row)
    chart=folder/"charts/sky"/n/"charts.json"
    packet=read(chart)
    rgb_path=Path(c["variants"]["sky"]["image"])
    assert digest(rgb_path)==rec["source_sha256"]==row["audit"]["source_sha256"]
    rgb=np.asarray(Image.open(rgb_path).convert("RGB"))
    selected=rec["selected_origins"]
    inserts,audit=options(packet,selected,rgb)
    roots={"adopted":folder/"assembly"}
    prefix=PREFIX/n
    for a,adds in [("noop",[])]+list(inserts.items()):
        if a=="noop" or adds:
            dest=OUT/g/a/"assembly"
            replay(g,c,folder,a,dest,adds,selected)
            roots[a]=dest
        else:
            roots[a]=roots["noop"]
    frozen=read(roots["adopted"]/prefix/"placements.json")
    assert read(roots["noop"]/prefix/"placements.json")==frozen
    labels=read(Path(c["frame"])/"labels.json")
    states={b["id"]:b["block_id"] for b in labels["blocks"]}
    truth={tuple(f["xyz_world"]):states[f["block"]] for f in labels["faces"]}
    scores={}
    paired={}
    packs={}
    local=decoded_pack(roots["adopted"]/prefix/"reconstruction.zip")
    before_score,before_world=read_arm(roots["adopted"]/prefix)
    assert before_score==row["score"]["split-border"]
    for a,loc in roots.items():
        output=loc/prefix
        score,world=read_arm(output)
        new_receipt=read(output/"score.json")
        assert new_receipt["reference_sha256"]==rec["reference_sha256"]
        assert selected==new_receipt["selected_origins"] and new_receipt["joins"]==rec["joins"]
        new_local=decoded_pack(output/"reconstruction.zip")
        assert all(new_local.get(p)==b for p,b in local.items())
        if a=="noop":
            assert new_local==local and score==before_score
        scores[a]=score
        paired[a]=compare(before_world,world,truth)
        packs[a]=dict(sha256=digest(output/"reconstruction.zip"),root=str(loc),
                      local_added=len(new_local.keys()-local.keys()))
    save(OUT/g/"rows"/(n+".json.gz"),dict(id=n,panel=g,dataset=c["dataset"],split=c["split"],score=scores,paired=paired,
        packs=packs,attachments=audit,old_attachments=prior["attachments"],old_paired=prior["paired"],
        baseline_root=str(folder),chart_sha256=digest(chart),reference_sha256=rec["reference_sha256"],source_sha256=digest(rgb_path),
        selected_origins=selected,normal_offsets=dict(Counter(p.get("normal_offset",0) for ch in packet["charts"] for p in ch["cells"])),
        export_uv_charts=[ch["id"] for ch in packet["charts"] if "export_uv" in ch]))
    if audit:
        print(g,n,[(a,paired[a]) for a in ARMS],flush=True)


def workload():
    rows={(r["panel"],r["id"]):r for r in read(ARCHIVE)}
    old={(r["panel"],r["id"]):r for r in read(PREVIOUS) if r["active"]}
    cases=jobs()
    assert set(rows)==set(old)=={(g,c["id"]) for g,c in cases} and len(cases)==84
    return [(g,c,rows[g,c["id"]],old[g,c["id"]]) for g,c in cases]


def report(items):
    rows=[read(OUT/g/"rows"/(c["id"]+".json.gz")) for g,c,*_ in items]
    save(REVIEW/"conditions.json.gz",rows)
    summary={}
    for cohort in ("all","sky",*sorted({r["dataset"] for r in rows})):
        subset=[r for r in rows if cohort in ("all",r["panel"],r["dataset"])]
        perfect=best([r["score"] for r in subset],["noop",*ARMS])
        results={}
        for a in ("adopted","noop",*ARMS,"whole-export-chooser"):
            selected=[a]*len(subset) if a!="whole-export-chooser" else perfect
            stats={}
            for f in ("geometry","scores"):
                for k in ("strict","facade"):
                    total=Counter()
                    for r,choice in zip(subset,selected):
                        count=r["score"][choice][f][k]
                        total.update({m:count[m] for m in ("correct","missed","extras")})
                    total["f1"]=2*total["correct"]/(2*total["correct"]+total["missed"]+total["extras"])
                    stats[f+"_"+k]=dict(total)
            stats["paired"]={k:sum(r["paired"][choice][k] for r,choice in zip(subset,selected))
                             for k in ("gained","lost","added","removed","exact_gained","exact_lost")}
            results[a]=stats
        summary[cohort]=results
    save(REVIEW/"summary.json",summary)
    print(summary["all"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workers",type=int,default=5)
    cli.add_argument("--report",action="store_true")
    args=cli.parse_args()
    items=workload()
    if args.report:
        report(items)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run,items))
