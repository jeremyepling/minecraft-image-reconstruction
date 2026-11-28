"""Separate the effects of export size and ruler geometry using decoded floating exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random
from unittest.mock import patch

import numpy as np

from benchmark import relative_face_round as assembly
from benchmark.floating_one_face_control import PANELS, ROOT, WORKTREES, base, combined, inputs
from benchmark.floating_bridges_report import summarize
from benchmark.expanded_baseline import read, save
from benchmark.relative_face_assembly import axis_vector
from benchmark.mask_blocks import decoded_pack

OUT=ROOT/"runs/extra-budget"
REVIEW=ROOT/"benchmark/reviews/floating-extra-review-01"
LOCATION=Path("connected-atlas/sky")


def paths(group):
    if group=="holdout5":
        work=WORKTREES/"floating-extra-audit/runs"
        root=work/"floating-holdout5"
        return (root/"ordinary",root/"ordinary/atlas/connected",root/"source_material/charts",read(root/"manifest.json")["cases"]),root/"horizontal-cells/assembly",work/"face-proposals/holdout5/assembly"
    key={"holdout3":"new6","holdout4":"new7"}.get(group,group)
    return inputs(group),base(group),WORKTREES/"floating-face-proposals/runs/face-proposals"/key/"assembly"


def measured_priorities(charts,group,seeds):
    priority={}
    for chart in sorted(charts,key=lambda c:-c.get("confidence",0.)):
        origin=group["origins"].get(chart["id"])
        if origin is None:
            continue
        u,v=[axis_vector(a) for a in chart["axes"]]
        normal=axis_vector(chart["normal_axis"])
        start=np.asarray(origin)+np.minimum(0,u)+np.minimum(0,v)-np.maximum(0,normal)
        seed_uv=seeds.get(chart["id"],set())
        for cell in chart["cells"]:
            uv=(cell["u"],cell["v"])
            xyz=tuple(map(int,start+uv[0]*u+uv[1]*v))
            if xyz not in priority:
                assert group["blocks"][xyz]==cell["block_id"]
                priority[xyz]=(uv in seed_uv,not cell.get("fallback"),cell.get("confidence",0.))
    assert priority.keys()==group["blocks"].keys()
    return priority


def run(job):
    panel,case=job
    (src,original,charts,_),base_root,candidate_root=paths(panel)
    name=case["id"]
    original_charts=read(original/"sky"/name/"charts.json")["charts"]
    seeds={c["id"]:{(p["u"],p["v"]) for p in c["cells"]} for c in original_charts}
    previous=decoded_pack(base_root/LOCATION/name/"reconstruction.zip")
    new=decoded_pack(candidate_root/LOCATION/name/"reconstruction.zip")
    limit=min(len(previous),len(new))
    full_assemble=assembly.assemble
    for strategy in ("uniform","strongest"):
        def reduced(raw,**settings):
            original_orient=combined.solver.oriented
            seen=[]
            def tracked(rows):
                oriented,hint=original_orient(rows)
                seen.extend(oriented)
                return oriented,hint
            with patch.object(combined.solver,"oriented",tracked):
                result=full_assemble(raw,**settings)
            if not result["groups"]:
                assert not previous
                return result
            main=result["groups"][0]
            assert {xyz:"minecraft:"+m.removeprefix("minecraft:") for xyz,m in main["blocks"].items()}==previous
            if len(previous)<=limit:
                return result
            ordered=sorted(previous)
            Random("extra-budget/"+name).shuffle(ordered)
            if strategy=="strongest":
                ranks=measured_priorities(seen,main,seeds)
                ordered.sort(key=ranks.__getitem__,reverse=True)
            kept=set(ordered[:limit])
            main["blocks"]={p:m for p,m in main["blocks"].items() if p in kept}
            main["assigned"] &= kept
            return result
        target=OUT/panel/strategy/"assembly"
        with patch.object(combined,"inputs",lambda _:paths(panel)[0]),patch.object(assembly,"assemble",reduced):
            combined.run((panel,case,charts,target))
        assert len(decoded_pack(target/LOCATION/name/"reconstruction.zip"))==limit


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("panels",nargs="*",default=(*PANELS,"holdout3","holdout4"))
    panels=cli.parse_args().panels
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(run,[(g,c) for g in panels for c in paths(g)[0][3]]))
    report={}
    for g in panels:
        (_,_,_,cases),prior,new=paths(g)
        dest={s:OUT/g/s/"assembly" for s in ("uniform","strongest")}
        roots={"whole-wall":prior,**dest,"new-ruler":new}
        results=summarize(cases,roots,str(LOCATION))
        shrinking=[c for c in cases if len(decoded_pack(prior/LOCATION/c["id"]/"reconstruction.zip")) >
                   len(decoded_pack(new/LOCATION/c["id"]/"reconstruction.zip"))]
        if shrinking:
            results["fewer"]=summarize(shrinking,roots,str(LOCATION))["cohorts"]["all"]
        report[g]=results
        print(g,len(shrinking),{k:v["geometry_strict"] for k,v in results.get("fewer",{}).items()},flush=True)
    save(REVIEW/("size-controls-"+"-".join(panels)+".json.gz"),report)
