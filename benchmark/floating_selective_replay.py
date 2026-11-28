"""Run the current V1 Floating predictor with verified frame-local stage checkpoints."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import sys
from unittest.mock import patch

import cv2
import numpy as np
import PIL
from PIL import Image

from benchmark import floating_v1_expanded as expanded, floating_v1_fields as field_model
from benchmark import floating_pregrid_depth_refit as pre, floating_v1_run as v1
from benchmark.expanded_baseline import read,save
from benchmark.floating_stage_cache import ROOT, digest, execute, identity
from benchmark.full_block_lattice import atlas
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import axis_vector
from benchmark.relative_face_scoring import evaluate_relative_voxels,yaw_from_anchor_normal
from engine import surface_fields
from engine.export import JAVA_26_3_FORMAT,write_pack
from engine.surface_model import DEFAULTS
from engine.atlas_count import CALIBRATED_INDEX

STAGES=("fields","surfaces-counts","native-charts","expanded","pregrid","assembly","score")
ARCHIVE=ROOT/"benchmark/reviews/floating-visible-builds-v1-01"
OLD_CODE={"engine/surface_fields.py":"26027251d9ffbd53c31a0d841685a221ce299ca752f1e60224e196b2eec06bd3",
          "benchmark/floating_v1_fields.py":"b22c12e5b0e6ac5d6c42a762910b2d1136bafe88e8c71732cd4a14cd1f59ea88",
          "engine/atlas.py":"ca3edca316ea037c7f7377d59afda879dfcc6772e99cb7ef77e0fb8678b8ee1e",
          "engine/geometry.py":"784cfb92ba636e3a534f09f17bfbd45f55ea8064c8ac7633727bb94ede19db80"}


def field_code():
    return {p:hashlib.sha256((ROOT/p).read_text(encoding="utf-8").encode()).hexdigest() for p in OLD_CODE}


def stage_inputs(stage,case,prior,opt):
    rgb=Path(case["variants"]["sky"]["image"])
    runtime=dict(python=list(sys.version_info[:3]),platform=platform.machine(),numpy=np.__version__,opencv=cv2.__version__,pillow=PIL.__version__)
    info=dict(schema=1,stage=stage,frame=case["id"],source=digest(rgb),runtime=runtime,
              parent=next(reversed(prior.values()))["fingerprint"] if prior else None)
    if info["source"]!=case["rgb_sha256"]:
        raise ValueError(f"Changed frozen source: {rgb}")
    if stage=="fields":
        try:
            info["onnx_runtime"]=version("onnxruntime")
        except PackageNotFoundError:
            info["onnx_runtime"]=None
        info.update(model={k:v[2] for k,v in surface_fields.MODELS.items()},preprocessing=surface_fields.SETTINGS,producer=field_code())
        # Check an explicit override even during warm replay.
        for env,_,expected in surface_fields.MODELS.values():
            if os.getenv(env) and digest(Path(os.environ[env]))!=expected:
                raise ValueError(f"Incorrect model at {env}")
    if stage in ("surfaces-counts","native-charts","expanded","pregrid","assembly"):
        info.update(atlas=atlas().provenance,index=digest(CALIBRATED_INDEX))
    if stage=="surfaces-counts":
        info.update(defaults=digest(DEFAULTS),boundary="wall-depth-anything",through="counts")
    if stage=="pregrid":
        info["minimum_wall_pixels"]=opt.min_area
    if stage=="assembly":
        info.update(edge_pixels=opt.edge_pixels,min_edges=1,coplanar_pixels=.5,pack_format=JAVA_26_3_FORMAT)
    if stage=="score":
        frame=Path(case["frame"])
        info.update(reference={k:digest(frame/k) for k in (*case["reference_sha256"],"frame.json")},
                    orientation=opt.orientation,framing=case["framing"])
        if any(info["reference"][k]!=v for k,v in case["reference_sha256"].items()):
            raise ValueError(f"Changed frozen reference: {frame}")
    return info


def fields(case,out,opt):
    name=case["id"]
    source_dir=Path(case["frame"]).parents[3]
    historical=opt.field_archive/name if opt.field_archive else None
    if historical and (historical/"receipt.json").is_file() and field_code()==OLD_CODE and opt.rerun_from!="fields":
        proof=read(historical/"receipt.json")
        # The published field fingerprint is independent of the private donor.
        original=next(r["fields"] for r in read(ARCHIVE/"conditions.json.gz") if r["id"]==name)
        v1.require_fields(case,historical,proof)
        if proof!=original or proof["preprocessing"]!=surface_fields.SETTINGS:
            raise ValueError("Unverified source-field archive")
        dest=out/name
        dest.mkdir()
        for filename in ("normal.npz","depth.npz","receipt.json"):
            shutil.copyfile(historical/filename,dest/filename)
        save(out/"provider.json",dict(method="verified archived source fields",code=OLD_CODE))
    else:
        with patch.object(field_model,"SOURCE",source_dir),patch.object(field_model,"OUT",out):
            field_model.produce(case)
        save(out/"provider.json",dict(method="new source inference",code=field_code()))


def refit(case,out,roots,opt):
    n=case["id"]
    src=roots["surfaces-counts"]
    package=roots["expanded"]/"package"
    rel=Path("sky")/n/"charts.json"
    pin=dict(source=dict(sha256=case["rgb_sha256"]),selected_package_chart_sha256=digest(package/"charts"/rel),
             surfaces_sha256=digest(src/"sky"/n/"source-handoff/surfaces.json"))
    with patch.object(pre,"fields",v1.source_fields),patch.object(pre,"OUT",out),patch.object(v1,"FIELDS",roots["fields"]):
        prepared=pre.prepare_charts("v1",case,pin,[],src,package,opt.min_area)
    evidence=dict(membership={},complete=[],partial=[])
    if prepared:
        packet,sky,audit,candidates,membership,complete,partial,_=prepared
        evidence=dict(membership=membership,complete=sorted(complete),partial=sorted(partial),parts=audit)
        for arm in ("complete","partial"):
            filtered,dropped=pre.border_filter(candidates[arm],sky)
            save(out/(arm+".json"),dict(packet,charts=filtered,sky_drops=dropped))
    save(out/"evidence.json",evidence)


def assemble_arm(packet,planes,rgb,members,edge_pixels,destination,*,save_trace=False,group_selector=None):
    solver=v1.horizontal.solver
    original_orient=solver.oriented
    original_join=solver.coplanar_joins
    oriented=[]
    def horizontal(charts):
        walls,hint=original_orient(charts)
        all_charts=walls+v1.horizontal.horizontal_charts(walls,planes,rgb.shape[1::-1],rgb)
        if save_trace or group_selector is not None:
            oriented.extend(all_charts)
        return all_charts,hint
    def coplanar(charts,tolerance):
        joins=original_join(charts,tolerance)
        return [j for j in joins if charts[j["i"]]["id"] not in members or
                members[charts[j["i"]]["id"]]!=members.get(charts[j["j"]]["id"])]
    with patch.object(solver,"oriented",horizontal),patch.object(solver,"coplanar_joins",coplanar):
        result=expanded.with_visible_export(solver.assemble)(packet["charts"],min_edges=1,edge_pixels=edge_pixels,coplanar_pixels=.5)
    groups = result["groups"]
    if group_selector is not None and groups:
        main = group_selector(groups, oriented, planes, rgb.shape[1::-1])
    else:
        main = groups[0] if groups else dict(blocks={}, origins={}, assigned=set())
    destination.mkdir()
    pack=destination/"reconstruction.zip"
    write_pack(pack,{p:b.removeprefix("minecraft:") for p,b in main["blocks"].items()},JAVA_26_3_FORMAT)
    expected={p:"minecraft:"+b.removeprefix("minecraft:") for p,b in main["blocks"].items()}
    if decoded_pack(pack)!=expected:
        raise ValueError("Java pack changed local coordinates or IDs")
    record=dict(selected=sorted(main["origins"]),assigned=sorted(main["assigned"]),joins=result["joins"],
                orientation=result["orientation_hint"],size=len(expected))
    if save_trace:
        record.update(selected_origins=main["origins"],conflicts=result["conflicts"])
        save(destination/"oriented-charts.json",dict(charts=oriented))
    save(destination/"assembly.json",record)
    return record


def assembly(case,out,roots,opt):
    name=case["id"]
    rel=Path("sky")/name/"charts.json"
    support=roots["surfaces-counts"]/"sky"/name/"source-handoff"
    rgb=np.asarray(Image.open(case["variants"]["sky"]["image"]).convert("RGB"))
    planes=read(support/"surfaces.json")["planes"]
    evidence=read(roots["pregrid"]/"evidence.json")
    members=evidence["membership"]
    options={"baseline":roots["expanded"]/"split-border/charts"/rel}
    options.update({a:roots["pregrid"]/(a+".json") for a in ("complete","partial") if (roots["pregrid"]/(a+".json")).is_file()})
    records={}
    for arm,file in options.items():
        packet=read(file)
        v1.paths.check_handoff(packet,support,case["rgb_sha256"])
        records[arm]=assemble_arm(packet,planes,rgb,members if arm!="baseline" else {},opt.edge_pixels,out/arm)
    choice="baseline"
    if "complete" in records:
        fitted=set(evidence["complete"])
        sel=set(records["complete"]["selected"])
        original=set(records["baseline"]["selected"])
        parents=lambda ids:{members.get(k,k) for k in ids}
        connected=(fitted and fitted<=sel and parents(original)<=parents(sel) and
                   all(any(cid in j["chart_ids"] for j in records["complete"]["joins"]) for cid in fitted))
        if connected and records["complete"]["size"]>=records["baseline"]["size"]:
            choice="complete"
        if pre.corner_proof(records["partial"]["selected"],records["partial"]["joins"],set(evidence["partial"]),members):
            choice="partial"
    save(out/"selected.json",dict(arm=choice,pack_sha256=digest(out/choice/"reconstruction.zip")))


def score(case,out,roots,opt):
    base=roots["assembly"]
    choice=read(base/"selected.json")["arm"]
    directory=base/choice
    pack=decoded_pack(directory/"reconstruction.zip")
    rec=read(directory/"assembly.json")
    ref=v1.score_reference(case)
    anchor=rec["orientation"]
    yaw=yaw_from_anchor_normal(anchor["normal_camera"],axis_vector(anchor["local_normal_axis"]),ref.pose) if anchor else None
    quarter=0 if not pack else (yaw or {}).get("yaw_quarters")
    result,world=evaluate_relative_voxels(pack,ref,yaw_quarters=quarter,relaxed_yaw=True,
                                        assigned={tuple(p) for p in rec["assigned"]})
    ordinary=result["fixed"] or result["relaxed_yaw"]
    scores={m:{k:dict(ordinary[m][k]) for k in ("strict","facade")} for m in ("geometry","scores")}
    if result["fixed"] is None and opt.orientation=="charge":
        for table in scores.values():
            for item in table.values():
                item.update(correct=0,missed=item["correct"]+item["missed"],extras=len(pack),f1=0.)
    save(out/"score.json",dict(id=case["id"],choice=choice,pack_sha256=digest(directory/"reconstruction.zip"),
         reference=ref.hashes,source_sha256=case["rgb_sha256"],resolved=result["fixed"] is not None,
         **scores,aligned=[[list(p),b] for p,b in sorted(world.items())]))

def produce(stage,out,case,roots,opt):
    n=case["id"]
    if stage=="fields":
        fields(case,out,opt)
    elif stage=="surfaces-counts":
        v1.build_handoff(Path(case["variants"]["sky"]["image"]),roots["fields"]/n,out/"sky"/n/"source-handoff")
    elif stage=="native-charts":
        with patch.object(v1,"OUT",out),patch.object(v1,"SOURCE",roots["surfaces-counts"]):
            v1.native_charts(case)
    elif stage=="expanded":
        with patch.object(v1,"OUT",roots["native-charts"]),patch.object(v1,"SOURCE",roots["surfaces-counts"]),patch.object(v1,"FIELDS",roots["fields"]):
            _,decisions=expanded.expanded_packet(case,out)
            save(out/"decisions.json",decisions)
    elif stage=="pregrid":
        refit(case,out,roots,opt)
    elif stage=="assembly":
        assembly(case,out,roots,opt)
    else:
        score(case,out,roots,opt)


def process(job):
    case,opt=job
    cv2.setNumThreads(1)
    stages=STAGES[:STAGES.index(opt.through)+1]
    path=opt.cache/identity(case["id"])[:12]
    notes=execute(path,stages,lambda s,r:stage_inputs(s,case,r,opt),
                  lambda s,d,r:produce(s,d,case,r,opt),rerun_from=opt.rerun_from)
    return dict(id=case["id"],stages=notes)


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache",type=Path,default=ROOT/"runs/replay")
    cli.add_argument("--case",action="append",help="Exact original V1 frame ID; repeat to select multiple.")
    cli.add_argument("--release",type=Path,help="Verified original V1 release with data/<split>/frames/<id>.")
    cli.add_argument("--field-archive",type=Path,help="Read-only, published V1 per-frame source fields.")
    cli.add_argument("--through",choices=STAGES,default="score")
    cli.add_argument("--rerun-from",choices=STAGES)
    cli.add_argument("--orientation",choices=["charge","legacy-best-yaw"],default="charge")
    cli.add_argument("--min-area",type=int,default=5000)
    cli.add_argument("--edge-pixels",type=float,default=2.5)
    cli.add_argument("--workers",type=int,default=4)
    cli.add_argument("--profile",type=Path,help="Save aggregate stage actions and frame-level elapsed seconds.")
    opt=cli.parse_args(argv)
    if opt.rerun_from and STAGES.index(opt.rerun_from)>STAGES.index(opt.through):
        cli.error("--rerun-from must precede or equal --through")
    if opt.workers<1 or opt.min_area<1 or not np.isfinite(opt.edge_pixels) or opt.edge_pixels<=0:
        cli.error("worker count, minimum area and edge tolerance must be positive")
    requested=set(opt.case or ())
    cases=[c for c in v1.tasks() if not requested or c["id"] in requested]
    if requested-{c["id"] for c in cases}:
        cli.error("Unknown or excluded original V1 ID")
    if opt.release:
        for c in cases:
            frame=opt.release/"data"/c["split"]/"frames"/c["id"]
            c["frame"]=str(frame)
            c["variants"]["sky"]["image"]=str(frame/"rgb.png")
    with ProcessPoolExecutor(opt.workers) as pool:
        profiles=list(pool.map(process,[(c,opt) for c in cases]))
    actions=Counter(r["action"] for p in profiles for r in p["stages"])
    if opt.profile:
        save(opt.profile,dict(actions=actions,frames=profiles))
    print(json.dumps(dict(frames=len(profiles),actions=actions)),flush=True)


if __name__=="__main__":
    main()
