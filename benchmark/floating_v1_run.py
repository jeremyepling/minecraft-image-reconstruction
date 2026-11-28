"""Replay the Floating source handoff and sparse connected atlas on original Visible Builds v1."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_cloud_replay as paths, garden_anchor_extension as garden
from benchmark import native_corner_registration as registration, native_surface_counter as counter
from benchmark import relative_face_round as assembly, relative_source_charts, shared_corner_rows as shared
from benchmark import expanded_scores, floating_patch_combined as horizontal
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.floating_build_local_run import build_handoff

OUT=ROOT/"runs/floating-v1"
SOURCE=OUT/"source"
FIELDS=ROOT/"runs/v1-fields"
FIXTURES=WORKTREES/"floating-extra-review/runs/floating-v1-audit/fixtures.json"
PREFIX=Path("connected-atlas/sky")


def tasks():
    import gzip, hashlib, json
    archived=ROOT/"benchmark/reviews/floating-visible-builds-v1-01/fixtures.json.gz"
    payload=FIXTURES.read_bytes() if FIXTURES.is_file() else gzip.decompress(archived.read_bytes())
    assert hashlib.sha256(payload).hexdigest()=="4c43236b168e47d139b6b959ab3e35184fbb5ee03e8edf21e7f51844f45861b1"
    cases=json.loads(payload)["cases"]
    assert len(cases)==144 and all("orchid" not in c["id"].lower() for c in cases)
    return cases


def handoff(name):
    return SOURCE/"sky"/name/"source-handoff"


def dense(case,folder,image,provenance):
    n=case["id"]
    destination=OUT/"dense/sky"/n
    relative_source_charts.extract((case["variants"]["sky"],n,"sky",folder,destination))
    path=destination/"dense-charts.json"
    return read(path) | provenance | dict(baseline_chart_sha256=digest(path),dense_chart_reuse=None)


def predict(case):
    n=case["id"]
    dest=OUT/"seeds/connected/sky"/n/"charts.json"
    if dest.exists():
        paths.check_handoff(read(dest),handoff(n),case["rgb_sha256"])
        return
    src=Path(case["variants"]["sky"]["image"])
    cache=FIELDS/n
    from time import sleep
    while not (cache/"receipt.json").is_file():
        sleep(2)
    assert digest(src)==case["rgb_sha256"]
    proof=read(cache/"receipt.json")
    require_fields(case,cache,proof)
    build_handoff(src,cache,handoff(n))
    with ExitStack() as env:
        env.enter_context(patch.object(paths,"RUN",SOURCE))
        env.enter_context(patch.object(shared,"OUTPUT",OUT/"shared"))
        env.enter_context(patch.object(shared,"dense_packet",dense))
        shared.predict(case)
        for mod in (counter,registration):
            env.enter_context(patch.object(mod,"BASE",OUT/"shared/sky"))
            env.enter_context(patch.object(mod,"OUTPUT",OUT/"native"))
        counter.predict(case)
        registration.predict(case)
        env.enter_context(patch.object(garden,"OUTPUT",OUT/"seeds"))
        env.enter_context(patch.object(garden,"NATIVE_COUNTER",OUT/"native"))
        garden.all_walls((case,read(garden.baseline_file(n))))


def require_fields(case,cache,proof):
    from engine.surface_fields import MODELS
    assert proof["source_sha256"]==case["rgb_sha256"]
    assert proof["model_sha256"]=={k:MODELS[k][2] for k in ("normal","depth")}
    assert all(digest(cache/(k+".npz"))==proof["fields_sha256"][k] for k in ("normal","depth"))


def source_fields(group,case,src,size):
    cache=FIELDS/case["id"]
    require_fields(case,cache,read(cache/"receipt.json"))
    from benchmark import floating_depth_before_atlas as depth
    values={}
    for kind in ("normal","depth"):
        with np.load(cache/(kind+".npz")) as field:
            values[kind]=cv2.resize(field["value"].astype("f"),size)
    return depth._unit(values["normal"]),values["depth"],{k:digest(cache/(k+".npz")) for k in values}


def score_reference(case):
    # V1 stores the clearance check outside frame.json. The independently rebuilt
    # visible-only reference is read by the evaluator only after pack serialization.
    frame=Path(case["frame"])
    original=expanded_scores.read
    def verified(path):
        val=original(path)
        if path==frame/"frame.json":
            assert case["framing"]["accepted"]
            val=dict(val,framing=case["framing"])
        return val
    with patch.object(expanded_scores,"read",verified):
        return expanded_scores.load_reference(case)


def sparse(case):
    n=case["id"]
    destination=OUT/"sparse/assembly"
    mode=(OUT/"seeds/connected","charts.json",False,1,2.5,.5,False)
    with patch.object(paths,"RUN",SOURCE),patch.object(assembly,"ASSEMBLY",destination),patch.dict(
        assembly.MODES,{"connected-atlas":mode}),patch.object(assembly,"load_reference",score_reference):
        assembly.evaluate((case,"sky","connected-atlas"))
    return destination


def run(job):
    stage,case,await_sparse=job
    if await_sparse:
        from time import sleep
        while not (OUT/"sparse/assembly"/PREFIX/case["id"]/"score.json").exists():
            sleep(2)
    cv2.setNumThreads(1)
    if stage=="source":
        predict(case)
    elif stage=="sparse":
        predict(case)
        sparse(case)
    else:
        from benchmark.floating_v1_expanded import run as expand
        predict(case)
        expand(case)
    return case["id"]


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["source","sparse","current"],default="current")
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--filter",default="")
    p.add_argument("--ready",action="store_true")
    p.add_argument("--await-sparse",action="store_true")
    p.add_argument("--limit",type=int)
    args=p.parse_args()
    cases=[c for c in tasks() if args.filter in c["id"] and
           (not args.ready or (FIELDS/c["id"]/"receipt.json").is_file())]
    cases=[c for c in cases if not (OUT/"rows"/(c["id"]+".json.gz")).is_file()][:args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(run,(args.stage,c,args.await_sparse)) for c in cases]
        for task in as_completed(futures):
            print("finished",task.result(),flush=True)
