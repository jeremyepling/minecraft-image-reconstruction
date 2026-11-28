"""Replay single signed patch-offset changes on exported Java 26.3 builds.

Run as a script with the sibling floating-depth-refit checkout and its immutable
archives in place. Source manifests contain only image-derived chart, field and
local export data. Scoring receipts and the recorded camera stay separate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = ROOT.parent / "floating-depth-refit"
REVISION = "47d6e7c1c6b8d6eba3d20b9990c7ca3f498c3dfa"
sys.path.insert(0, str(ARCHIVED))

def moge_raw():
    """Run the installed CPython 3.12 ONNX wheel in its matching interpreter."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    model,image_path,output,max_side,tokens = sys.argv[2:]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((int(max_side),)*2,Image.Resampling.LANCZOS)
    rgb = (np.asarray(image,np.float32)/255).transpose(2,0,1)[None]
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 4
    settings.inter_op_num_threads = 1
    net = ort.InferenceSession(model,sess_options=settings,providers=["CPUExecutionProvider"])
    points,mask = net.run(["points","mask"],dict(image=rgb,num_tokens=np.asarray(int(tokens),np.int64)))
    np.savez(output,points=points[0],mask=mask[0],onnxruntime=ort.__version__)


if __name__=="__main__" and sys.argv[1:2]==["--moge-raw"]:
    moge_raw()
    sys.exit()

import cv2
import numpy as np
from PIL import Image
from benchmark import floating_depth_refit as past
from benchmark.expanded_baseline import digest, read, save
from benchmark.expanded_wall_rulers import NATIVE
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import oriented
from engine.surface_fields import MODELS, SETTINGS
from engine.geometry import project, bilinear
from benchmark.split_depth import relative_depth

OUT = ROOT / "runs/local-reprojection-options"
ARCHIVE = past.REVIEW / "conditions.json.gz"
DEVELOPMENT = {"train-riverside-mill-1-warm-low-left", "train-viaduct-01-riverstone-low-left",
               "test-castle-01-ashlar-high-right"}
TRANSFER = {"arcbridge", "market-court", "sawtooth", "stable-yard", "steppedhall", "terrace", "twinspire"}


def manifest():
    archive = {(r["panel"], r["id"]): r for r in read(ARCHIVE) if r["active"]}
    jobs = []
    for panel in past.PANELS:
        for case in past.paths(panel)[0][3]:
            name = case["id"]
            row = archive.get((panel, name))
            if row is None:
                continue
            family = case.get("family") or ("twinspire" if "twinspire" in name else None)
            if panel == "sky":
                group = "six-sky"
            elif name in DEVELOPMENT and row["patches"]:
                group = "development"
            elif family in TRANSFER and row["patches"]:
                group = "held-out"
            else:
                continue
            jobs.append((group, panel, case, row))
    return sorted(jobs, key=lambda job: (["development", "six-sky", "held-out"].index(job[0]), job[2]["id"]))


def archive_root(panel, name, row):
    candidates = [past.OUT / panel / past.EXTENDED, past.OUT / panel / "source-refit",
                  past.PRIOR / panel / "source", past.FLAT / panel / "large-contraction"]
    for candidate in candidates:
        folder = candidate / "assembly" / past.LOCATION / name
        if not (folder / "score.json").is_file():
            continue
        assert digest(folder / "reconstruction.zip") == row["pack_sha256"][past.EXTENDED]
        return candidate
    raise FileNotFoundError((panel, name))


def field_paths(panel, case, source):
    group = Path(case["dataset"]) / case["id"]
    if panel == "sky":
        root = past.combined.assembly.SHARED / "baseline/sky" / group / "handoffs"
    elif panel == "original":
        archive = ROOT.parent / "full-expanded-controls/runs/expanded-baseline-01" / group / "handoffs"
        root = archive if archive.is_dir() else NATIVE.parent / "fields" / group
    else:
        root = source.parent / "fields" / group
    return {k: str(root / (k + ".npz")) for k in MODELS}


def frozen_patches(row, charts):
    patches = []
    for old in row["patches"] + row["extra_patches"]:
        fitted = old.get("source-refit", old)
        patches.append(dict(id=f"p{len(patches)}", chart=old["chart"], H=fitted["H"],
                            parent_H=old["parent_H"], old_uv=old["old_uv"], uv=fitted["uv"],
                            incumbent=old["depth"]))
    for chart in charts:
        pieces = [p for p in patches if p["chart"] == chart["id"]]
        if not pieces:
            continue
        # The producer appends each patch's resampled cells in this exact order.
        # Index ownership stays unambiguous even where two fitted UV grids overlap.
        cells = chart["export_cells"]
        cursor = len(cells) - sum(len(p["uv"]) for p in pieces)
        for piece in pieces:
            stop = cursor + len(piece["uv"])
            segment = cells[cursor:stop]
            assert [[c["u"], c["v"]] for c in segment] == piece["uv"]
            assert all(c["normal_offset"] == piece["incumbent"] for c in segment)
            piece["export_indices"] = list(range(cursor, stop))
            cursor = stop
    return patches


def gross(previous, current, truth):
    before, after = set(previous), set(current)
    right_before = {p for p in before & truth.keys() if previous[p] == truth[p]}
    right_after = {p for p in after & truth.keys() if current[p] == truth[p]}
    return dict(gained=len((after-before) & truth.keys()), lost=len((before-after) & truth.keys()),
                added=len(after-before-truth.keys()), removed=len(before-after-truth.keys()),
                id_gained=len(right_after-right_before), id_lost=len(right_before-right_after))


def run_case(job):
    group, panel, case, row = job
    cv2.setNumThreads(1)
    name = case["id"]
    directory = OUT / panel / name
    archived = archive_root(panel, name, row)
    relative = Path("sky") / name / "charts.json"
    packet_path = archived / "charts" / relative
    packet = read(packet_path)
    incumbent = archived / "assembly" / past.LOCATION / name
    receipt = read(incumbent / "score.json")
    ordinary = past.paths(panel)[0][0]
    image = Path(case["variants"]["sky"]["image"])
    assert digest(image) == row["source_sha256"] == receipt["source_sha256"] == packet["source_sha256"]
    assert digest(packet_path) == receipt["chart_sha256"]
    frame = Path(case["frame"])
    assert all(digest(frame/k) == sha for k,sha in receipt["reference_sha256"].items())
    truth = {tuple(b["xyz_world"]): b["block_id"] for b in read(frame/"labels.json")["blocks"]}
    charts = packet["charts"]
    pieces = frozen_patches(row, charts)
    source = dict(image=str(image), surfaces=str(ordinary/"sky"/name/"source-handoff/surfaces.json"),
                  chart_packet=str(packet_path), fields=field_paths(panel,case,ordinary),
                  origins=receipt["selected_origins"], oriented={c["id"]:{k:c[k] for k in ("normal_axis","axes","H")}
                  for c in oriented(charts)[0]}, patches=pieces, options={})
    offsets = {p["id"]:p["incumbent"] for p in pieces}
    options = [("incumbent",None,None)]
    for piece in pieces:
        for bias in (-1,1):
            target = piece["incumbent"] + bias
            options.append((f"{piece['id']}_to_{target}",piece,target))
    measurements = {}
    baseline = None
    for key, piece, target in options:
        option_offsets = dict(offsets)
        # The existing writer adds the full frame ID to its path. Keep the
        # replay prefix short enough for Windows temporary-file writes.
        export_key = sha256(name.encode()).hexdigest()[:12]
        run_root = OUT / "packs" / panel / export_key / key
        folder = run_root / "assembly" / past.LOCATION / name
        output_packet = packet
        if piece is not None:
            option_offsets[piece["id"]] = target
            modified = []
            for chart in charts:
                if chart["id"] == piece["chart"]:
                    changed = list(chart["export_cells"])
                    for i in piece["export_indices"]:
                        changed[i] = dict(changed[i], normal_offset=target)
                    chart = dict(chart, export_cells=changed)
                modified.append(chart)
            output_packet = dict(packet, charts=modified)
        save(run_root / "charts" / relative, output_packet)
        with patch.object(past.combined, "inputs", lambda _: past.paths(panel)[0]):
            past.combined.run((panel, case, run_root / "charts", run_root / "assembly"))
        score = read(folder/"score.json")
        assert score["reference_sha256"] == receipt["reference_sha256"] and score["joins"] == receipt["joins"]
        assert score["selected_origins"] == receipt["selected_origins"]
        assert digest(folder/"reconstruction.zip") == score["pack_sha256"]
        coords = read(folder/"placements.json")
        assert decoded_pack(folder/"reconstruction.zip") == {tuple(p):m for p,m in coords["local"]}
        local = directory/key/"local.json"
        save(local,coords["local"])
        source["options"][key] = dict(local_blocks=str(local), offsets=option_offsets,
                                      origins=score["selected_origins"])
        aligned = {tuple(p):m for p,m in coords["aligned"]}
        if baseline is None:
            baseline = aligned
        measured = score["score"]["fixed"] or score["score"]["relaxed_yaw"]
        if key == "incumbent":
            assert measured == row["score"][past.EXTENDED]
            assert coords == read(incumbent / "placements.json")
        measurements[key] = dict(score=measured, paired=gross(baseline,aligned,truth),
                                 pack_sha256=score["pack_sha256"], pack=str(folder/"reconstruction.zip"))
    save(directory/"source.json",source)
    oracle = max(measurements,key=lambda key:measurements[key]["score"]["geometry"]["strict"]["f1"])
    info = dict(group=group, panel=panel, id=name, frame=str(frame), dataset=case["dataset"],
                family=case.get("family") or "twinspire", reference_sha256=receipt["reference_sha256"],
                source_sha256=row["source_sha256"], chart_sha256=receipt["chart_sha256"],
                archive_sha256=digest(ARCHIVE), surfaces_sha256=digest(Path(source["surfaces"])),
                fields_sha256={k:digest(Path(v)) for k,v in source["fields"].items()},
                options=measurements, oracle=oracle)
    save(directory/"results.json",info)
    print(group,name,len(options),oracle,measurements[oracle]["paired"],flush=True)


def moge_field(image_path, directory):
    model = ROOT.parent / "normal-round/.deps" / MODELS["normal"][1]
    assert digest(model) == MODELS["normal"][2]
    runtime = ROOT.parent / "surface-round/runs/depth-probe/runtime"
    python = Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
    raw = directory/"moge-point-head.npz"
    subprocess.run([str(python),str(Path(__file__).resolve()),"--moge-raw",str(model),str(image_path),
                    str(raw),str(SETTINGS["normal_max_side"]),str(SETTINGS["normal_tokens"])],
                   check=True,env=dict(os.environ,PYTHONPATH=str(runtime)))
    with np.load(raw,allow_pickle=False) as arrays:
        inverse = relative_depth(arrays["points"],arrays["mask"])
        runtime_version = str(arrays["onnxruntime"])
    path = directory/"moge-point-inverse-depth.npy"
    np.save(path,inverse,allow_pickle=False)
    return inverse,dict(field_sha256=digest(path), image_sha256=digest(image_path),
                        model_sha256=MODELS["normal"][2], output=["points","mask"],
                        adapter_sha256=digest(ARCHIVED/"benchmark/split_depth.py"),
                        field=str(path), onnxruntime=runtime_version, preprocessing=SETTINGS)


def ordinal_depth(source, field):
    """Use only this depth channel on the frozen source component's measured contacts.

    Detrend in screenshot pixels using adjacent flat-parent cells. A contact needs
    agreeing signs across its native mask. Relative depth has no calibrated unit,
    so +1/+2 tie, as do -1/-2. The incumbent wins all score ties.
    """
    patches = source["patches"]
    charts = {c["id"]:c for c in read(Path(source["chart_packet"]))["charts"]}
    planes = {p["id"]:p["support"] for p in read(Path(source["surfaces"]))["planes"]}
    with Image.open(source["image"]) as image:
        size = np.array(image.size)
    evidence = {}
    for piece in patches:
        chart = charts[piece["chart"]]
        h = np.array(piece["parent_H"])
        alpha,box = past._native_alpha(planes[piece["chart"]],tuple(size))
        native = np.zeros(tuple(size[::-1]),bool)
        x,y,r,b = box
        native[y:b,x:r] = np.asarray(alpha)>0
        chart_uv = {(c["u"],c["v"]) for c in chart["cells"]}
        excluded = {tuple(p) for item in patches if item["chart"]==piece["chart"] for p in item["old_uv"]}
        parent = chart_uv - excluded
        moving = {tuple(p) for p in piece["old_uv"]}
        def depth_at(uv):
            xy = project(h,np.asarray(uv,float)+.5)
            values = bilinear(field[...,None],xy*field.shape[1::-1]/size)[...,0]
            valid = past.sample(native,xy,1) & np.isfinite(values) & (values>0)
            return xy, values, valid
        # A planar surface's inverse camera depth is affine in pixel coordinates.
        links = [(p,(p[0]+dx,p[1]+dy)) for p in sorted(parent) for dx,dy in ((1,0),(0,1))
                 if (p[0]+dx,p[1]+dy) in parent]
        if len(links)<6:
            continue
        coords,depth,valid = depth_at(links)
        dx = np.diff(coords,axis=1)[:,0]
        dd = np.diff(depth,axis=1)[:,0]
        keep = valid.all(1)
        for _ in range(3):
            if keep.sum()<6:
                break
            slope,_,rank,_ = np.linalg.lstsq(dx[keep],dd[keep],rcond=None)
            if rank < 2:
                break
            residual = abs(dd-dx@slope)
            flat = residual[valid.all(1)]
            keep = valid.all(1) & (residual<=np.quantile(flat,.75))
        else:
            contacts = [(p,(p[0]+a,p[1]+b)) for p in sorted(moving)
                        for a,b in ((1,0),(-1,0),(0,1),(0,-1)) if (p[0]+a,p[1]+b) in parent]
            if not contacts:
                continue
            coordinates,levels,ok = depth_at(contacts)
            signed = ((levels[:,0]-levels[:,1]) - (coordinates[:,0]-coordinates[:,1])@slope)[ok.all(1)]
            if len(signed) < 2:
                continue
            threshold = max(4*np.median(flat),.018*np.median(depth[valid]))
            votes = np.where(abs(signed)>threshold,np.sign(signed),0)
            counts = {sign:int(np.count_nonzero(votes==sign)) for sign in (-1,0,1)}
            winner = max(counts,key=counts.get)
            if counts[winner]/len(votes) >= 2/3:
                evidence[piece["id"]] = dict(sign=winner,contacts=len(votes),votes=counts,
                                                confidence=counts[winner]/len(votes),threshold=float(threshold))
    scores = {}
    for name, option in source["options"].items():
        scores[name] = sum(vote["confidence"] for pid,vote in evidence.items()
                           if int(np.sign(option["offsets"][pid])) == vote["sign"])
    return dict(choice=max(scores,key=scores.get),scores=scores,evidence=evidence)


def depth_case(job):
    group,panel,case,_ = job
    cv2.setNumThreads(1)
    directory = OUT/panel/case["id"]
    source = read(directory/"source.json")
    results = read(directory/"results.json")
    if source["patches"]:
        field = Path(source["fields"]["depth"])
        with np.load(field,allow_pickle=False) as arrays:
            da = ordinal_depth(source,arrays["value"])
        da["provenance"] = dict(model_sha256=MODELS["depth"][2], field_sha256=digest(field),
                                image_sha256=results["source_sha256"],output="predicted_depth")
        inverse,receipt = moge_field(Path(source["image"]),directory)
        moge = dict(ordinal_depth(source,inverse),provenance=receipt)
    else:
        da = moge = dict(choice="incumbent",reason="no eligible patch",scores=dict(incumbent=0))
    save(directory/"depth-only.json",dict(depth_anything=da,moge=moge))
    print(group,case["id"],da["choice"],moge["choice"],flush=True)


def projection_case(job):
    group,panel,case,_ = job
    directory = OUT/panel/case["id"]
    source = read(directory/"source.json")
    results = read(directory/"results.json")
    depth = read(directory/"depth-only.json")
    da,moge = depth["depth_anything"],depth["moge"]
    # Load the experiment's projection code beside this script, while retaining
    # the archived assembly and scoring packages on sys.path.
    import importlib.util
    spec = importlib.util.spec_from_file_location("local_projection",Path(__file__).with_name(
        "floating_local_reprojection_projection.py"))
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    scores = {key:(da["scores"][key]+moge["scores"][key])/2 for key in source["options"]}
    model_only = dict(choice=max(scores, key=scores.get), scores=scores)
    reprojection = projection.rank_options(source)
    blend = projection.rank_options(source,depth_scores=scores)
    registration = results["options"]["incumbent"]["score"]
    known = projection.rank_reference_camera(source,Path(case["frame"]),registration["yaw_quarters"],
                                             registration["translation"])
    decisions = dict(depth_anything=da, moge=moge, model_only=model_only, reprojection=reprojection,
                     combined=blend, reference_camera=known)
    save(directory/"depth.json",decisions)
    print(group,case["id"],{k:v["choice"] for k,v in decisions.items()},flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("options", "depth", "projection"), default="options")
    parser.add_argument("--group", choices=("development", "six-sky", "held-out"))
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--cases",nargs="+")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git","-C",str(ARCHIVED),"rev-parse","HEAD"],text=True).strip()
    assert revision == REVISION
    jobs = manifest()
    save(OUT/"cohort.json", dict(revision=revision, archive_sha256=digest(ARCHIVE),
         cases=[dict(group=g,panel=p,id=c["id"],patches=len(r["patches"])+len(r["extra_patches"]))
                for g,p,c,r in jobs]))
    if args.cases:
        jobs = [j for j in jobs if j[2]["id"] in args.cases]
    if args.group:
        jobs = [j for j in jobs if j[0] == args.group]
    if args.resume:
        filename = {"options":"results.json","depth":"depth-only.json","projection":"depth.json"}[args.stage]
        jobs = [j for j in jobs if not (OUT/j[1]/j[2]["id"]/filename).exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map({"options":run_case,"depth":depth_case,"projection":projection_case}[args.stage],jobs))


if __name__ == "__main__":
    main()
