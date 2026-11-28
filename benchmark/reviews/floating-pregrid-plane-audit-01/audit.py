"""Trace frozen grid samples and decoded placements to visible native planes."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import argparse, json, sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
# The historical source-handoff and adjacent horizontal-chart helpers are in the
# expanded review checkout. This script reads them without changing their runs.
sys.path.insert(0,str(ROOT.parent/"floating-extra-review"))
from benchmark.expanded_baseline import WORKTREES, read, save, digest
from benchmark.floating_extra_budget import paths
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.relative_face_assembly import oriented, axis_vector
from benchmark.floating_horizontal_bridges import horizontal_charts
from benchmark.relative_face_scoring import _yaw
from benchmark.mask_blocks import decoded_pack
from benchmark.typed_structures import geometry_score, score as typed_score
from benchmark.camera_labels import screen_matrix
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

OUT=ROOT/"runs/pregrid-plane-audit"
PIN=ROOT/"benchmark/reviews/floating-expanded-benchmark-01"
STABLE=WORKTREES.parent/"minecraft-image-reconstruction"
PARTS=WORKTREES/"floating-package-rechecks/benchmark/reviews/floating-depth-before-atlas-01"
PART_RECEIPTS={(r["panel"],r["id"]):r["audit"] for r in read(PARTS/"conditions.json.gz")}
DIRECTIONS={(0,0,1):"south",(0,0,-1):"north",(1,0,0):"east",(-1,0,0):"west",(0,1,0):"up",(0,-1,0):"down"}
MIXED_MIN_PIXELS=64
MIXED_MIN_SHARE=.10

def reference(case):
    frame=Path(case["frame"])
    labels=read(frame/"labels.json")
    with np.load(frame/"labels.npz",allow_pickle=False) as a:
        optional=horizontal_slivers(labels,a["face_id"])
    truth={tuple(b["xyz_world"]):b["block_id"] for b in labels["blocks"]}
    hashes={f:digest(frame/f) for f in ("camera.json","labels.json","labels.npz")}
    return labels,optional,truth,hashes


def totals(scores):
    results={}
    for field in ("geometry","scores"):
        for mode in ("strict","facade"):
            counts={name:sum(s[field][mode][name] for s in scores) for name in ("correct","missed","extras")}
            correct,missed,extra=(counts[name] for name in ("correct","missed","extras"))
            results[field+"_"+mode]=dict(**counts,f1=2*correct/(2*correct+missed+extra))
    return results


def archive_item(zip,row,key):
    f=row["files"][key]
    data=zip.read(f["entry"])
    assert len(data)==f["bytes"] and sha256(data).hexdigest()==f["sha256"]
    return data


def surface_mask(shape, size):
    alpha,(x,y,r,b)=_native_alpha(shape["support"],size)
    result=np.zeros(size[::-1],bool)
    result[y:b,x:r]=np.asarray(alpha)>0
    return result


def plane_rows(counter):
    return [dict(direction=k[0],signed_depth=k[1],pixels=n) for k,n in counter.most_common()]


def mask_measure(mask,owners,keys,materials,pred_direction):
    ids, counts=np.unique(owners[mask],return_counts=True)
    phys=Counter()
    ids_by_phys=defaultdict(Counter)
    background=0
    for i,n in zip(ids,counts):
        i,n=int(i),int(n)
        if i==0:
            background+=n
        else:
            phys[keys[i]]+=n
            ids_by_phys[keys[i]][materials[i]]+=n
    visible=sum(phys.values())
    same=Counter({k:v for k,v in phys.items() if k[0]==pred_direction})
    main=next(iter(same.most_common()),(None,0))[0]
    denominator=visible if visible else 1
    parallel=sum(same.values())-(same[main] if main else 0)
    other=visible-sum(same.values())
    same_plane_materials=ids_by_phys[main] if main else Counter()
    secondary_planes={k:count for k,count in same.items() if k!=main and count>=MIXED_MIN_PIXELS and count/denominator>=MIXED_MIN_SHARE}
    different_directions=Counter()
    for k,count in phys.items():
        if k[0]!=pred_direction:
            different_directions[k[0]]+=count
    return dict(total=int(mask.sum()),visible_pixels=visible,background=background,dominant_same_direction=main,
                dominant_any_plane=phys.most_common(1)[0][0] if phys else None,
                same_direction_other_plane_pixels=parallel,different_direction_pixels=other,
                parallel_mix=bool(secondary_planes),
                orientation_mix=any(n>=MIXED_MIN_PIXELS and n/denominator>=MIXED_MIN_SHARE for n in different_directions.values()),
                major_planes=plane_rows(phys),same_plane_material_pixels=dict(same_plane_materials),
                same_plane_material_mix=sum(v>=MIXED_MIN_PIXELS and v/denominator>=MIXED_MIN_SHARE for v in same_plane_materials.values())>=2)


def describe(row,views):
    p,n=row["panel"],row["id"]
    case=views[p,n]
    with ZipFile(PIN/"exports.zip") as bundle:
        chart_packet=json.loads(archive_item(bundle,row,"charts"))
        score=json.loads(archive_item(bundle,row,"score"))
        placement=json.loads(archive_item(bundle,row,"placements"))
        local=decoded_pack(BytesIO(archive_item(bundle,row,"pack")))
    assert local=={tuple(q):m for q,m in placement["local"]}
    assert score["chart_sha256"]==row["files"]["charts"]["sha256"]
    assert score["pack_sha256"]==row["files"]["pack"]["sha256"]
    ref=reference(case)
    labels,optional,truth,ref_hash=ref
    assert ref_hash=={k:v["sha256"] for k,v in row["reference"].items()}
    image_path=STABLE/row["source"]["path"]
    assert image_path==Path(case["variants"]["sky"]["image"]) and digest(image_path)==row["source"]["sha256"]==score["source_sha256"]
    camera=read(Path(case["frame"])/"camera.json")
    with np.load(Path(case["frame"])/"labels.npz",allow_pickle=False) as a:
        owners=a["face_id"]
    camera_projection=screen_matrix(camera)
    xyz_eye=np.array(camera["eye_world"])
    ids=[]; keys={}; materials={}; site_for={}; max_px=0.

    for f in labels["faces"]:
        i=f["id"]
        normal=np.asarray(f["normal_world"])
        corners=np.asarray(f["corners_world"])
        distances=corners@normal
        assert np.ptp(distances)<1.e-5
        direction=f["direction"]
        keys[i]=(direction,round(float(distances[0]),5))
        site_for[i]=tuple(f["xyz_world"])
        materials[i]=truth[site_for[i]]
        xyz_project=(corners-xyz_eye)@camera_projection[:,:3].T+camera_projection[:,3]
        err=np.max(np.abs(xyz_project[:,:2]/xyz_project[:,2,None]-np.array(f["corners_px"])))
        max_px=max(max_px,float(err))

        ids.append((i,int(f["visible_pixels"])))
    assert max_px<=1.e-4
    observed=np.bincount(owners.ravel())
    assert all(observed[i]==n for i,n in ids)
    original=score["score"]["fixed"] or score["score"]["relaxed_yaw"]
    q=original["yaw_quarters"]
    shift=tuple(original["translation"])
    rotated={_yaw(p,q):m for p,m in local.items()}
    best,correct=best_translation_fast(rotated.keys(),truth)
    assert best==shift and correct==original["geometry"]["strict"]["correct"]
    def align(point):
        return tuple(a+b for a,b in zip(_yaw(point,q),shift))
    predicted={align(x):m for x,m in local.items()}
    assert predicted=={tuple(x):m for x,m in placement["aligned"]}
    assert original["geometry"]==geometry_score(predicted,truth,optional)
    for k,v in typed_score(predicted,truth,optional,set()).items():
        if k in ("strict","facade","material"):
            assert original["scores"][k]==v
    src=paths(p)[0][0]/"sky"/n/"source-handoff/surfaces.json"
    assert digest(src)==row["surfaces_sha256"]==score["surfaces_sha256"]
    planes=read(src)["planes"]
    plane_by={r["id"]:r for r in planes}
    with Image.open(image_path) as im:
        rgb=np.asarray(im.convert("RGB"))
    size=tuple(owners.shape[::-1])
    walls,_=oriented(chart_packet["charts"])
    need=set(score["selected_origins"])
    if need-{c["id"] for c in walls}:
        walls+=horizontal_charts(walls,planes,size,rgb)
    charts={c["id"]:c for c in walls}
    assert not (need-charts.keys())

    exported_owners={}
    chart_rows=[]
    positions_all=[]
    def source_cell(center):
        if not np.isfinite(center).all():
            return 0
        x,y=np.floor(center).astype(int)
        return int(owners[y,x]) if 0<=x<size[0] and 0<=y<size[1] else 0

    for cid in sorted(score["selected_origins"],key=lambda i:-charts[i].get("confidence",0)):
        c=charts[cid]
        origin=score["selected_origins"][cid]
        axis_u, axis_v=[axis_vector(a) for a in c["axes"]]
        normal=axis_vector(c["normal_axis"])
        world_n=_yaw(normal,q)
        direction=DIRECTIONS[tuple(world_n)]
        positive_face=np.maximum(0,world_n)
        begin=np.asarray(origin)+np.minimum(0,axis_u)+np.minimum(0,axis_v)-np.maximum(0,normal)
        base_id=cid.removeprefix("horizontal-").split("~p")[0]
        raw_mask=surface_mask(plane_by[base_id],size)
        active_mask=raw_mask
        region=None
        part_hash=None
        if "~p" in cid:
            region=int(cid.split("~p")[-1])
            native_source=PARTS/(n+"-"+base_id+"-ids.png")
            part_receipt=PART_RECEIPTS[p,n][base_id]["source"]
            part_hash=digest(native_source)
            assert part_hash==part_receipt["ids_sha256"]
            assert part_receipt["source_sha256"]==score["source_sha256"]
            parts=np.asarray(Image.open(native_source))
            assert np.array_equal(parts>0,raw_mask)
            active_mask=(parts==region)&raw_mask
        active=mask_measure(active_mask,owners,keys,materials,direction)
        parent=mask_measure(raw_mask,owners,keys,materials,direction) if region is not None else None
        mask_primary=active["dominant_same_direction"]
        blocked=Counter()
        anchor_centers=Counter()
        native_table=Counter()
        associations=Counter()
        accepted=c.get("export_uv")
        accepted=None if accepted is None else {tuple(x) for x in accepted}
        uv=np.array([(s["u"],s["v"]) for s in c["cells"]])
        image_xy=project(np.array(c["H"]),uv+.5)
        selected_count=0; duplicate=0; removed_inside=0
        for cell, xy in zip(c["cells"],image_xy):
            ij=(cell["u"],cell["v"])
            if accepted is not None and ij not in accepted:
                removed_inside+=1
                continue
            out=tuple(int(x) for x in begin+axis_u*ij[0]+axis_v*ij[1])
            if out in exported_owners:
                duplicate+=1
                continue
            selected_count+=1
            world=align(out)
            material="minecraft:"+cell["block_id"].removeprefix("minecraft:")
            exported_owners[out]=material
            assert predicted[world]==material
            owner=source_cell(xy)
            native=keys[owner] if owner else None
            source_status="background_or_off_frame"
            projected_plane=round(float(np.dot(np.array(world)+positive_face,world_n)),5)
            native_cuboid_plane=None
            geometry_source_correct=bool(owner and world==site_for[owner])
            pixel_owner_mat=materials[owner] if owner else None
            if owner:
                if native[0]!=direction:
                    source_status="different_normal"
                elif native!=mask_primary:
                    source_status="parallel_other_source_plane"
                else:
                    source_status="same_as_source_mask_plane"
                native_cuboid_plane=float(np.dot(np.array(site_for[owner])+positive_face,world_n))
                native_table[native]+=1
            final_status=("background_or_off_frame" if not owner else
                "different_normal" if native[0]!=direction else
                "in_front_of_native_plane" if projected_plane>native_cuboid_plane else
                "behind_native_plane" if projected_plane<native_cuboid_plane else
                "right_plane_other_coordinate" if not geometry_source_correct else
                "right_coordinate_wrong_material" if material!=pixel_owner_mat else "right_coordinate_exact")
            good=(world in truth)
            assigned_id_correct=good and material==truth[world]
            anchor_centers[source_status]+=1
            blocked[final_status]+=1
            row_extra=not good
            associations[source_status+"|"+final_status]+=1
            if row_extra: associations["extra:"+source_status+"|"+final_status]+=1
            if row_extra:
                blocked["extra:"+final_status]+=1
                anchor_centers["extra:"+source_status]+=1
            positions_all.append(dict(chart=cid,uv=ij,xy=xy.tolist(),xyz=world,source=source_status,native_plane=native,face_id=owner,
                                      assigned_depth=projected_plane,native_cube_depth=native_cuboid_plane,result=final_status,extra=row_extra,output_material=material,
                                      native_material=pixel_owner_mat,exact_at_aligned_position=assigned_id_correct))
        total_source_native=sum(native_table.values())
        native_dominant=native_table.most_common(1)
        uniform_wrong_depth=bool(native_dominant and native_dominant[0][1]>=.9*total_source_native
            and total_source_native>=10 and
            max(blocked["in_front_of_native_plane"],blocked["behind_native_plane"])>=.9*total_source_native)
        chart_rows.append(dict(id=cid,source_id=base_id,region=region,direction=direction,accepted_exported=selected_count,
                               region_map_sha256=part_hash,
                               duplicate_cells=duplicate,export_uv_dropped=removed_inside,source_mask=active,source_parent_mask=parent,
                               center_source=anchor_centers,center_vs_placement=blocked,source_placement_pairs=associations,dominant_centers=plane_rows(native_table),
                               coherent_wrong_depth=uniform_wrong_depth))
    assert exported_owners==local
    saves=OUT/"conditions"/p
    saved=dict(id=n,panel=p,dataset=row["dataset"],split=row["split"],source=row["source"],pack_sha256=row["files"]["pack"]["sha256"],
               reference=ref_hash,score=original,max_native_corner_error_px=max_px,charts=chart_rows)
    save(saves/(n+".json.gz"),saved)
    if n in EXAMPLES:
        save(saves/(n+"-sites.json.gz"),positions_all)
    print(p,n,len(chart_rows),len(local),flush=True)
    return saved


EXAMPLES={"train-stable-yard-1-warm-low-left","train-stable-yard-1-warm-high-right","test-castle-01-ashlar-high-right",
          "test-castle-02-ashlar-high-right","train-viaduct-01-riverstone-low-left","train-viaduct-02-riverstone-high-right",
          "train-pillars-01-marble-low-left","test-walled-garden-1-warm-high-right"}


def all_results(views):

    aggregate={}
    for label,check in [("all",views),("sky",[r for r in views if r["panel"]=="sky"])]:
        chosen=[(r,c) for r in check for c in r["charts"]]
        stats=dict(conditions=len(check),charts=len(chosen),emitted=sum(c["accepted_exported"] for r,c in chosen),
                   geometry=totals([r["score"] for r in check]))
        for k in ("center_source","center_vs_placement","source_placement_pairs"):
            counter=Counter()
            for _,c in chosen:counter.update(c[k])
            stats[k]=counter
        stats["mask_measure"]={k:sum(c["source_mask"][k] for _,c in chosen) for k in ("parallel_mix","orientation_mix","same_plane_material_mix")}
        stats["affected_conditions"]={k:len({(r["panel"],r["id"]) for r,c in chosen if c["source_mask"][k]})
                                     for k in ("parallel_mix","orientation_mix","same_plane_material_mix")}
        texture_only=[(r,c) for r,c in chosen if c["source_mask"]["same_plane_material_mix"]
                      and not (c["source_mask"]["parallel_mix"] or c["source_mask"]["orientation_mix"])]
        stats["same_plane_material_only"]={
            "charts":len(texture_only),"conditions":len({(r["panel"],r["id"]) for r,c in texture_only})}
        stats["coherent_placed_wrong_depth"]=[dict(panel=r["panel"],id=r["id"],chart=c["id"],cells=c["accepted_exported"],
                                                  center_vs_placement=c["center_vs_placement"]) for r,c in chosen if c["coherent_wrong_depth"]]
        aggregate[label]=stats
    case_ranking={}
    for key,values in [("parallel_mix",("center_source","parallel_other_source_plane")),
                       ("normal_mix",("center_source","different_normal")),
                       ("wrong_plane",("center_vs_placement",None))]:
        rows=[]
        for r in views:
            counter=Counter()
            for c in r["charts"]:counter.update(c[values[0]])
            n=(counter[values[1]] if values[1] else counter["in_front_of_native_plane"]+counter["behind_native_plane"])
            rows.append((n,r["panel"],r["id"]))
        case_ranking[key]=sorted(rows,reverse=True)[:15]
    return aggregate,case_ranking


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--examples",action="store_true")
    args=parser.parse_args()
    pin=read(PIN/"manifest.json")
    assert digest(PIN/"exports.zip")==pin["archive"]["sha256"]
    cases=pin["conditions"]
    assert len(cases)==84
    if args.examples:
        cases=[c for c in cases if c["id"] in EXAMPLES]
    lookup={(p,c["id"]):c for p in {c["panel"] for c in cases} for c in paths(p)[0][3]}
    with ProcessPoolExecutor(6) as pool:
        checked=list(pool.map(describe,cases,[lookup]*len(cases)))
    stats,ranking=all_results(checked)
    save(OUT/("example-audit.json" if args.examples else "full-audit.json"),
         dict(archive_sha256=pin["archive"]["sha256"],manifest_sha256=digest(PIN/"manifest.json"),
              min_mask_share=MIXED_MIN_SHARE,min_mask_pixels=MIXED_MIN_PIXELS,stats=stats,ranking=ranking,views=checked))
    print("AGGREGATE",stats,flush=True)
    print("RANK",ranking,flush=True)
