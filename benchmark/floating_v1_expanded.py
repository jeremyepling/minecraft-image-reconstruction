"""Apply the frozen expanded, split-border and pre-grid Floating rules to newly admitted V1 inputs."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark import floating_corner_pair_repair as pair, floating_edge_aligned_ruler as edge
from benchmark import floating_package_rechecks as pink, floating_pregrid_depth_refit as pregrid
from benchmark import floating_v1_run as v1
from benchmark.expanded_baseline import digest,read,save
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.floating_corner_overwrite import homography
from benchmark.floating_edge_repeat_fit import repeat_fit,validate_repeats
from benchmark.floating_face_proposals import image_axes,source_offers
from benchmark.floating_peer_atlas_search import search
from benchmark.floating_verified_wall_expansion import peer_pitches,project
from benchmark.relative_oracle_charts import fit_homography
from benchmark.floating_v1_export_filter import with_visible_export


def source_transitions(chart, rgb):
    """Measure within-cell color changes in both chart axes on spatially distributed cells."""
    uv = np.array([(cell["u"], cell["v"]) for cell in chart["cells"]])
    uv = uv[np.lexsort((uv[:, 1], uv[:, 0]))]
    sampled = uv[np.linspace(0, len(uv) - 1, min(192, len(uv))).astype(int)]
    grid = (np.arange(32) + .5) / 32
    offsets = np.stack(np.meshgrid(grid, grid), axis=-1)
    pixels = np.floor(project(np.asarray(chart["H"]), sampled[:, None, None, :] + offsets)).astype(int)
    x = np.clip(pixels[..., 0], 0, rgb.shape[1] - 1)
    y = np.clip(pixels[..., 1], 0, rgb.shape[0] - 1)
    patches = rgb[y, x].astype(np.float32)
    medians = []
    for axis in (1, 2):
        steps = np.linalg.norm(np.diff(patches, axis=axis), axis=-1)
        counts = (steps >= 4).sum(axis=axis).mean(axis=1)
        medians.append(float(np.median(counts)))
    return medians


def expand(seeds,planes,image,rgb,*,texture_checks=None):
    linear=None
    charts=[]
    changed=[]
    for seed in seeds:
        cid=seed["id"]
        if not seed["cells"] or abs(seed["normal_camera"][1])>=.67:
            charts.append(seed)
            continue
        mask=pink.native_mask(planes[cid],image.size)
        points=np.column_stack(np.nonzero(mask)[::-1])+.5
        grown=preserve_and_rank(seed,resample(seed,np.asarray(seed["H"]),mask,rgb,points),rgb,mask)
        if len(grown["cells"])>=256:
            if linear is None:
                linear=image_axes(image,list(planes.values()))
            offers=source_offers(rgb,mask,grown["normal_camera"],linear)
            pick=max((o for o in offers if o["accept"]),key=lambda o:o["score"],default=None)
            if pick is not None:
                H=fit_homography([0,0],np.asarray(pick["quad"])[[3,2,1,0]])
                measured=resample(grown,H,mask,rgb,points)
                alt=preserve_and_rank(dict(seed,cells=[]),measured,rgb,mask)
                alt["confidence"]=grown["confidence"]
                if 4*len(alt["cells"])<=len(grown["cells"]):
                    grown=alt
                    changed.append(cid)
        if cid not in changed and len(grown["cells"]) >= max(512, 16 * len(seed["cells"])):
            transitions = source_transitions(grown, rgb)
            bypass = max(transitions) <= 2
            if texture_checks is not None:
                texture_checks[cid] = dict(seed_cells=len(seed["cells"]), grown_cells=len(grown["cells"]),
                                           median_transitions=transitions, bypass=bypass)
            if bypass:
                grown = seed
        charts.append(grown)
    return charts,changed


def edge_refit(chart,all_charts,mask,rgb):
    ratios=list(peer_pitches(chart,all_charts).values())
    answer,_=search(chart,ratios,mask,rgb)
    if answer is None:
        return chart
    peer=edge.extend(chart,answer,mask,rgb)
    xy=np.column_stack(np.nonzero(mask))
    center=np.median(xy,axis=0)[::-1]
    vps=[edge.vanishing(s,center,np.ptp(xy,axis=0).max())[0] for s in edge.gather(rgb,mask,np.asarray(peer["H"]))]
    if any(v is None for v in vps):
        return chart
    initial=np.asarray(chart["H"])
    shape=edge.aligned(initial,vps,project(np.linalg.inv(initial),center))
    proposal,audit=search(dict(chart,H=shape.tolist()),ratios,mask,rgb,edge.middle)
    if proposal is None or not audit.get("near",{}).get("ok"):
        return chart
    improved,refined=repeat_fit(proposal,audit["near"]["material"],mask,rgb)
    fresh=edge.extend(peer,improved,mask,rgb)
    if not validate_repeats(np.asarray(fresh["H"]),refined,mask,rgb)["accepted"]:
        return chart
    peers=[p for p in all_charts if p["id"]!=chart["id"]]
    return edge.correct_material(fresh,chart,peers,mask,rgb)


def pair_fix(packet,seeds,planes,counts,rgb,name):
    location=Path("sky")/name
    dense={p["id"]:p for p in read(v1.OUT/"dense"/location/"dense-charts.json")["charts"]}
    shared={p["id"]:p for p in read(v1.OUT/"shared"/location/"charts.json")["charts"]}
    charts={p["id"]:p for p in packet["charts"]}
    seams=read(v1.OUT/"shared"/location/"seams.json")["seams"]
    def offered(cid):
        old=dense.get(cid)
        new=charts.get(cid)
        return bool(old and old["cells"] and new and new.get("method")=="shared-corner-rows"
                    and cid in shared and cid in seeds
                    and all(np.allclose(homography(c),homography(new),rtol=1e-9,atol=1e-9) for c in (shared[cid],seeds[cid]))
                    and not np.allclose(homography(old),homography(new),rtol=1e-7,atol=1e-6))
    offered_pairs=[s for s in seams if s["decision"]=="accepted" and all(offered(s[k]) for k in ("donor_id","receiver_id"))]
    if not offered_pairs:
        return packet["charts"],[]
    options,proposals,main,edges=pair.alternatives(packet,dense,offered_pairs,list(planes.values()),counts,rgb)
    arm=pair.select(options,main,edges)
    return proposals.get(arm,packet["charts"]),next((o["pair"] for o in options if o["arm"]==arm),[])


def expanded_packet(case, destination, *, field_cache=None):
    n=case["id"]
    rel=Path("sky")/n/"charts.json"
    handoff=v1.handoff(n)
    seeds_packet=read(v1.OUT/"seeds/connected"/rel)
    seeds={c["id"]:c for c in seeds_packet["charts"]}
    planes={p["id"]:p for p in read(handoff/"surfaces.json")["planes"]}
    counts={p["id"]:p for p in read(handoff/"counts.json")["surfaces"]}
    image=Image.open(case["variants"]["sky"]["image"]).convert("RGB")
    rgb=np.asarray(image)
    texture_checks={}
    raw,contracted=expand(seeds_packet["charts"],planes,image,rgb,texture_checks=texture_checks)
    initial=pink.guard_targets(raw,seeds,planes,rgb)
    edge_charts=[edge_refit(c,raw,pink.native_mask(planes[c["id"]],image.size),rgb) if c["id"] in initial else c for c in raw]
    edge_ids=[a["id"] for a,b in zip(raw,edge_charts,strict=True) if a!=b]
    # Both surviving O2 transforms are rechecked on the actual edge-aligned result
    # if two source decisions activate in the same previously unseen frame.
    selected,pairs=pair_fix(dict(seeds_packet,charts=edge_charts),seeds,planes,counts,rgb,n)
    packet=dict(seeds_packet,charts=selected)
    group="v1"
    base=destination/"split-border"
    package=destination/"package"
    save(package/"charts"/rel,packet)
    flags=pink.guard_targets(selected,seeds,planes,rgb) if selected!=raw else initial
    fields={}
    if flags:
        normal,dist,_=v1.source_fields(group,case,v1.SOURCE,image.size,cache=field_cache)
        for cid in flags:
            mask=pink.native_mask(planes[cid],image.size)
            coarse,meta=pink.depth.partition(rgb,mask,normal,dist)
            fields[cid]=pink.depth.refine_adjacent(rgb,mask,normal,dist,coarse,meta)[0]
    split,_=pink.partition_charts(selected,flags,fields)
    bordered,dropped=pink.border_filter(split,pink.sky_support(image))
    baseline_file=base/"charts"/rel
    save(baseline_file,dict(packet,charts=bordered))
    return packet,dict(contracted=contracted,edge=edge_ids,pair=pairs,split=sorted(flags),border_dropped=dropped,
                       texture_checks=texture_checks)


def run(case):
    n=case["id"]
    group="v1"
    rel=Path("sky")/n/"charts.json"
    base=v1.OUT/"split-border"
    package=v1.OUT/"package"
    baseline_file=base/"charts"/rel
    handoff=v1.handoff(n)
    packet,decisions=expanded_packet(case,v1.OUT)
    pin=dict(source=dict(sha256=case["rgb_sha256"]),reference={k:dict(sha256=v) for k,v in case["reference_sha256"].items()},
             selected_package_chart_sha256=digest(package/"charts"/rel),surfaces_sha256=digest(handoff/"surfaces.json"))
    def inputs(group):
        return v1.SOURCE,v1.OUT/"seeds/connected",package/"charts",[case]
    with ExitStack() as ctx:
        ctx.enter_context(patch.object(v1.assembly,"load_reference",v1.score_reference))
        ctx.enter_context(patch.object(v1.assembly,"assemble",with_visible_export(v1.assembly.assemble)))
        ctx.enter_context(patch.object(pregrid,"data",inputs))
        ctx.enter_context(patch.object(pregrid,"OUT",v1.OUT/"pregrid"))
        ctx.enter_context(patch.object(pregrid,"fields",v1.source_fields))
        pregrid.replay(group,case,base/"charts",base/"assembly",{})
        roots=dict(adopted=base/"assembly",noop=base/"assembly")
        pregrid.fit_case(group,case,pin,[],False,v1.SOURCE,package,baseline_file,roots)
    row=pregrid.finalize(read(v1.OUT/"pregrid/v1/rows"/(n+".json.gz")),case)
    sparse=v1.OUT/"sparse/assembly"
    if not (sparse/v1.PREFIX/n/"score.json").is_file():
        v1.sparse(case)
    row.update(sparse=v1.horizontal.read(sparse/v1.PREFIX/n/"score.json"),**decisions,source_fields=read(v1.FIELDS/n/"receipt.json"))
    save(v1.OUT/"rows"/(n+".json.gz"),row)
