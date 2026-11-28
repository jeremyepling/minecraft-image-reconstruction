"""Test whether a visible horizontal patch can register two independently counted wall rims."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay, relative_face_assembly as solver
from benchmark import relative_face_round as round_
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, read, save
from benchmark.expanded_wall_rulers import cases
from engine.surface_perimeter import _native_alpha

OUT = ROOT / "runs/bridges"
FRESH = WORKTREES / "floating-fresh-validation/runs/floating-fresh"
OLD = WORKTREES / "expanded-wall-rulers/runs/wall-rulers"
ORDINARY = WORKTREES / "crop-ceiling-expanded/runs/native-crop-ceiling/source"


def horizontal_rims(charts, planes, size):
    boundaries = [solver.face_boundary(chart) for chart in charts]
    edges=[e for boundary in boundaries for e in boundary]
    if not edges:
        return []
    corners=np.stack([p for e in edges for p in e[:2]])
    margin=np.median([e[5] for e in edges])*2
    lower,upper=corners.min(axis=0)-margin,corners.max(axis=0)+margin
    groups = []
    for plane in planes:
        if abs(plane["normal_camera"][1]) < .67:
            continue
        alpha, (x, y, x1, y1) = _native_alpha(plane["support"], size)
        # An unbounded backdrop or floor cannot provide a build-relative bridge.
        if min(x,y,size[0]-x1,size[1]-y1)<=1 or np.any([x,y]<lower) or np.any([x1,y1]>upper):
            continue
        mask = np.asarray(alpha) > 0
        # Three pixels accommodates the separately fitted chart and mask edge.
        guard = 3
        mask = cv2.dilate(np.pad(mask.astype("uint8"), guard), np.ones((7,7),np.uint8))
        rims = []
        for idx, boundary in enumerate(boundaries):
            for a,b,l,r,axis,length,_,_ in boundary:
                if axis == 1:
                    continue
                points = np.floor(np.stack([.8*a+.2*b,.5*(a+b),.2*a+.8*b])).astype(int)
                pts = points - [x-guard,y-guard]
                if np.any(pts < 0) or np.any(pts >= mask.shape[::-1]):
                    continue
                if mask[pts[:,1], pts[:,0]].all():
                    rims.append((idx,a,b,l,r,axis,length))
        if len({r[0] for r in rims}) >= 2:
            groups.append((plane["id"], rims))
    return groups


def candidates(charts, planes, size):
    groups = horizontal_rims(charts, planes, size)
    ballots = defaultdict(lambda: defaultdict(list))
    inverses = [np.linalg.inv(c["H"]) for c in charts]
    bases = [np.array([solver.axis_vector(a) for a in c["axes"]]) for c in charts]
    for plane, rims in groups:
        for first, second in combinations(rims, 2):
            i,a,b,l,r,axis,length = first
            j,c,d,s,t,axis2,other = second
            if i == j or np.linalg.norm((a+b-c-d)*.5) > 5 * min(length,other):
                continue
            if axis == axis2:
                if charts[i]["normal_axis"] != charts[j]["normal_axis"]:
                    continue
                # Sample an integer lattice endpoint and verify its next endpoint independently.
                at = c
                uv_end = solver.project(inverses[i], d)
                uv_start = solver.project(inverses[i], c)
                if uv_start is None or uv_end is None or np.linalg.norm((uv_end-uv_start)@bases[i]-(t-s)) > .18:
                    continue
            else:
                mat = np.column_stack((b-a,c-d))
                if abs(np.linalg.det(mat)) < .12 * length*other:
                    continue
                u,v = np.linalg.solve(mat,c-a)
                if not (-2 <= u <= 3 and -2 <= v <= 3):
                    continue
                at = a + (b-a)*u
            uv1, uv2 = solver.project(inverses[i],at), solver.project(inverses[j],at)
            if uv1 is None or uv2 is None:
                continue
            x1, x2 = uv1@bases[i], uv2@bases[j]
            if max(abs(x1[1]-l[1]),abs(x2[1]-s[1])) > .16:
                continue
            shift = x1-x2
            error = np.linalg.norm(shift-np.rint(shift))
            if error > .18:
                continue
            delta = tuple(map(int,np.rint(shift)))
            ballots[i,j][delta].append((plane,tuple(l),tuple(s),float(error),[a.tolist(),b.tolist()]))
    results = []
    for (i,j), options in ballots.items():
        ranked = []
        for delta, rows in options.items():
            n = min(len({r[1] for r in rows}),len({r[2] for r in rows}))
            ranked.append((n,-float(np.mean([r[3] for r in rows])),delta,rows))
        ranked.sort(reverse=True)
        n,negative,delta,rows = ranked[0]
        if n < 2 or (len(ranked)>1 and ranked[1][0]>=n):
            continue
        results.append(dict(i=i,j=j,shift=delta,edges=n,error=-negative,kind="horizontal-bridge",
                            plane=rows[0][0],image_edges=[r[4] for r in rows[:4]]))
    return sorted(results,key=lambda r:(-r["edges"],r["error"])),len(groups)


def horizontal_charts(walls, planes, size, rgb):
    """Use two measured perpendicular rim directions to count an observed top or underside."""
    from benchmark.expanded_wall_rulers import resample
    from benchmark.expanded_wall_source_resample import preserve_and_rank

    planes_by_id = {p["id"]:p for p in planes}
    result = []
    for pid, rims in horizontal_rims(walls, planes, size):
        proposals = []
        for left, right in combinations(rims,2):
            i,a,b,_,_,axis,length = left
            j,c,d,_,_,second,other = right
            if i == j or axis == second:
                continue
            mat = np.column_stack((b-a,c-d))
            if abs(np.linalg.det(mat)) < .12*length*other:
                continue
            u,v = np.linalg.solve(mat,c-a)
            if not (-2 <= u <= 3 and -2 <= v <= 3):
                continue
            cross = a+(b-a)*u
            columns = {}
            error = []
            for k, direction in ((i,axis),(j,second)):
                wall = walls[k]
                h = np.array(wall["H"])
                uv = solver.project(np.linalg.inv(h),cross)
                which = next(t for t,code in enumerate(wall["axes"]) if abs(code)==direction+1)
                unit = np.sign(wall["axes"][which])*h[:,which]
                columns[direction] = unit / (h @ [*uv,1])[2]
                error.append(abs(uv[which]-round(uv[which])))
            if max(error) <= .18:
                h = np.column_stack((columns[0],columns[2],[*cross,1]))
                if abs(np.linalg.det(h))>1e-6:
                    proposals.append((sum(error),abs(u-.5)+abs(v-.5),h))
        if not proposals:
            continue
        h = min(proposals,key=lambda p:p[:2])[2]
        plane = planes_by_id[pid]
        alpha,(x,y,x1,y1) = _native_alpha(plane["support"],size)
        mask=np.zeros(rgb.shape[:2],bool)
        mask[y:y1,x:x1]=np.asarray(alpha)>0
        points=np.column_stack(np.nonzero(mask)[::-1])+.5
        # Image camera normals have a downward-positive Y component.
        raw=dict(id="horizontal-"+pid,normal_axis=2 if plane["normal_camera"][1]<0 else -2,
                 normal_camera=plane["normal_camera"],axes=[1,3],cells=[])
        chart=resample(raw,h,mask,rgb,points)
        if not chart["cells"]:
            continue
        chart=preserve_and_rank(raw,chart,rgb,mask)
        # Confirm cell boundaries on the materialized grid agree with two actual wall charts.
        joins,_=solver.propose_joins([*walls,chart],min_edges=1,edge_pixels=2.5)
        adjoining = sorted({walls[join["i"]]["id"] for join in joins if join["j"]==len(walls)})
        if len(adjoining)>=2:
            result.append(dict(chart,supporting_walls=adjoining))
    return result


def run(job):
    case, fresh, tiles = job
    cv2.setNumThreads(1)
    name = case["id"]
    src = Path(case["frame"]) / "rgb.png"
    source = FRESH / "ordinary" if fresh else ORDINARY
    charts = (FRESH if fresh else OLD) / "source_material/charts"
    handoff = source / "sky" / name / "source-handoff"
    parts = read(handoff / "surfaces.json")["planes"]
    with Image.open(src) as image:
        size = image.size
        rgb = np.asarray(image.convert("RGB"))
    output=OUT.parent/"horizontal-cells" if tiles else OUT
    audit = {}
    existing = solver.coplanar_joins
    orient = solver.oriented
    def add_tiles(inputs):
        walls,hint=orient(inputs)
        extra=horizontal_charts(walls,parts,size,rgb)
        audit.update(tiles=len(extra),tile_cells=sum(len(c["cells"]) for c in extra),horizontal_charts=extra)
        return walls+extra,hint
    def add_bridges(oriented, tolerance):
        result = existing(oriented, tolerance)
        extra, eligible = candidates(oriented, parts, size)
        audit.update(eligible_masks=eligible,proposed=extra)
        return result + extra
    def read_ref(path):
        value = read(path)
        if path == src.with_name("frame.json") and "framing" not in value:
            proof = next(r for r in read(ROOT/"training/framing-audit-v2.json")["frames"] if r["id"]==name)
            assert proof["accepted"] and proof["rgb_sha256"] == case["rgb_sha256"]
            value = dict(value,framing=proof)
        return value
    with patch.object(solver,"coplanar_joins",existing if tiles else add_bridges),patch.object(solver,"oriented",add_tiles if tiles else orient),patch.object(floating_cloud_replay,"RUN",source),patch.object(
            round_,"ASSEMBLY",output/"assembly"),patch.dict(round_.MODES,{"connected-atlas":(charts,"charts.json",False,1,2.5,.5,False)}),patch.object(expanded_scores,"read",read_ref):
        round_.evaluate((case,"sky","connected-atlas"))
    save(output/"audit"/(name+".json"),dict(id=name,**audit))


def fresh_cases(supplemental=False):
    names=(("train-watchpost2-cool","train-observatory-02-verdigris","train-riverside-mill-2-warm") if supplemental else
           ("validation-cloister2-cool","train-arcade2-cool","validation-harbor-02-seafoam",
            "train-factory-02-kiln","validation-hillside-chapel-2-warm","train-market-court-2-warm"))
    for ds in ("visible-builds-v2","visible-builds-v3","decorative-builds-v1"):
        for row in read(SOURCE/ds/"manifest.json")["frames"]:
            if not any(row["id"] == p+"-"+view for view in ("low-left","high-right") for p in names):
                continue
            frame = SOURCE/ds/"data"/row["split"]/"frames"/row["id"]
            yield dict(row,dataset=ds,frame=str(frame),variants={"sky":dict(image=str(frame/"rgb.png"),source_sha256=row["rgb_sha256"])})


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fresh",action="store_true")
    cli.add_argument("--tiles",action="store_true")
    cli.add_argument("--supplemental",action="store_true")
    cfg=cli.parse_args()
    fresh=cfg.fresh or cfg.supplemental
    rows=list(fresh_cases(cfg.supplemental) if fresh else cases())
    assert len(rows)==(6 if cfg.supplemental else 12 if cfg.fresh else 38)
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(run,[(r,fresh,cfg.tiles) for r in rows]))
