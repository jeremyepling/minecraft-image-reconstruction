"""Re-establish a visible depth patch's tangent grid, keeping parent joins and signed depth fixed."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_source_resample import TEXELS, atlas, rank_tile
from benchmark.floating_bridges_report import summarize
from benchmark.floating_one_face_control import face_points, visible_quad
from benchmark.floating_part_depth import (LOCATION, PANELS, SAMPLES as DEPTH_SAMPLES, boundary_order,
    combined, components, fields, paths, reference_offsets)
from benchmark.floating_part_report import totals
from benchmark.relative_face_scoring import yaw_from_anchor_normal
from benchmark.relative_oracle_charts import UNIT, fit_homography
from benchmark.shared_corner_rows import sample
from engine.geometry import bilinear, project
from engine.surface_perimeter import _native_alpha

OUT = ROOT / "runs/depth-refit"
REVIEW = ROOT / "benchmark/reviews/floating-depth-refit-01"
PRIOR = WORKTREES / "floating-part-grids/runs/part-depth"
FLAT = WORKTREES / "floating-extra-isolation/runs/extra-isolation"
ARCHIVE = ROOT / "benchmark/reviews/floating-part-depth-01/conditions.json.gz"
ARMS = ("same-grid-resample", "source-refit", "native-face-refit")
EXTENDED = "source-refit-with-small-steps"
STEP_ONLY = "small-steps-with-inherited-grid"
GUARDED = "signed-boundary-guard"
SAMPLES = np.stack(np.meshgrid((np.arange(9)+.5)/9, (np.arange(9)+.5)/9), -1)


def patch_mask(prior, uv, native):
    """Rasterize the existing component in pixel centers, clipping to original visible support."""
    points = np.column_stack(np.nonzero(native)[::-1]) + .5
    uv_pixels = np.floor(project(np.linalg.inv(prior), points)).astype(int)
    lo, hi = uv.min(0), uv.max(0)
    lookup = np.zeros((hi-lo+1)[::-1], bool)
    lookup[tuple((uv-lo).T[::-1])] = True
    inside = ((uv_pixels >= lo) & (uv_pixels <= hi)).all(1)
    in_bounds = np.flatnonzero(inside)
    inside[in_bounds] = lookup[tuple((uv_pixels[in_bounds]-lo).T[::-1])]
    xy = points[inside]
    mask = np.zeros_like(native)
    mask[tuple(xy.astype(int).T[::-1])] = True
    return mask, xy


def edge_refit(prior, uv, rgb, mask):
    """Search pitch and phase against RGB edge contrast only within this relief patch."""
    resolution = 16
    lo, hi = uv.min(0)-1, uv.max(0)+2
    yy, xx = np.mgrid[:(hi[1]-lo[1])*resolution, :(hi[0]-lo[0])*resolution]
    xy = project(prior, np.stack((xx, yy), -1)/resolution + lo)
    uv_rgb = cv2.remap(rgb, *(xy[...,d].astype('f')-.5 for d in (0,1)), cv2.INTER_LINEAR)
    uv_rgb = cv2.GaussianBlur(uv_rgb.astype('f'), (5,5), 1.)
    valid = sample(mask, xy, 1)
    pivot = np.median(uv+.5, axis=0)
    transform = np.eye(3)
    axes = []
    for axis in (0,1):
        values = uv_rgb if axis == 0 else uv_rgb.transpose(1,0,2)
        covered = valid if axis == 0 else valid.T
        # Only count a boundary when both sides have pixels on the same patch.
        good = covered[:,2:] & covered[:,:-2] & covered[:,1:-1]
        grad = np.linalg.norm(values[:,2:] - values[:,:-2], axis=2)
        weight = good.sum(0)
        profile = np.where(weight >= 6, (np.minimum(grad,100.)*good).sum(0) / np.maximum(weight,1), 0.)
        domain = np.arange(1,len(profile)+1)/resolution + lo[axis]
        lines = np.arange(uv[:,axis].min(), uv[:,axis].max()+2)
        def score(pitch, phase):
            edges = pivot[axis] + pitch*(lines-pivot[axis]) + phase
            counts = np.interp(edges, domain, weight, left=0, right=0)
            strengths = np.interp(edges, domain, profile)
            guards = .5*(np.interp(edges-.25*pitch,domain,profile)+np.interp(edges+.25*pitch,domain,profile))
            accepted = counts >= 6
            if accepted.sum() < 3:
                return 0.
            return float(np.average((strengths-guards)[accepted],weights=np.sqrt(counts[accepted])))
        baseline = score(1., 0.)
        options = [(score(p,s),p,s) for p in np.linspace(.85,1.15,13) for s in np.linspace(-.375,.375,13)]
        best, pitch, phase = max(options,key=lambda row:row[0]-.5*abs(1.-row[1])-.15*abs(row[2]))
        if best < max(1.5,baseline+1.0):
            pitch, phase = 1., 0.
        transform[axis,axis] = pitch
        transform[axis,2] = (1-pitch)*pivot[axis] + phase
        axes.append(dict(pitch=float(pitch),phase=float(phase),identity_score=baseline,best_score=best))
    return prior @ transform, axes


def small_steps(chart, uv, depth, native, field):
    """Detach small parallel parts only when adjacent pixels show an ordinal depth step."""
    pixels = project(np.asarray(chart["H"]), uv[:,None,:]+DEPTH_SAMPLES)
    size = np.array(native.shape[::-1])
    normal, distance = field["normal"],field["depth"]
    normals = np.median(bilinear(normal,pixels*normal.shape[1::-1]/size),axis=1)
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    inv_depth = np.median(bilinear(distance[:,:,None],pixels*distance.shape[1::-1]/size)[:,:,0],axis=1)
    good = (sample(native,pixels,1).sum(1)>=5) & (normals@chart["normal_camera"]>=.9) & (depth==0)
    xy = pixels[:,4,:]
    index = {tuple(uv[i]):i for i in np.flatnonzero(good)}
    links = [(i,index[t]) for (u,v),i in index.items() for t in ((u+1,v),(u,v+1)) if t in index]
    if len(links)<12:
        return []
    a,b = np.array(links).T
    change = inv_depth[b]-inv_depth[a]
    delta = xy[b]-xy[a]
    keep = np.ones(len(links),bool)
    for _ in range(3):
        slope,_,rank,_ = np.linalg.lstsq(delta[keep],change[keep],rcond=None)
        if rank<2:
            return []
        residual = change-delta@slope
        keep = abs(residual)<=np.quantile(abs(residual),.70)
    # The field is relative inverse depth. Estimate its local noise on this chart.
    threshold = max(4*np.median(abs(residual)),.018*np.median(inv_depth[good]))
    neighbors = {i:[] for i in index.values()}
    for left,right,step in zip(a,b,residual):
        if abs(step)<threshold:
            neighbors[left].append(right)
            neighbors[right].append(left)
    groups = []
    pending = set(neighbors)
    while pending:
        first = pending.pop()
        group,stack = {first},[first]
        while stack:
            for i in neighbors[stack.pop()]:
                if i in pending:
                    group.add(i)
                    stack.append(i)
                    pending.remove(i)
        groups.append(group)
    lengths = {i:len(g) for g in groups for i in g}
    proposals = []
    for group in groups:
        if not 2<=len(group)<=12:
            continue
        jumps = []
        for left,right,step in zip(a,b,residual):
            if left in group and right not in group and lengths[right]>=2*len(group):
                jumps.append(-step)
            elif right in group and left not in group and lengths[left]>=2*len(group):
                jumps.append(step)
        if jumps and abs(np.mean(np.sign(jumps)))==1 and min(abs(np.asarray(jumps)))>=threshold:
            proposals.append((np.array(sorted(group)),int(np.sign(jumps[0])),len(jumps),
                              float(np.median(jumps)),float(threshold)))
    return proposals


def boundary_guard(prior, uv, depth, piece, native, field):
    """Confirm the sign at smaller, independent source samples across the proposed join."""
    distance = field["depth"]
    scale = np.asarray(distance.shape[1::-1]) / np.asarray(native.shape[::-1])
    center = project(prior,uv+.5)
    values = bilinear(distance[:,:,None],center*scale)[:,0]
    index = {tuple(pos):i for i,pos in enumerate(uv)}
    neighbors = [(i,index[pos]) for (u,v),i in index.items()
                 for pos in ((u+1,v),(u,v+1)) if pos in index]
    a,b = np.array(neighbors).T
    flat = (depth[a]==depth[b]) & sample(native,center[a],1) & sample(native,center[b],1)
    if flat.sum()<8:
        return dict(accept=False,reason="few slope samples")
    delta = center[b]-center[a]
    change = values[b]-values[a]
    keep = flat.copy()
    for _ in range(3):
        slope,_,rank,_ = np.linalg.lstsq(delta[keep],change[keep],rcond=None)
        if rank<2:
            return dict(accept=False,reason="unresolved slope")
        err = abs(change-delta@slope)
        keep = flat & (err<=np.quantile(err[flat],.7))
    threshold = max(4*np.median(err[flat]), .018*np.median(values))
    member = np.zeros(len(uv),bool)
    member[piece] = True
    boundary = (member[a] & (depth[b]==0)) | (member[b] & (depth[a]==0))
    votes = []
    for i,j in zip(a[boundary],b[boundary]):
        if not member[i]:
            i,j = j,i
        towards = uv[i]-uv[j]
        edge = (uv[i]+uv[j])/2+.5
        along = np.array([-towards[1],towards[0]])
        strip = edge + np.array([-.25,0.,.25])[:,None]*along
        left,right = [project(prior,strip+fraction*towards) for fraction in (.30,-.30)]
        valid = sample(native,left,1) & sample(native,right,1)
        if valid.sum()<2:
            continue
        dl = bilinear(distance[:,:,None],left*scale)[:,0]
        dr = bilinear(distance[:,:,None],right*scale)[:,0]
        votes.append(float(depth[i]*np.median((dl-dr-(left-right)@slope)[valid])))
    yes = sum(v>threshold for v in votes)
    return dict(accept=bool(yes>=3 and 3*yes>=2*len(votes)),contacts=len(votes),
                convincing=int(yes),threshold=float(threshold),signed_neighbor_residuals=votes)


def face_refit(prior, mask, xy, chart, labels, owners, camera):
    """A native fully visible outward unit face supplies only the patch's 2D grid."""
    orientation = yaw_from_anchor_normal(chart["normal_camera"],(0,0,1),camera)
    if orientation["yaw_quarters"] is None:
        return None, None
    count = Counter(owners[mask].tolist())
    blocks = {b["id"]:b for b in labels["blocks"]}
    normal = orientation["best_world_normal"]
    candidates = [f for f in labels["faces"] if f["normal_world"] == normal
                  and blocks[f["block"]].get("geometry_kind","full_cube")=="full_cube"
                  and count[f["id"]]==f["visible_pixels"] and f["visible_pixels"]>=16]
    center = xy.mean(0)
    candidates.sort(key=lambda f:(np.linalg.norm(np.mean(f["corners_px"],0)-center),f["id"]))
    for face in candidates:
        if not visible_quad(face,mask,owners):
            continue
        _, corners = face_points(face,camera)
        positions = project(np.linalg.inv(prior),corners)
        anchor = np.floor(positions.mean(0)).astype(int)
        idx = np.arange(4)
        orders = [np.roll(o,k) for o in (idx,idx[::-1]) for k in range(4)]
        order = min(orders,key=lambda o:np.sum((positions[o]-UNIT-anchor)**2))
        h = fit_homography(anchor,corners[order])
        assert np.max(abs(project(h,UNIT+anchor)-corners[order])) < 1e-4
        return h, dict(face=face["id"],anchor=anchor.tolist(),corners_px=corners[order].tolist())
    return None, None


def sample_cells(h, xy, mask, rgb, depth):
    """Use >50% of 81 inset samples on visible patch pixels; rank native RGB texels for each cell."""
    mapped = project(np.linalg.inv(h), xy)
    bins = np.unique(np.floor(mapped).astype(int),axis=0)
    output = []
    for chunk in np.array_split(bins, max(1,(len(bins)+255)//256)):
        coverage = sample(mask,project(h,chunk[:,None,None,:]+SAMPLES),1).mean((1,2))
        quads = project(h,chunk[:,None,:]+UNIT)
        shortest = np.linalg.norm(quads-np.roll(quads,1,axis=1),axis=2).min(1)
        for idx in np.flatnonzero((coverage>.5) & (shortest>=3.5)):
            key = chunk[idx]
            points = project(h,key+TEXELS)
            texels = np.floor(points).astype(int)
            colors = rgb[texels[...,1].clip(0,len(rgb)-1),texels[...,0].clip(0,rgb.shape[1]-1)]
            hit = rank_tile(dict(rgb=colors,mask=sample(mask,points,1)),atlas())
            output.append(dict(u=int(key[0]),v=int(key[1]),normal_offset=int(depth),
                        block_id=hit["block_id"] or "smooth_stone",fallback=not hit["block_id"],
                        mask_coverage=float(coverage[idx])))
    return output


def run(job):
    panel, case, archived = job
    cv2.setNumThreads(1)
    name = case["id"]
    rel = Path("sky")/name/"charts.json"
    flat = FLAT/panel/"large-contraction"
    original = PRIOR/panel/"source"
    if not (original/"charts"/rel).exists():
        original = flat
    packet = read(original/"charts"/rel)
    image = Path(case["variants"]["sky"]["image"])
    assert digest(image)==packet["source_sha256"]==archived["source_sha256"]
    roots = {"source-depth":original/"assembly"}
    charts_by_arm = {arm:[] for arm in (*ARMS,STEP_ONLY,EXTENDED,GUARDED)}
    parts = []
    extra_parts = []
    if any(c["source_offsets"] for c in archived["charts"]):
        rgb = np.asarray(Image.open(image).convert("RGB"))
        handoff = paths(panel)[0][0]/"sky"/name/"source-handoff"
        combined.paths.check_handoff(packet,handoff,archived["source_sha256"])
        planes = {p["id"]:p for p in read(handoff/"surfaces.json")["planes"]}
        field, hashes = fields(panel,case,paths(panel)[0][0])
        assert hashes == archived["field_sha256"]
        frame = Path(case["frame"])
        labels = read(frame/"labels.json")
        camera = read(frame/"camera.json")
        with np.load(frame/"labels.npz") as data:
            owners = data["face_id"]
        counts = np.bincount(owners.ravel().astype(int))
        assert all(counts[f["id"]]==f["visible_pixels"] for f in labels["faces"])
        for chart in packet["charts"]:
            cells = chart["cells"]
            depth = np.array([c.get("normal_offset",0) for c in cells])
            if not depth.any():
                for arm in charts_by_arm:
                    charts_by_arm[arm].append(chart)
                continue
            uv = np.array([(c["u"],c["v"]) for c in cells])
            prior = np.asarray(chart["H"])
            alpha, (x,y,r,b) = _native_alpha(planes[chart["id"]]["support"],rgb.shape[1::-1])
            native = np.zeros(rgb.shape[:2],bool)
            native[y:b,x:r] = np.asarray(alpha)>0
            replacements = {arm:[] for arm in ARMS}
            replacements[GUARDED] = []
            chart_accepted = False
            for sign in (-1,1):
                for piece in components(depth==sign,uv):
                    old_uv = uv[piece]
                    mask, xy = patch_mask(prior,old_uv,native)
                    if not len(xy):
                        continue
                    h_src, edges = edge_refit(prior,old_uv,rgb,mask)
                    h_ref, face = face_refit(prior,mask,xy,chart,labels,owners,camera)
                    record = dict(chart=chart["id"],depth=sign,old_uv=old_uv.tolist(),parent_H=prior.tolist(),
                                  edge_axes=edges,reference_anchor=face)
                    guard = boundary_guard(prior,uv,depth,piece,native,field)
                    record["boundary_guard"] = guard
                    for arm,h in zip(ARMS,(prior,h_src,h_ref)):
                        # Both refit arms use the identical sampler even on abstention.
                        h = h if h is not None else prior
                        if arm==ARMS[0] or not np.allclose(h,prior):
                            sampled = sample_cells(h,xy,mask,rgb,sign)
                        else:
                            sampled = unchanged
                        if arm==ARMS[0]:
                            unchanged = sampled
                        replacements[arm].append((set(piece.tolist()),sampled))
                        if arm=="source-refit":
                            guarded_cells = sampled if guard["accept"] else [dict(cells[i],normal_offset=0) for i in piece]
                            replacements[GUARDED].append((set(piece.tolist()),guarded_cells))
                            chart_accepted |= guard["accept"]
                        record[arm] = dict(H=h.tolist(),uv=[[c["u"],c["v"]] for c in sampled])
                    parts.append(record)
            replacements[EXTENDED] = list(replacements["source-refit"])
            replacements[STEP_ONLY] = list(replacements["source-refit"])
            for piece,sign,contacts,jump,threshold in small_steps(chart,uv,depth,native,field):
                mask,xy = patch_mask(prior,uv[piece],native)
                h,edges = edge_refit(prior,uv[piece],rgb,mask)
                sampled = sample_cells(h,xy,mask,rgb,sign)
                replacements[EXTENDED].append((set(piece.tolist()),sampled))
                if chart_accepted:
                    replacements[GUARDED].append((set(piece.tolist()),sampled))
                inherited = sample_cells(prior,xy,mask,rgb,sign)
                replacements[STEP_ONLY].append((set(piece.tolist()),inherited))
                pixels = project(prior,uv[:,None,:]+DEPTH_SAMPLES)
                _, reference_depth, _ = reference_offsets(chart,pixels,sample(native,pixels,1).sum(1)>=5,
                                                           owners,labels,camera)
                moved = np.zeros(len(uv),int)
                moved[piece] = sign
                extra_parts.append(dict(chart=chart["id"],depth=sign,contacts=contacts,depth_jump=jump,
                    threshold=threshold,old_uv=uv[piece].tolist(),parent_H=prior.tolist(),edge_axes=edges,
                    native_boundary=boundary_order(uv,moved,reference_depth),
                    H=h.tolist(),uv=[[c["u"],c["v"]] for c in sampled]))
            for arm in charts_by_arm:
                replaced = set().union(*(p for p,_ in replacements[arm]))
                retained = [c for i,c in enumerate(cells) if i not in replaced]
                rebuilt = [c for _,rows in replacements[arm] for c in rows]
                charts_by_arm[arm].append(dict(chart,export_cells=retained+rebuilt))
    for arm,charts in charts_by_arm.items():
        if not parts:
            roots[arm] = original/"assembly"
            continue
        if arm in (STEP_ONLY,EXTENDED) and not extra_parts:
            roots[arm] = roots["source-refit"]
            continue
        root = OUT/panel/arm
        save(root/"charts"/rel,dict(packet,charts=charts))
        with patch.object(combined,"inputs",lambda _:paths(panel)[0]):
            combined.run((panel,case,root/"charts",root/"assembly"))
        roots[arm] = root/"assembly"
    row = summarize([case],roots,str(LOCATION))["rows"][0]
    assert row["score"]["source-depth"]==archived["score"]["source"]
    for folder in roots.values():
        receipt = read(folder/LOCATION/name/"score.json")
        assert receipt["reference_sha256"]==archived["reference_sha256"]
        assert receipt["joins"]==read(original/"assembly"/LOCATION/name/"score.json")["joins"]
    row.update(panel=panel,split=case["split"],layout=archived["layout"],active=archived["active"],
               source_sha256=archived["source_sha256"],patches=parts,extra_patches=extra_parts,
               pack_sha256={a:read(p/LOCATION/name/"score.json")["pack_sha256"] for a,p in roots.items()})
    save(OUT/panel/"cases"/(name+".json.gz"),row)
    print(panel,name,len(parts),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--case")
    args = parser.parse_args()
    archive = {(r["panel"],r["id"]):r for r in read(ARCHIVE)}
    jobs = [(p,c,archive[p,c["id"]]) for p in PANELS for c in paths(p)[0][3]]
    if args.case:
        jobs = [j for j in jobs if j[1]["id"]==args.case]
    if args.resume:
        jobs = [j for j in jobs if not (OUT/j[0]/"cases"/(j[1]["id"]+".json.gz")).exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))
    rows = [read(path) for p in PANELS for path in sorted((OUT/p/"cases").glob("*.json.gz"))]
    if len(rows)==92:
        groups = dict(active=[r for r in rows if r["active"]],sky=[r for r in rows if r["panel"]=="sky"])
        assert [len(g) for g in groups.values()]==[84,6]
        for row in rows:
            # Bounded perfect chooser. Keep the prior export when strict F1 ties.
            options = ("source-depth","source-refit",EXTENDED,GUARDED,"native-face-refit")
            best = max(options,key=lambda a:row["score"][a]["geometry"]["strict"]["f1"])
            row["score"]["bounded-chooser"] = row["score"][best]
            row["paired"]["bounded-chooser"] = row["paired"][best]
            row["chosen_arm"] = best
        save(REVIEW/"conditions.json.gz",rows)
        save(REVIEW/"summary.json",{key:totals(group) for key,group in groups.items()})


if __name__=="__main__":
    main()
