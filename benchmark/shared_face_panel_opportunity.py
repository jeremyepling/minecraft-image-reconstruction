"""Measure connected, same-offset wrong wall panels using visible reference labels after prediction."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.camera_metrics import registration
from benchmark.roof_span_worst_stages import read_view
from benchmark.shared_face_error_budget import REVIEW, current_pack
from benchmark.strip_visibility_audit import block_footprints
from benchmark.texture_camera_eval import digest, read, save
from benchmark.visible_error_budget import projection, tally


def connected(points):
    remaining = set(points)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, active = {start}, [start]
        while active:
            pt = active.pop()
            for axis in range(3):
                for step in (-1,1):
                    neighbor = list(pt)
                    neighbor[axis] += step
                    key = tuple(neighbor)
                    if key in remaining:
                        remaining.remove(key)
                        found.add(key)
                        active.append(key)
        yield found


def panels(pairs, tangent_limit, *, shared_vector):
    grouped = defaultdict(dict)
    for p in pairs:
        axis = p['axis']
        depth = p['signed_depth']
        if axis==1 or not depth or p['tangent_distance']>tangent_limit:
            continue
        key = (p['reference_surface'],tuple(p['normal']),p['target'][axis],depth)
        if shared_vector:
            key += (tuple(p['extra_minus_target']),)
        grouped[key][tuple(p['target'])] = p
    results = []
    for key, group in grouped.items():
        for component in connected(group):
            if len(component)<3:
                continue
            pairs_here = [group[t] for t in sorted(component)]
            extra = {tuple(p['extra']) for p in pairs_here}
            results.append(dict(targets=len(component), extras=len(extra), signed_depth=key[3],
                reference_surface=key[0], normal=list(key[1]), target_plane=key[2],
                target_xyz=sorted(component), extra_xyz=sorted(extra),
                pairs=[(p['target'],p['extra']) for p in pairs_here],
                target_xyz_extent=(np.ptp(np.array(sorted(component)),axis=0)+1).tolist(),
                extra_connected=len(list(connected(extra)))==1,
                shared_vector=list(key[4]) if shared_vector else None))
    return sorted(results,key=lambda g:(-g['targets'],g['reference_surface']))


def inspect(task):
    job, diagnostics = task
    split, frame, prior, control = job
    key = frame['id']
    local, pack_sha, shift, _ = current_pack(job)
    assert pack_sha == diagnostics['pack_sha256']
    _, _, _, camera, pose, labels, ids, truth, _, _, _ = read_view(suite.CONTROL, split, frame, prior, control)
    projected, ax, world_axis, world = projection(camera,pose,local,shift,ids.shape)
    placed = set(map(tuple,world))
    lut, axes, footprint, bits, face_square, _ = block_footprints(labels,ids)
    optional = ((bits==1)|(bits==2)) & (face_square<4)
    n = len(footprint)
    source = ids.ravel()
    owner = projected.ravel()
    axes_in_world = world_axis[ax.ravel()]
    target_blocks = lut[source]
    target_xyz = np.array([[0,0,0]]+[b['xyz_world'] for b in labels['blocks']])
    missing = np.array([False]+[tuple(pt) not in placed for pt in target_xyz[1:]],bool)
    normals = np.array([[0,0,0]]+[f['normal_world'] for f in labels['faces']])
    hit = (owner>=0) & (source>0)
    same = hit & (axes_in_world==axes[source])
    count = tally(target_blocks,hit,n)
    correct_axis = tally(target_blocks,same,n)
    is_extra = np.array([tuple(pt) not in truth for pt in world],bool)
    extra_hit = np.zeros(hit.shape,bool)
    extra_hit[hit] = is_extra[owner[hit]]
    extra_votes = tally(target_blocks,extra_hit,n)
    eligible = missing & ~optional & (count*2>footprint) & (correct_axis*2>count) & (extra_votes*2>count)
    assert int(eligible.sum()) == diagnostics['covering_predictions'].get('required',{}).get('same_axis_extra',0)
    transform = registration(np.array(camera['P']),pose)
    eye_local = -np.linalg.solve(np.array(camera['P'])[:,:3],np.array(camera['P'])[:,3])
    eye = transform[:3,:3]@eye_local + transform[:3,3] + shift
    valid = np.flatnonzero(same & extra_hit & eligible[target_blocks])
    axis = axes[source[valid]]
    predicted_sign = np.sign(eye[axis] - world[owner[valid],axis] - .5)
    valid = valid[predicted_sign == normals[source[valid],axis]]
    # A single visible reference face + single extra cube must own a strict
    # majority of all hit rays on this target. Ties and mosaics abstain.
    face_owner = source[valid].astype('int64')*len(world) + owner[valid] if len(valid) else []
    ids_pair, scores = np.unique(face_owner,return_counts=True)
    pairs = []
    for pair, support in zip(ids_pair,scores):
        fid, owner_id = divmod(int(pair),len(world))
        b = lut[fid]
        if support*2<=count[b]:
            continue
        f = labels['faces'][fid-1]
        axis = int(axes[fid])
        delta = world[owner_id] - target_xyz[b]
        tangents = np.delete(delta,axis)
        pairs.append(dict(target=target_xyz[b].tolist(),extra=world[owner_id].tolist(),axis=axis,
            normal=normals[fid].tolist(),reference_surface=f['surface_id'],extra_minus_target=delta.tolist(),
            signed_depth=int(delta@normals[fid]),tangent_distance=int(abs(tangents).sum()),
            majority_rays=int(support),covered_rays=int(count[b])))
    assert len({tuple(p['target']) for p in pairs})==len(pairs)
    variants = {'depth_only':panels(pairs,0,shared_vector=True),
                'common_vector_tangent_at_most_1':panels(pairs,1,shared_vector=True),
                'common_vector_tangent_at_most_2':panels(pairs,2,shared_vector=True),
                'common_depth_varying_tangent_at_most_2':panels(pairs,2,shared_vector=False)}
    return dict(id=key,layout=diagnostics['layout'],split=split,pack_sha256=pack_sha,
                eligible=int(eligible.sum()), pairs=pairs, variants=variants)


def summarize(rows,variant):
    total, depths, shapes = Counter(),Counter(),Counter()
    examples = []
    for row in rows:
        groups = row['variants'][variant]
        t = {tuple(pos) for g in groups for pos in g['target_xyz']}
        e = {tuple(pos) for g in groups for pos in g['extra_xyz']}
        assert sum(g['targets'] for g in groups)==len(t)
        total.update(components=len(groups),targets=len(t),distinct_extras=len(e),active_frames=int(bool(t)),
                     connected_extra_components=sum(g['extra_connected'] for g in groups))
        for g in groups:
            depths[str(g['signed_depth'])] += g['targets']
            shapes[str(g['targets'])] += 1
            examples.append(dict(id=row['id'],targets=g['targets'],extras=g['extras'],signed_depth=g['signed_depth'],
                reference_surface=g['reference_surface'],extent=g['target_xyz_extent']))
    return dict(**total,targets_by_extra_depth=depths,components_by_size=shapes,
                largest=sorted(examples,key=lambda x:-x['targets'])[:12])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',type=Path,default=REVIEW/'shared-face-round-01/panel-opportunity.json')
    args=parser.parse_args()
    base=REVIEW/'shared-face-round-01/error-budget.json'
    frames={r['id']:r for r in read(base)['frames']}
    tasks=[(j,frames[j[1]['id']]) for j in suite.jobs(suite.framed.EVALUATION)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(inspect,tasks))
    groups={k:summarize(rows,k) for k in rows[0]['variants']}
    pairs=[p for r in rows for p in r['pairs']]
    distinct=sum(len({tuple(p['extra']) for p in r['pairs']}) for r in rows)
    assert sum(r['eligible'] for r in rows)==507
    counts=dict(views=len(rows),eligible=507,majority_pairs=len(pairs),distinct_majority_extras=distinct,
        wall_pairs=sum(p['axis']!=1 for p in pairs),same_tangent_wrong_wall_depth=sum(
            p['axis']!=1 and p['signed_depth']!=0 and p['tangent_distance']==0 for p in pairs))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,dict(revision='abfc0be',error_budget_sha256=digest(base),counts=counts,
        definitions=dict(eligible='Required miss with majority coverage, majority right normal axis, and majority of covered pixels hitting extra blocks.',
            majority_pair='One exact reference unit face and one matching outward face of one extra cube must jointly own >50% of the target covered pixels.',
            panel='At least three face-connected target positions on the same recorded wall surface with a common signed extra depth. Tangent tolerances use Manhattan distance in blocks.',
            signed_depth='Extra coordinate minus missing reference coordinate dotted with outward surface normal. Positive means extra plane nearer the camera.',
            interpretation='Reference-only count of recognizable wrong panels, with each required target and each extra counted once per image per variant. Variants overlap. This does not establish which corrections source-image inference could select.'),
        variants=groups,frames=rows))
    print(counts)
    for name,group in groups.items():
        print(name, {k:v for k,v in group.items() if k!='largest'})
    print(groups['depth_only']['largest'][:6])


if __name__=='__main__':
    main()

