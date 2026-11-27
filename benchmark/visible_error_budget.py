"""Classify every missed visible block and extra in the frozen complete-structure evaluation."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.camera_metrics import registration
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE
from engine.surface_perimeter import _native_alpha
from engine.surface_model import intrinsic


def tally(identifiers, values, size):
    return np.bincount(identifiers, weights=values, minlength=size).astype(int)


def stage_evidence(parts, counts, size, camera, world_axis, reference_axis):
    """Best measured chart covering a reference pixel with the correct face axis."""
    best = np.zeros(size[::-1], np.uint8)
    p = np.asarray(camera['P'],float)
    p = p / np.linalg.norm(p[2,:3])
    focal = np.linalg.norm(p[0,:3] - size[0]*.5*p[2,:3])
    rotation = np.linalg.solve(intrinsic(size,focal),p[:,:3])
    by_id = {record['id']:record for record in counts}
    for part in parts:
        alpha = _native_alpha(part['support'], size)
        if alpha is not None:
            mask, (l,t,r,b) = alpha
            inside = np.asarray(mask, bool)
            cut = best[t:b,l:r]
            cut[inside] = np.maximum(cut[inside],1)
            normal = part.get('normal_camera')
            if normal is not None:
                axis = world_axis[np.argmax(np.abs(rotation.T @ normal))]
                correct_axis = inside & (reference_axis[t:b,l:r] == axis)
                record = by_id[part['id']]
                measured = all(x is not None for x in record['dimensions']['blocks'])
                phase = all(x is not None for x in record['phase'])
                rank = 2 + int(measured) + int(measured and phase)
                cut[correct_axis] = np.maximum(cut[correct_axis], rank)
    return best


def projection(camera, pose, local, shift, shape):
    p = np.array(camera['P'], float)
    p /= np.linalg.norm(p[2, :3])
    eye = -np.linalg.solve(p[:, :3], p[:, 3])
    t = registration(p, pose)
    p[:2] *= SAMPLE_STRIDE
    owner, axis = _frontmost_cells(local, p, eye, shape, face_axes=True)
    world = np.floor((np.array(local, int).reshape(-1, 3) + .5) @ t[:3,:3].T + t[:3,3]).astype(int)
    return owner, axis, np.argmax(abs(t[:3,:3]), axis=0), world + shift


def audit(job):
    split, frame, prior, row, run, layout = job
    key = frame['id']
    image, parts, counts, camera, pose, labels, ids, truth, _, placed, _ = read_view(run, split, frame, prior, row)
    assert row['rgb_sha256'] == prior['rgb_sha256'] == frame['files']['rgb.png']
    metrics = row['relative_geometry']
    missing = truth - placed
    extras = placed - truth
    blocks = {b['id']:tuple(b['xyz_world']) for b in labels['blocks']}
    count = len(blocks) + 1
    nfaces = len(labels['faces']) + 1
    block_lut = np.zeros(nfaces, int)
    axis_lut = np.zeros(nfaces, int)
    normals = np.zeros((nfaces,3), int)
    evidence = defaultdict(set)
    for face in labels['faces']:
        i, b = face['id'], face['block']
        n = face['normal_world']
        block_lut[i], axis_lut[i], normals[i] = b, np.argmax(np.abs(n)), n
        assert face['visible_pixels'] > 0
        evidence[b].add('top' if n[1]>0 else 'underside' if n[1]<0 else 'wall')
    source = ids.ravel()
    pixel_blocks = block_lut[source]
    footprint = np.bincount(pixel_blocks, minlength=count)
    by_face = defaultdict(Counter)
    by_area = Counter()
    same_axis_plane = Counter()
    uncovered_surface = Counter()
    extra_context = Counter()
    per_face_area = defaultdict(Counter)
    uncovered_stages = defaultdict(Counter)
    placed_stages = Counter()
    covering_cells = defaultdict(Counter)
    camera_ok = camera.get('P') is not None
    if camera_ok:
        cells = sorted(map(tuple, read(run/split/key/'cells.json')['local']))
        owner, axis, world_axis, world = projection(camera,pose,cells,row['translation'],ids.shape)
        assert set(map(tuple,world)) == placed
        evidence_rank = stage_evidence(parts,counts,image.size,camera,world_axis,axis_lut[ids]).ravel()
        stage_pixels = np.array([tally(pixel_blocks,evidence_rank>=k,count) for k in range(1,5)])
        owner, axis = owner.ravel(), axis.ravel()
        shared = (owner>=0) & (source>0)
        covered = tally(pixel_blocks, shared, count)
        belongs_to_build = np.array([tuple(c) in truth for c in world],bool)
        owned_by_true = np.zeros(source.shape,bool)
        owned_by_true[shared] = belongs_to_build[owner[shared]]
        covered_true = tally(pixel_blocks,owned_by_true,count)
        matching = shared & (world_axis[axis]==axis_lut[source])
        aligned = tally(pixel_blocks, matching, count)
        # Compare the grid-plane coordinates along the recorded visible normal.
        # The predicted index may differ along the tangent even on the same plane.
        d = np.flatnonzero(matching)
        target_coords = np.zeros((count,3),int)
        for b, coord in blocks.items():
            target_coords[b] = coord
        delta = ((world[owner[d]] - target_coords[pixel_blocks[d]]) * normals[source[d]]).sum(1)
        plane_votes = np.column_stack([tally(pixel_blocks[d], yes, count) for yes in (delta>0,delta<0,delta==0)])
        inside_by_owner = tally(owner[shared], np.ones(shared.sum(),bool), len(cells))
        owner_total = np.bincount(owner[owner>=0], minlength=len(cells))
        for index, coord in enumerate(world):
            if tuple(coord) in extras:
                covered_n, total = inside_by_owner[index], owner_total[index]
                kind = ('no_first_hit' if total==0 else 'on_build' if covered_n*2>total
                        else 'on_background' if covered_n*2<total else 'half_on_build')
                extra_context[kind] += 1
    else:
        assert not placed
    for b, coord in blocks.items():
        n = int(footprint[b])
        stages = ('no_mask','mask_only','right_axis','dimensions','grid')
        stage_name = stages[int(np.count_nonzero(stage_pixels[:,b]*2>=n))] if camera_ok else 'no_camera'
        if coord not in missing:
            placed_stages[stage_name] += 1
            continue
        kind = '+'.join(sorted(evidence[b]))
        assert n>0
        area = '1_to_15' if n<16 else '16_to_63' if n<64 else '64_or_more'
        by_area[area] += 1
        per_face_area[kind][area] += 1
        if not camera_ok:
            branch = 'no_camera'
        elif covered[b]*2<n:
            branch = 'uncovered'
            uncovered_stages[kind][stage_name] += 1
            uncovered_surface['mask_covers_less' if stage_name=='no_mask' else 'mask_covers_half'] += 1
        elif covered[b]*2==n:
            branch = 'coverage_tie'
        elif aligned[b]*2>covered[b]:
            branch = 'same_axis'
            votes = plane_votes[b]
            # Count a signed depth outcome only when most same-axis pixels agree.
            if votes.max()*2<=votes.sum():
                same_axis_plane['mixed'] += 1
            else:
                same_axis_plane[('front','behind','same_plane')[int(votes.argmax())]] += 1
        elif aligned[b]*2<covered[b]:
            branch = 'different_axis'
        else:
            branch = 'axis_tie'
        by_face[kind][branch] += 1
        if branch in ('same_axis','different_axis','axis_tie'):
            value = ('already_correct' if covered_true[b]*2>covered[b] else
                     'extra' if covered_true[b]*2<covered[b] else 'tie')
            covering_cells[branch][value] += 1
    assert sum(sum(x.values()) for x in by_face.values()) == metrics['missed']
    assert sum(extra_context.values()) == metrics['extras']
    return dict(id=key,split=split,layout=layout,family=row['family'],camera=camera_ok,
                pack_sha256=row['pack_sha256'],rgb_sha256=row['rgb_sha256'],translation=row['translation'],
                correct=metrics['correct'],missed=metrics['missed'],extras=metrics['extras'],
                hidden_extras=metrics['hidden_extras'],face_by_projection=by_face,miss_area=by_area,
                face_by_area=per_face_area,uncovered_stages=uncovered_stages,
                covering_cells=covering_cells,placed_stages=placed_stages,
                same_axis_depth=same_axis_plane,uncovered_surface=uncovered_surface,
                extra_projection=extra_context)


def aggregate(rows):
    result = {field:sum(r[field] for r in rows) for field in ('correct','missed','extras','hidden_extras')}
    result['views'] = len(rows)
    result['failed_camera_views'] = sum(not r['camera'] for r in rows)
    for key in ('face_by_projection','face_by_area','uncovered_stages','covering_cells'):
        counts = defaultdict(Counter)
        for r in rows:
            for kind, value in r[key].items():
                counts[kind].update(value)
        result[key] = counts
    for key in ('same_axis_depth','miss_area','uncovered_surface','extra_projection','placed_stages'):
        counts = Counter()
        for r in rows:
            counts.update(r[key])
        result[key] = counts
    return result


def run(args):
    previous = read(args.run/'summary.json')
    manifest = checked(args.run/'cases.json',previous['cases_sha256'])
    index = {r['id']:r for r in read(manifest)}
    rank = framed.REVIEW/'facade-placement-01/worst/ranking.json'
    layouts = {key:r['name'] for r in read(rank)['all_layouts'] for key in r['frame_ids']}
    jobs = [(s,f,p,index[f['id']],args.run,layouts[f['id']]) for s,f,p in framed.cases(framed.EVALUATION)]
    assert len(jobs)==len({f['id'] for _,f,*_ in jobs})==104
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(audit,jobs))
    totals = aggregate(rows)
    assert all(totals[k]==previous['groups']['fixed'][k] for k in ('correct','missed','extras','hidden_extras'))
    families = sorted({r['family'] for r in rows})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,dict(control_revision=previous['control'],method=previous['candidate'],revision=previous['revision'],
                       control_cases_sha256=digest(manifest),ranking_sha256=digest(rank),
                       totals=totals,families={family:aggregate([r for r in rows if r['family']==family])
                       for family in families},frames=rows))
    print(totals)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('../wall-section-round/runs/control'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/visible-error-budget-01/baseline.json'))
    parser.add_argument('--workers',type=int,default=4)
    run(parser.parse_args())
