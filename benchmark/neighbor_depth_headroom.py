"""Answer-assisted depth opportunity on exactly the native pixels of residual misses."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from benchmark import recovery_budget as budget
from benchmark.camera_labels import exposed_faces, rasterize, screen_matrix
from benchmark.texture_camera_eval import checked, digest, read


def predicted_front(world, pose):
    width, height = pose['image_size']
    eye = np.asarray(pose['eye_world'])
    projection = screen_matrix(pose)
    depth = np.zeros((height, width), float)
    owner = np.zeros((height, width), np.int32)
    faces = [None]
    for cell, normal, direction, corners in exposed_faces(world, eye):
        hit = rasterize((corners-eye) @ projection[:, :3].T + projection[:,3], width, height)
        if hit is None:
            continue
        box, polygon, invdepth, *_ = hit
        closer = polygon & (invdepth > depth[box])
        if not closer.any():
            continue
        depth[box][closer] = invdepth[closer]
        owner[box][closer] = len(faces)
        faces.append((cell, direction))
    return owner, faces


def max_moves(edges):
    used = {}
    def augment(point, seen):
        for extra in edges[point]:
            if extra in seen:
                continue
            seen.add(extra)
            if extra not in used or augment(used[extra], seen):
                used[extra] = point
                return True
        return False
    return sum(augment(point,set()) for point in edges)


def projected_depth_pairs(world, misses, extras, pose, labels, ids, oriented):
    predicted, refs = predicted_front(world, pose)
    blocks = {b['id']:tuple(b['xyz_world']) for b in labels['blocks']}
    missing_face = np.zeros(len(labels['faces'])+1, bool)
    faces = {}
    for face in labels['faces']:
        if blocks[face['block']] in misses:
            missing_face[face['id']] = True
            faces[face['id']] = face
    valid = missing_face[ids] & (predicted != 0)
    # Count shared native pixels for each reference-face/predicted-face pair.
    pair = ids.astype(np.int64)*len(refs) + predicted
    counts = Counter()
    for apply_mask in (False, True):
        pixels = pair[valid & oriented] if apply_mask else pair[valid]
        codes, areas = np.unique(pixels, return_counts=True)
        prefix = 'masked_' if apply_mask else ''
        seen = defaultdict(set)
        edges = defaultdict(lambda: defaultdict(set))
        for code, n in zip(codes, areas):
            true_id, placed_id = divmod(int(code), len(refs))
            face = faces[true_id]
            if n < max(4, .25*face['visible_pixels']):
                continue
            target = blocks[face['block']]
            source, direction = refs[placed_id]
            seen['overlap_any_axis'].add(target)
            if direction != face['direction']:
                continue
            axis = int(np.argmax(np.abs(face['normal_world'])))
            diff = abs(source[axis]-target[axis])
            seen['overlap_same_direction'].add(target)
            if diff == 0:
                seen['same_plane'].add(target)
                continue
            same_column = all(source[i]==target[i] for i in range(3) if i!=axis)
            for radius in (1, 2, 4, 8):
                if diff > radius:
                    continue
                label = f'within_{radius}'
                seen[label].add(target)
                if same_column:
                    seen['normal_only_'+label].add(target)
                if source in extras:
                    edges[label][target].add(source)
                    if same_column:
                        edges['normal_only_'+label][target].add(source)
        for label, points in seen.items():
            counts[prefix+label] = len(points)
        for label, options in edges.items():
            counts[prefix+'extra_'+label] = len(options)
            counts[prefix+'distinct_relocations_'+label] = max_moves(options)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--runs',type=Path,default=budget.STABLE/'runs')
    parser.add_argument('--data',type=Path,default=budget.DATA)
    parser.add_argument('--fixtures',type=Path,default=budget.native.BUILDS)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--splits',nargs='+',choices=budget.SPLITS,default=budget.SPLITS)
    args=parser.parse_args()
    result, provenance = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in args.splits:
            folder = args.predictions/split
            manifest = read(folder/'manifest.json')
            cases = read(checked(folder/'cases.json',manifest['cases_sha256']))
            scored = {r['id']:r for r in cases}
            if split in ('old','fresh'):
                fixture = 'visible-depth-rounds/transfer' if split=='old' else 'canopy-placement/confirmation'
                frames = read(args.runs/fixture/'screen.json')['selected']
            else:
                frames = budget.native.rows() if split=='native' else budget.frames(args.data,split)
            assert set(scored)=={f['id'] for f in frames} and len(frames)==budget.SIZES[split]
            jobs = []
            for frame in frames:
                row = scored[frame['id']]
                current = dict(offset=row['translation'], aligned=row['relative_geometry'],
                               pack_sha256=row['pack_sha256'],world_cells_sha256=digest(folder/frame['id']/'cells.json'))
                score = dict(row,methods={'current':current})
                jobs.append((split,frame,score,args.data,args.runs,args.fixtures,args.predictions))
            result.extend(pool.map(budget.measure,jobs))
            provenance[split] = dict(manifest_sha256=digest(folder/'manifest.json'), cases_sha256=manifest['cases_sha256'],
                                     nonzero_origin_offsets=sum(any(r['translation']) for r in cases))
            print(split, len(jobs), flush=True)
    totals = sum((r['counts'] for r in result),Counter())
    assert totals['missed']==sum(totals[k] for k in ('camera_missing','unsampled','outside_mask',
                             'wrong_axis_only','weak_axis_support','strong_axis_support'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(provenance=provenance,totals=totals,cases=result),indent=2)+'\n')
    print('correct, missed, extra',*(totals[k] for k in ('correct','missed','extras')))


if __name__=='__main__':
    main()
