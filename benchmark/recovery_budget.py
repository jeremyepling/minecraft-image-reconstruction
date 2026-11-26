"""Diagnose frozen relative geometry errors using reference visibility at scoring time."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark import visible_depth_native as native
from benchmark.camera_labels import exposed_faces, rasterize, screen_matrix
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry_replay import SIZES
from benchmark.texture_camera_eval import DATA, ROOT, STABLE, checked, digest, frames, read
from engine.surface_model import intrinsic
from engine.visible_raster import SAMPLE_STRIDE, observed_support
from training.visible_labels import visible_labels


SCORES = ROOT / 'benchmark/reviews/relative-build-01/scores.zip'
SPLITS = ('validation', 'test', 'regression', 'native', 'old', 'fresh')
STEPS = ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))


def adjacent_pair_count(misses, extras):
    """Maximum number of distinct extras that could be moved onto a miss by one face step."""
    neighbors = {p: [q for dx,dy,dz in STEPS if (q := (p[0]+dx,p[1]+dy,p[2]+dz)) in extras]
                 for p in sorted(misses)}
    matches = {}

    def augment(point, visited):
        for neighbor in neighbors[point]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            if neighbor not in matches or augment(matches[neighbor], visited):
                matches[neighbor] = point
                return True
        return False

    return sum(augment(point, set()) for point in neighbors)


def false_foreground(pose, extras, depth):
    """Pixels where a decoded extra cube lies strictly in front of visible reference geometry."""
    width,height = pose['image_size']
    eye = np.asarray(pose['eye_world'])
    matrix = screen_matrix(pose)
    obstructed = np.zeros(depth.shape, dtype=bool)
    for _,_,_,corners in exposed_faces(extras, eye):
        raster = rasterize((corners-eye) @ matrix[:,:3].T + matrix[:,3], width, height)
        if raster is not None:
            box, polygon, inverse_depth, _, _ = raster
            # The truth depth buffer is float32; ignore coplanar ties and tiny numerical differences.
            obstructed[box] |= polygon & (inverse_depth * depth[box] > 1.00001)
    return obstructed


def parts_and_labels(split, frame, data, runs, fixtures, score):
    key, hashes = frame['id'], frame['files']
    if split in ('old', 'fresh'):
        fixture = runs / ('visible-depth-rounds/transfer' if split == 'old' else 'canopy-placement/confirmation')
        path = fixture/'upstream'/key/'surfaces.json'
        parts = read(checked(path, score['surface_sha256']))['planes']
        camera = frame['camera']['P']
    else:
        path = runs/'visible-depth-rounds/inputs'/(key+'.json')
        source = read(checked(path, score['input_sha256']))
        assert source['rgb_sha256'] == hashes['rgb.png']
        parts, camera = source['parts'], source['camera']['P']
    if split in ('native','old','fresh'):
        stem = frame.get('source_id', key)
        folder = fixtures/frame['directory']/'captures'
        checked(folder/(stem+'.png'), hashes['rgb.png'])
        pose = read(checked(folder/(stem+'.json'), hashes['camera.json']))
        raw = read(checked(folder/(stem+'-blocks.json'), hashes['world-readback.json']))
        states = {tuple(b[k] for k in 'xyz'): b for b in raw}
        labels, ids, depth = visible_labels(pose, states, (0,0,0))
    else:
        folder = data/'data'/split/frame['directory']
        checked(folder/'rgb.png', hashes['rgb.png'])
        pose = read(checked(folder/'camera.json', hashes['camera.json']))
        labels = read(checked(folder/'labels.json', hashes['labels.json']))
        with np.load(checked(folder/'labels.npz', hashes['labels.npz'])) as raster:
            ids, depth = raster['face_id'], raster['depth_z']
    return camera, parts, pose, labels, ids, depth


def measure(job):
    split, frame, score, data, runs, fixtures = job
    key = frame['id']
    camera, parts, pose, labels, ids, depth = parts_and_labels(split, frame, data, runs, fixtures, score)
    truth = {tuple(b['xyz_world']) for b in labels['blocks']}
    frozen = score['methods']['current']
    assert frozen['offset'] == [0,0,0]
    folder = runs/'canopy-placement/release'/split/key
    archive = read(checked(folder/'cells.json', frozen['world_cells_sha256']))
    emitted = decoded_pack(checked(folder/'reconstruction.zip', frozen['pack_sha256']))
    assert set(emitted) == set(map(tuple, archive['local']))
    transform = registration(np.asarray(camera), pose) if camera is not None else None
    assert not emitted or transform is not None
    world = registered(emitted, transform)
    assert world == set(map(tuple, archive['world']))
    misses, extras = truth-world, world-truth
    correct = len(world & truth)
    expected = frozen['aligned']
    counts = Counter(target=len(truth), correct=correct, extras=len(extras), missed=len(misses))
    assert all(expected[k] == v for k,v in counts.items())
    row = dict(split=split, id=key, layout=frame.get('layout', frame.get('shape')),
               camera_supported=camera is not None, counts=counts)
    if camera is None:
        counts['camera_missing'] = len(misses)
        return row
    counts['one_step_pairs'] = adjacent_pair_count(misses, extras)

    block_count = len(labels['blocks'])+1
    stride = SAMPLE_STRIDE
    sampled = ids[stride//2::stride, stride//2::stride]
    size = pose['image_size']
    normalized = np.asarray(camera)/np.linalg.norm(np.asarray(camera)[2,:3])
    focal = np.linalg.norm(normalized[0,:3]-.5*size[0]*normalized[2,:3])
    rotation = np.linalg.solve(intrinsic(size, focal), normalized[:,:3])
    axes = np.zeros((3,*sampled.shape), dtype=bool)
    any_mask = np.zeros(sampled.shape, dtype=bool)
    native_axes = np.zeros((3,*ids.shape), dtype=bool)
    for part in parts:
        if part.get('support'):
            mask = observed_support(part['support'], sampled.shape, size)
            any_mask |= mask
            if part.get('normal_camera') is not None:
                world_normal = transform[:3,:3] @ rotation.T @ part['normal_camera']
                normal_axis = np.argmax(abs(world_normal))
                axes[normal_axis] |= mask
                support = part['support']
                sw,sh = support['size']
                raw = np.zeros((sh,sw), dtype=bool)
                for y,x0,x1 in support['runs']:
                    raw[y,x0:x1] = True
                # Here the sample is the native pixel center, including the half-pixel offset.
                u = ((np.arange(size[0])+.5)*sw/size[0]).astype(int)
                v = ((np.arange(size[1])+.5)*sh/size[1]).astype(int)
                native_axes[normal_axis] |= raw[np.ix_(v,u)]

    owner = np.zeros(len(labels['faces'])+1, dtype=int)
    axis = np.zeros_like(owner)
    native_pixels = np.zeros(block_count, dtype=int)
    direction_pixels = Counter()
    for face in labels['faces']:
        index, block = face['id'], face['block']
        owner[index] = block
        axis[index] = np.argmax(abs(np.asarray(face['normal_world'])))
        native_pixels[block] += face['visible_pixels']
        direction_pixels[block,face['direction']] += face['visible_pixels']
    points = owner[sampled]
    same_axis = np.take_along_axis(axes, axis[sampled][None], axis=0)[0]
    samples = np.bincount(points.ravel(), minlength=block_count)
    covered = np.bincount(points[any_mask], minlength=block_count)
    oriented = np.bincount(points[same_axis], minlength=block_count)
    native_oriented = np.take_along_axis(native_axes, axis[ids][None], axis=0)[0]
    dense = np.bincount(owner[ids[native_oriented]], minlength=block_count)
    half = np.bincount(owner[ids[::2,::2][native_oriented[::2,::2]]], minlength=block_count)
    occluded = false_foreground(pose, extras, depth)
    blocked = np.bincount(owner[ids[occluded]], minlength=block_count)
    for block in labels['blocks']:
        if tuple(block['xyz_world']) not in misses:
            continue
        i = block['id']
        if samples[i] == 0:
            bucket = 'unsampled'
        elif covered[i] == 0:
            bucket = 'outside_mask'
        elif oriented[i] == 0:
            bucket = 'wrong_axis_only'
        elif oriented[i] < 4 or 2*oriented[i] < samples[i]:
            bucket = 'weak_axis_support'
        else:
            bucket = 'strong_axis_support'
        counts[bucket] += 1
        if bucket == 'unsampled':
            counts['unsampled_with_stride2_axis_hit'] += int(half[i] > 0)
            counts['unsampled_with_native_axis_hit'] += int(dense[i] > 0)
            counts['unsampled_with_16_native_axis_pixels'] += int(dense[i] >= 16)
            counts['unsampled_with_16_visible_pixels'] += int(native_pixels[i] >= 16)
            directions = ['up','down','west','east','north','south']
            dominant = max(directions, key=lambda direction: direction_pixels[i,direction])
            counts['unsampled_dominant_'+(dominant if dominant in ('up','down') else 'side')] += 1
        counts['sub16_native_pixels'] += int(native_pixels[i] < 16)
        counts['majority_blocked_by_extra'] += int(2*blocked[i] >= native_pixels[i])
        counts['fully_blocked_by_extra'] += int(blocked[i] == native_pixels[i])
        if bucket == 'strong_axis_support':
            counts['strong_and_majority_blocked'] += int(2*blocked[i] >= native_pixels[i])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=STABLE/'runs')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--fixtures', type=Path, default=native.BUILDS)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(SCORES) as z:
        baseline = json.loads(z.read('scores.json'))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in SPLITS:
            if split in ('old','fresh'):
                fixture = 'visible-depth-rounds/transfer' if split=='old' else 'canopy-placement/confirmation'
                selected = read(args.runs/fixture/'screen.json')['selected']
            else:
                selected = native.rows() if split=='native' else frames(args.data, split)
            scores = {r['id']:r for r in baseline['cases'] if r['split']==split}
            assert len(selected) == len(scores) == SIZES[split]
            jobs = [(split, f, scores[f['id']], args.data, args.runs, args.fixtures) for f in selected]
            result = list(pool.map(measure, jobs))
            rows.extend(result)
            print(split, sum((r['counts'] for r in result), Counter()), flush=True)
    totals = sum((r['counts'] for r in rows), Counter())
    assert totals['missed'] == sum(totals[k] for k in ('camera_missing','unsampled','outside_mask',
                                    'wrong_axis_only','weak_axis_support','strong_axis_support'))
    output = dict(baseline_scores_sha256=digest(SCORES), totals=totals, cases=rows)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(output, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
