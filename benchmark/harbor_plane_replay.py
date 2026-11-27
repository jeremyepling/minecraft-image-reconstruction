"""Replay a primary carrier's level or depth partition, keeping ordinary downstream consumers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from benchmark.camera_horizon_replay import replay
from benchmark.harbor_carrier_audit import BASE, CASES, REVIEW, ROOT, SOURCE, build_dir, frames, native_face_planes, read, save
from engine import grounded_depth_envelope, visible_placement
from engine.visible_raster import SAMPLE_STRIDE, rasterize


def sampled_ids(labels, offset):
    """Reproject the visible reference faces at the placer's actual sample centers."""
    width, height = labels['camera']['image_size']
    shape = (height // SAMPLE_STRIDE, width // SAMPLE_STRIDE)
    depth = np.zeros(shape)
    ids = np.zeros(shape, int)
    P = np.array(labels['camera']['world_to_pixel_homogeneous'])
    for face in labels['faces']:
        world = np.asarray(face['corners_world'])
        homogeneous = world @ P[:, :3].T + P[:, 3]
        assert np.max(np.abs(homogeneous[:, :2]/homogeneous[:, 2:] - face['corners_px'])) < 1.e-4
        homogeneous[:, :2] = (homogeneous[:, :2] - offset * homogeneous[:, 2:]) / SAMPLE_STRIDE
        hit = rasterize(homogeneous, shape[1], shape[0])
        if hit:
            box, samples, distances, _, _ = hit
            closer = samples & (distances > depth[box])
            depth[box][closer] = distances[closer]
            ids[box][closer] = face['id']
    return ids


def run(job):
    entry, case, mode, tag = job
    key = entry['id']
    target = ROOT / 'runs/harbor-placement-evidence-01' / key / mode
    original = build_dir(entry, case)
    status = read(original / 'build-manifest.json')['placement']
    camera = dict(P=status['P'], scale_status=status['camera_status'], reason=status['reason'],
                  camera_source=status.get('camera_source'))
    frame = Path(entry['frame'])
    labels = read(frame / 'labels.json')
    sampled = sampled_ids(labels, 0.)
    translated = sampled_ids(labels, .5)
    with np.load(frame / 'labels.npz') as data:
        expected = data['face_id'][2::4, 2::4]
    differing = int((translated != expected).sum())
    if differing:
        raise ValueError(f'{key}: {differing} raster disagreements at native pixel centers')
    planes = native_face_planes(labels, np.asarray(case['registration']['transform']))
    plane_axis = np.array([-1] + [planes[i][0] for i in range(1, len(planes)+1)])[sampled]
    plane_level = np.array([0] + [planes[i][1] for i in range(1, len(planes)+1)])[sampled]
    estimate = visible_placement._estimate_plane_cells
    audit = []

    def primary(image, parts, masks, projection, eye, *args, **kwargs):
        if mode == 'rgb-oracle':
            expected_parts = read(BASE / entry['dataset'] / key / 'handoffs/placement-input.json.gz')['parts']
            if [(p['id'], p['support']) for p in parts] != [(p['id'], p['support']) for p in expected_parts]:
                raise ValueError(f'{key}: source carrier IDs or masks differ from audited placement inputs')
        levels = {}
        reconcile = kwargs.get('reconcile')

        def record(selections):
            selected = reconcile(selections) if reconcile else selections
            levels.update({parts[i]['id']: (axis, level) for i, axis, level in selected})
            return selected

        cells = estimate(image, parts, masks, projection, eye, *args, **(kwargs | {'reconcile': record}))
        for i, part in enumerate(parts):
            if part['id'] != tag or tag not in levels:
                continue
            axis, initial = levels[tag]
            support = masks[i]
            same = support & (plane_axis == axis)
            options, frequency = np.unique(plane_level[same], return_counts=True)
            dominant = int(options[np.argmax(frequency)])
            resolved = {}
            if mode == 'single':
                resolved[dominant] = support
            elif mode == 'split':
                for level in options:
                    resolved[int(level)] = same & (plane_level == level)
                residual = support & ~same
                resolved[initial] = resolved.get(initial, np.zeros_like(residual)) | residual
            elif mode == 'rgb-oracle':
                from benchmark.harbor_partition_probe import bounded_rgb_regions
                groups = bounded_rgb_regions(image, support)
                for group in np.unique(groups[support]):
                    region = support & (groups == group)
                    eligible = region & same
                    if group and eligible.any():
                        choices, votes = np.unique(plane_level[eligible], return_counts=True)
                        level = int(choices[votes.argmax()])
                    else:
                        level = initial
                    resolved[level] = resolved.get(level, np.zeros_like(region)) | region
            if resolved:
                new = set()
                for level, submask in resolved.items():
                    if submask.sum() >= 4:
                        def select(*_):
                            return level
                        candidate, = estimate(image, [part], [submask], projection, eye, select,
                                              partial=kwargs.get('partial'))
                        new.update(candidate)
            else:
                new = cells[i]
            audit.append(dict(tag=tag, ordinary=initial, axis=axis, dominant=dominant,
                              sublevels={int(k):int(v) for k,v in zip(options,frequency)},
                              before=len(cells[i]), after=len(new),
                              primary_added=len(new - cells[i]), primary_removed=len(cells[i] - new)))
            cells[i] = new
        return cells

    def bypass(image, parts, camera, prior, **kwargs):
        return prior, dict(removed=0, measured=False)

    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch.object(visible_placement, '_estimate_plane_cells', primary))
        if mode == 'bypass':
            stack.enter_context(patch.object(grounded_depth_envelope, 'recover', bypass))
        result = replay(entry, camera, BASE / entry['dataset'] / key, target)
    result.update(mode=mode, carrier=tag, handoff=audit, native_raster_disagreements=differing,
                  scored_reference_only=mode in ('single','split','rgb-oracle'))
    if not audit:
        raise ValueError(f'{key}: target carrier absent from actual placement')
    from benchmark.typed_structures import aligned_sites
    from benchmark.mask_blocks import decoded_pack
    T = np.asarray(case['registration']['transform'])
    before = set(aligned_sites(list(decoded_pack(original/'reconstruction.zip')), T, case['translation']))
    after = set(aligned_sites(list(decoded_pack(target/'reconstruction.zip')), T, result['translation']))
    truth = {tuple(b['xyz_world']) for b in labels['blocks']}
    result['paired'] = dict(gained=len((after-before)&truth),lost=len((before-after)&truth),
                            extras_added=len((after-before)-truth),extras_removed=len((before-after)-truth))
    if mode == 'control' and (decoded_pack(original/'reconstruction.zip') != decoded_pack(target/'reconstruction.zip')):
        raise ValueError(f'{key}: no-op differs from the released decoded pack')
    save(REVIEW / f'{key}-{mode}-score.json', result)
    print(key,mode,result['geometry']['strict'],result['scores']['material'],result['paired'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ids', nargs='+')
    p.add_argument('--modes', nargs='+', choices=['control','single','split','bypass','rgb-oracle'], required=True)
    p.add_argument('--workers', type=int, default=3)
    args=p.parse_args()
    cases={r['id']:r for r in read(CASES)}
    jobs=[]
    for e in frames(SOURCE,['visible-builds-v3']):
        if e['id'] in args.ids:
            tag=read(REVIEW/(e['id']+'.json'))['carriers'][0]['id']
            jobs.extend((e,cases[e['id']],m,tag) for m in args.modes)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))


if __name__=='__main__':
    main()
