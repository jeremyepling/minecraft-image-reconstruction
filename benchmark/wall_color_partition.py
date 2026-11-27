"""Source-only child-depth trial for a color-connected wall carrier.

This benchmark changes provisional cells only. Reference data is used after export
by the existing scorer, never to choose the carrier, partition, or plane levels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, frames, read, save
from benchmark.camera_horizon_replay import replay
from engine import visible_placement
from engine.atlas_plane_depth import options
from engine.camera_atlas_floor import _atlas
from engine.plane_repeat_evidence import frozen_repeat_weights, repeat_scores, select_repeat
from engine.visible_raster import SAMPLE_STRIDE

BASE = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01'
CASES = ROOT / 'benchmark/reviews/camera-model-controls-01/independent-matched-cases.json.gz'
REVIEW = ROOT / 'benchmark/reviews/wall-color-partition-01'


def bounded_rgb_regions(image, mask, tolerance=20, minimum=16):
    """Frozen training rule from the Harbor diagnostic; no reference-dependent cuts."""
    h, w = mask.shape
    if image.width < w*4 or image.height < h*4:
        return np.zeros(mask.shape, int)
    rgb = np.asarray(image)[:h*4, :w*4].reshape(h, 4, w, 4, 3)
    rgb = np.median(rgb, axis=(1, 3))
    horizontal = np.abs(rgb[:, 1:] - rgb[:, :-1]).max(axis=2) <= tolerance
    vertical = np.abs(rgb[1:] - rgb[:-1]).max(axis=2) <= tolerance
    remaining = mask.copy()
    result = np.zeros(mask.shape, int)
    group = 0
    for y, x in zip(*np.nonzero(mask)):
        if not remaining[y, x]:
            continue
        pending, region = [(y, x)], []
        remaining[y, x] = False
        while pending:
            yy, xx = pending.pop()
            region.append((yy, xx))
            for dy, dx, connected in (
                (0, -1, xx > 0 and horizontal[yy, xx-1]),
                (0, 1, xx+1 < w and horizontal[yy, xx]),
                (-1, 0, yy > 0 and vertical[yy-1, xx]),
                (1, 0, yy+1 < h and vertical[yy, xx]),
            ):
                j, i = yy+dy, xx+dx
                if connected and remaining[j, i]:
                    pending.append((j, i))
                    remaining[j, i] = False
        if len(region) >= minimum:
            group += 1
            rows, cols = zip(*region)
            result[rows, cols] = group
    return result


def baseline_folder(entry, case):
    key, dataset = entry['id'], entry['dataset']
    roots = [WORKTREES / 'camera-geocalib/runs/camera-geocalib-01/matched-control/builds' / dataset / key,
             WORKTREES / 'camera-horizon-floor/runs/camera-horizon-floor-01' / key / 'repeat',
             BASE / dataset / key]
    for folder in roots:
        pack = folder / 'reconstruction.zip'
        if pack.exists() and digest(pack) == case['pack_sha256']:
            return folder
    raise ValueError(f'No hash-matched baseline pack for {key}')


def install_trial(mode, audit):
    estimate = visible_placement._estimate_plane_cells
    atlas = _atlas('26.3', None)
    weights = frozen_repeat_weights()

    def partition(image, parts, masks, projection, eye, *args, **kwargs):
        selected = {}
        reconcile = kwargs.get('reconcile')

        def record(selections):
            resolved = reconcile(selections) if reconcile else selections
            selected.update({i: (axis, level) for i, axis, level in resolved})
            return resolved

        cells = estimate(image, parts, masks, projection, eye, *args,
                         **(kwargs | {'reconcile': record}))
        vertical = [i for i, (axis, _) in selected.items() if axis != 1]
        if not vertical or weights is None or atlas is None:
            return cells
        # Image-supported carrier selection; never reference coverage or frame ID.
        index = max(vertical, key=lambda i: int(masks[i].sum()))
        mask = masks[index]
        axis, initial = selected[index]
        groups = bounded_rgb_regions(image, mask)
        if groups.max() < 2:
            return cells
        yy, xx = np.indices(mask.shape)
        pixels = np.stack(((xx+.5)*SAMPLE_STRIDE, (yy+.5)*SAMPLE_STRIDE, np.ones(mask.shape)), axis=-1)
        rays = pixels @ np.linalg.inv(projection[:, :3]).T
        rgb = np.asarray(image.convert('RGB'))
        resolved, decisions = {}, []
        for group in np.unique(groups[mask]):
            region = mask & (groups == group)
            level, rows = initial, []
            if group and mode != 'control':
                rows = options(region, projection, eye, rays, axis, initial, rgb, atlas, radius=6)
                repeats = repeat_scores(rgb, region, projection, eye, axis, [r['level'] for r in rows])
                for row, (hi, lo) in zip(rows, repeats, strict=True):
                    row.update(repeat_high=hi, repeat_low=lo)
                if mode == 'ranked':
                    level = select_repeat(rows, initial, weights)
                elif mode == 'boundary':
                    incumbent = next((r for r in rows if r['level'] == initial), None)
                    winner = max(rows, key=lambda r: r['boundary'], default=None)
                    if (winner and incumbent and winner['boundary'] >= .60
                            and winner['boundary']-incumbent['boundary'] >= .15):
                        level = winner['level']
            resolved[level] = resolved.get(level, np.zeros_like(mask)) | region
            decisions.append(dict(group=int(group), samples=int(region.sum()), level=int(level), options=rows))
        before = cells[index]
        if any(level != initial for level in resolved):
            replacement = set()
            for level, region in resolved.items():
                candidate, = estimate(image, [parts[index]], [region], projection, eye,
                                      lambda *_: level, partial=kwargs.get('partial'))
                replacement.update(candidate)
            cells[index] = replacement
        audit.append(dict(id=parts[index]['id'], axis=int(axis), initial=int(initial),
                          added=len(cells[index]-before), removed=len(before-cells[index]), children=decisions))
        return cells
    return patch.object(visible_placement, '_estimate_plane_cells', partition)


def run(job):
    entry, case, mode = job
    original = baseline_folder(entry, case)
    status = read(original / 'build-manifest.json')['placement']
    camera = dict(P=status['P'], scale_status=status['camera_status'], reason=status['reason'],
                  camera_source=status.get('camera_source'))
    if camera['P'] is None:
        raise ValueError('This downstream trial requires a supported source-only camera')
    target = ROOT / 'runs/wall-color-partition-01' / entry['id'] / mode
    audit = []
    with install_trial(mode, audit):
        result = replay(entry, camera, BASE / entry['dataset'] / entry['id'], target)
    from benchmark.mask_blocks import decoded_pack
    from benchmark.typed_structures import aligned_sites
    before_raw, after_raw = decoded_pack(original/'reconstruction.zip'), decoded_pack(target/'reconstruction.zip')
    if mode == 'control' and before_raw != after_raw:
        raise ValueError('No-op export differs from verified default')
    transform = np.array(case['registration']['transform'])
    before = set(aligned_sites(list(before_raw), transform, case['translation']))
    after = set(aligned_sites(list(after_raw), transform, result['translation']))
    labels = read(Path(entry['frame']) / 'labels.json')
    truth = {tuple(b['xyz_world']) for b in labels['blocks']}
    result.update(mode=mode, source_only=True, baseline_revision='62611ee', baseline=case,
                  handoff=audit, paired=dict(gained=len((after-before)&truth), lost=len((before-after)&truth),
                  extras_added=len((after-before)-truth), extras_removed=len((before-after)-truth)))
    save(REVIEW / f"{entry['id']}-{mode}.json", result)
    print(entry['id'], mode, result['geometry']['strict'], result['paired'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ids', nargs='+')
    parser.add_argument('--modes', nargs='+', choices=['control', 'ranked', 'boundary'], required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    cases = {c['id']: c for c in read(CASES)}
    jobs = [(e, cases[e['id']], m) for e in frames(SOURCE, ['visible-builds-v3', 'decorative-builds-v1'])
            if e['id'] in args.ids for m in args.modes]
    if len(jobs) != len(args.ids)*len(args.modes):
        raise ValueError('One or more requested frames is missing')
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run, jobs))


if __name__ == '__main__':
    main()
