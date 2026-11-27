"""Test RGB-supported additions in the narrow unclaimed strip beneath an open skyline."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from engine.pillar_seams import native_mask
from engine.sky_filter import sky_masks
from engine import skyline_extent


CONTROL = Path('../visible-error-budget/runs/primary-production')


def proposals(image, parts):
    """Return newly supported pixels for a single, unambiguous photographed edge."""
    h, w = image.height, image.width
    if not parts or min(h, w) < 16:
        return []
    masks = [native_mask(part, image.size) for part in parts]
    occupied = np.logical_or.reduce(masks)
    x = np.arange(w)
    first = occupied.argmax(axis=0)
    safe = occupied.any(axis=0) & (first >= 7) & (first + 3 < h)
    y = first.clip(7, h - 4)
    rgb = np.asarray(image.convert('RGB'), dtype=np.int16)
    sky, cloud = sky_masks(image)
    sky |= cloud
    # Only the first object below a clear line of sight up to the image edge.
    top = (~sky).argmax(axis=0)
    distance = y - top
    safe &= sky[0] & (distance >= 1) & (distance <= 4)
    safe &= np.max(abs(rgb[top, x] - rgb[(top - 1).clip(0), x]), axis=1) >= 28
    owners = np.sum([mask[y, x] for mask in masks], axis=0)
    supported = []
    for part, mask in zip(parts, masks):
        cols = safe & (owners == 1) & mask[y, x] & mask[y + 1, x] & mask[y + 2, x]
        for step in range(1, 5):
            c = rgb[y - step, x]
            near = np.maximum(abs(c - rgb[y, x]), abs(c - rgb[y + 1, x]))
            cols &= (step > distance) | (near.max(axis=1) <= 4)
        xs = np.flatnonzero(cols)
        if not len(xs):
            continue
        rows, columns = [], []
        for group in np.split(xs, np.flatnonzero(np.diff(xs) != 1) + 1):
            if len(group) < 12 or np.max(abs(np.diff(top[group]))) > 1:
                continue
            for step in range(1, 5):
                selected = group[distance[group] >= step]
                rows.extend((y[selected] - step).tolist())
                columns.extend(selected.tolist())
        if rows:
            supported.append((part, mask, np.asarray(rows), np.asarray(columns)))
    return supported


def inspect(job):
    split, frame, prior, row, below_rim = job
    if split in ('extra', 'new', 'shelter_gate'):
        from PIL import Image
        from training.visible_labels import visible_labels
        if split == 'shelter_gate':
            root = Path('../framed-depth-transfer/runs/transfer-input')
            stage = root/'upstream'/frame['id']/'surfaces.json'
            parts = read(checked(stage, row['surface_sha256']))['planes']
            source = root/frame['directory']/'captures'/(frame['source_id'] + '.png')
            with Image.open(checked(source, frame['files']['rgb.png'])) as src:
                image = src.convert('RGB')
        else:
            image, parts, _, _ = framed.photo(split, frame, prior)
            root = framed.NEW_INPUTS if split == 'new' else framed.transfer.BUILDS
        path = root / frame['directory'] / 'captures' / frame['source_id']
        pose = read(checked(path.with_suffix('.json'), frame['files']['camera.json']))
        states = read(checked(path.with_name(path.name + '-blocks.json'), frame['files']['world-readback.json']))
        labels, ids, _ = visible_labels(pose, {tuple(b[k] for k in 'xyz'): b for b in states}, (0, 0, 0))
    else:
        image, parts, _, _, _, labels, ids, *_ = read_view(CONTROL, split, frame, prior, row)
    lookup = np.zeros(len(labels['faces']) + 1, dtype=np.int32)
    for face in labels['faces']:
        lookup[face['id']] = face['surface_id']
    surfaces = lookup[ids]
    changes = []
    if below_rim:
        blue, cloud = sky_masks(image)
        recovered, _ = skyline_extent.refine(image, parts, blue | cloud)
        additions = []
        for part, after in zip(parts, recovered):
            if part is after:
                continue
            mask = native_mask(part, image.size)
            new = native_mask(after, image.size)
            assert np.all(new[mask])
            ys, xs = np.nonzero(new & ~mask)
            additions.append((part, mask, ys, xs))
    else:
        additions = proposals(image, parts)
    for part, mask, ys, xs in additions:
        overlap = np.bincount(surfaces[mask], minlength=int(lookup.max()) + 1)
        dominant = int(np.argmax(overlap[1:])) + 1
        added = surfaces[ys, xs]
        changes.append(dict(surface=part['id'], reference=dominant, right=int((added == dominant).sum()),
                            background=int((added == 0).sum()), other=int(((added != 0) & (added != dominant)).sum())))
    return dict(id=frame['id'], split=split, rgb_sha256=row['rgb_sha256'], changes=changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', default=['train', *framed.EVALUATION])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--below-rim', action='store_true')
    args = parser.parse_args()
    info = read(CONTROL / 'summary.json')
    cases = {r['id']: r for r in read(checked(CONTROL / 'cases.json', info['cases_sha256']))}
    jobs = [(s, f, h, cases[f['id']], args.below_rim)
            for s, f, h in framed.cases([s for s in args.splits if s != 'shelter_gate'])]
    if 'shelter_gate' in args.splits:
        import json
        from zipfile import ZipFile
        review = framed.REVIEW/'visible-error-budget-01'
        with ZipFile(checked(review/'comparisons.zip', read(review/'results.json')['archive_sha256'])) as packs:
            transfer = {r['id']:r for r in json.loads(packs.read('production/transfer/cases.json'))}
        root = Path('../framed-depth-transfer/runs/transfer-input')
        jobs.extend(('shelter_gate', f, None, transfer[f['id']], args.below_rim)
                    for f in read(root/'screen.json')['selected'])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect, jobs))
    totals = Counter()
    for row in rows:
        totals['changed_views'] += int(bool(row['changes']))
        for changed in row['changes']:
            for key in ('right', 'background', 'other'):
                totals[key] += changed[key]
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / 'cases.json', rows)
    save(args.out / 'summary.json', dict(frames=len(rows), counts=dict(totals),
                                         control_cases_sha256=info['cases_sha256'],
                                         cases_sha256=digest(args.out / 'cases.json')))
    print(totals)


if __name__ == '__main__':
    main()
