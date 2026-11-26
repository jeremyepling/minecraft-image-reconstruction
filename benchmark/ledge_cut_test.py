"""Test a depth vote corroborated by a finite screenshot edge ending at a corner or T.

python -m benchmark.ledge_cut_test --source-root . --out runs/ledge-cuts-test
Only predictions, numerical model fields and the original RGB reach predict().
Each cut partitions a still-intact baseline parent on its original support lattice.
"""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_round import load_lines
from engine.single_wall_gate import SETTINGS as GATE, STRIDE, partition, prepare, votes
from runtime.mode import image_only


SETTINGS = dict(min_agreement=.60, min_segment_px=16, boundary_distance_px=6,
                min_aligned_samples=4, min_boundary_fraction=.25, min_aligned_span_px=12,
                tangent_tolerance_degrees=39, junction_angle_degrees=28,
                junction_radius_px=8, max_boundary_end_distance_px=14, continuation_px=12)
METHOD = 'ledge-corner-cut'
CONTROL = 'single-v2'


def at_junction(endpoint, inward, lines, dirs, lengths):
    """A branch must meet the terminal point. Ignore an X where both lines continue."""
    foot = np.einsum('nd,nd->n', endpoint - lines[:, 0], dirs).clip(0, lengths)
    dist = np.linalg.norm(endpoint - lines[:, 0] - foot[:, None] * dirs, axis=1)
    close = dist <= SETTINGS['junction_radius_px']
    cosines = np.abs(dirs @ inward)
    crossing = cosines < np.cos(np.deg2rad(SETTINGS['junction_angle_degrees']))
    branches = close & crossing
    evidence = dict(branches=int(branches.sum()),
                    nearest_transverse_px=float(dist[crossing].min()) if crossing.any() else None,
                    xy=endpoint.tolist())
    aligned = close & (cosines > np.cos(np.deg2rad(10)))
    extent = (lines - endpoint) @ inward
    continues = np.any(aligned & (extent.min(axis=1) < -SETTINGS['continuation_px']))
    for i in np.flatnonzero(branches):
        # A T also exists when the short branch ends against an unbroken main edge.
        same_branch = close & (np.abs(dirs @ dirs[i]) > np.cos(np.deg2rad(10)))
        span = ((lines[same_branch] - endpoint) @ dirs[i]).ravel()
        if not continues or min(-span.min(), span.max()) < SETTINGS['continuation_px']:
            return True, evidence
    return False, evidence


def corner_evidence(samples, segments):
    """Measure supported boundary length with line and mask samples in source pixels."""
    pair = np.asarray(samples)[:, 2:6].reshape(-1, 2, 2)[:, :, ::-1]
    points = (pair.mean(axis=1) + .5) * STRIDE
    across = pair[:, 1] - pair[:, 0]
    lines = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    keep = np.isfinite(lengths) & (lengths >= SETTINGS['min_segment_px'])
    lines, lengths = lines[keep], lengths[keep]
    dirs = (lines[:, 1] - lines[:, 0]) / lengths[:, None]
    best = None
    rejected = None
    for line, direction, length in zip(lines, dirs, lengths):
        relative = points - line[0]
        along = relative @ direction
        nearest = line[0] + along.clip(0, length)[:, None] * direction
        hits = ((np.linalg.norm(points - nearest, axis=1) <= SETTINGS['boundary_distance_px'])
                & (np.abs(across @ direction) <= np.sin(np.deg2rad(SETTINGS['tangent_tolerance_degrees']))))
        n = int(hits.sum())
        if n < max(SETTINGS['min_aligned_samples'], len(points) * SETTINGS['min_boundary_fraction']):
            continue
        span = float(np.ptp(along[hits]))
        if span < SETTINGS['min_aligned_span_px']:
            continue
        ends = []
        for endpoint, inward in zip(line, (direction, -direction)):
            if np.linalg.norm(points[hits] - endpoint, axis=1).min() > SETTINGS['max_boundary_end_distance_px']:
                continue
            junction, evidence = at_junction(endpoint, inward, lines, dirs, lengths)
            ends.append(evidence)
            if junction:
                if best is None or (n, span) > (best['samples'], best['span_px']):
                    best = dict(samples=n, span_px=span, line_xy=line.tolist(), junction_xy=endpoint.tolist())
        if rejected is None or (n, span) > (rejected['samples'], rejected['span_px']):
            rejected = dict(samples=n, span_px=span, line_xy=line.tolist(), ends=ends)
    return best, (rejected if best is None else None)


def predict(image, baseline, inventory, events, masks, segments):
    """Keep accepted default splits and evaluate the new rule on other intact parents."""
    intact = {p['id'] for p in baseline}
    choices, examined = {}, []
    for candidate, event in zip(inventory, events, strict=True):
        parent, proposal = candidate['parent'], candidate['proposal']
        if (parent, proposal) != (event['parent'], event['proposal']):
            raise ValueError('Mismatched depth and region candidates')
        strength = abs(event['median_step'])
        agreement = event['sign_agreement']
        if parent not in intact or strength < GATE['min_step'] or not SETTINGS['min_agreement'] <= agreement < GATE['min_agreement']:
            continue
        support, rejected = corner_evidence(candidate['samples'], segments)
        detail = dict(parent=parent, proposal=proposal, agreement=agreement, median_step=event['median_step'],
                      sample_count=event['sample_count'], edge=support, best_unconfirmed_line=rejected)
        examined.append(detail)
        if support is None:
            continue
        rank = (support['samples'] / event['sample_count'], strength, -proposal)
        if parent not in choices or rank > choices[parent][0]:
            choices[parent] = rank, detail
    output, added = [], []
    for part in baseline:
        winner = choices.get(part['id'])
        if winner:
            detail = winner[1]
            i = detail['proposal']
            children = partition(image, part, i, masks[i])
            if {'inside', 'remainder'} <= {p['single_wall_side'] for p in children}:
                output.extend(children)
                added.append(detail)
                continue
        output.append(part)
    return output, added, examined


def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Archived input differs: {path}')
    return path


def json_gz(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)


def captures(root):
    locations = {}
    for index in (root/'benchmark/data').glob('*/index.json'):
        for frame in read(index).get('frames', []):
            if 'image' in frame:
                key = frame.get('floor', 'smooth_stone') + '/' + frame['id']
                if key in locations:
                    raise ValueError(f'Duplicate screenshot: {key}')
                locations[key] = index.parent/frame['image']
    return locations


@image_only()
def run(root, out):
    reviews = root/'benchmark/reviews'
    active, sam, depth = (reviews/name for name in ('single-wall-depth-01', 'mobile-regions-01', 'single-wall-v2-fields-01'))
    lines_dir = root/'benchmark/data/mask-lines-v1'
    sources = [active, sam, depth, lines_dir/'scalelsd']
    manifest, proposals, depth_meta, line_meta = [read(s/'manifest.json') for s in sources]
    for directory in (sam, depth):
        verified(directory/'manifest.json', manifest['sources'][directory.name])
    if manifest['line_models']['scalelsd']['checkpoint_sha256'] != line_meta['checkpoint_sha256']:
        raise ValueError('ScaleLSD weights changed')
    previous_lines = read(reviews/'separation-01/manifest.json')['line_input_sha256']['scalelsd']
    verified(lines_dir/'scalelsd/manifest.json', previous_lines)
    images = captures(root)
    rows = read(verified(active/'cases.json', manifest['artifacts']['cases.json']))
    if len(rows) != 56 or len({r['key'] for r in rows}) != 56 or any(r['key'] not in images for r in rows):
        raise ValueError('Expected all 56 frozen views')
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(method=METHOD, settings=SETTINGS, baseline_gate=GATE, default=CONTROL,
                    freeze='One fixed rule before processing development. No search or retuning. Development first, regression second.')
    save(out/'protocol.json', protocol)
    results = []
    for split in ('development', 'previously_reviewed_regression'):
        for row in (r for r in rows if r['cohort'] == split):
            key, sha = row['key'], row['sha256']
            with Image.open(verified(images[key], sha)) as pic:
                image = pic.convert('RGB')
            parts = json_gz(verified(active/row['predictions'], manifest['artifacts'][row['predictions']]))
            npz_name, meta_name = 'proposals/'+sha+'.npz', 'proposals/'+sha+'.json.gz'
            mobile = json_gz(verified(sam/meta_name, proposals['artifacts'][meta_name]))
            indices = mobile['selected']
            with np.load(verified(sam/npz_name, proposals['artifacts'][npz_name]), allow_pickle=False) as data:
                masks = {i: np.unpackbits(data['packed'][i], count=int(data['shape'][2]), axis=1).astype(bool)
                         for i in indices}
            depth_name = 'fields/'+sha+'.npz'
            with np.load(verified(depth/depth_name, depth_meta['artifacts'][depth_name]), allow_pickle=False) as data:
                grid = data['inverse_depth']
            lines = load_lines(lines_dir, 'scalelsd', sha, image.size, line_meta)
            if grid.shape != (image.height//STRIDE, image.width//STRIDE):
                raise ValueError('Depth and screenshot strides differ')
            transform = lines['transform']
            if (any(transform['padding_ltrb']) or not np.allclose(transform['input_to_native_xy'],
                    np.asarray(image.size) / transform['input_size'])):
                raise ValueError('Expected ScaleLSD native coordinates from unpadded resize')
            begin = perf_counter()
            inventory = prepare(image, parts['current'], masks, mobile['candidates'], indices)
            events = votes(inventory, grid)
            prepared = perf_counter()
            result, added, examined = predict(image, parts[CONTROL], inventory, events, masks, lines['segments'])
            elapsed = perf_counter() - prepared
            relative = 'predictions/' + key + '.json.gz'
            path = out/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps(result, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
            results.append(dict(key=key, sha256=sha, cohort=split, layout=row['layout'], predictions=relative,
                                prediction_sha256=digest(path), cached_processing_s=elapsed,
                                shared_inventory_vote_seconds=prepared-begin, added_cuts=added, near_votes=examined))
        if split == 'development':
            save(out/'development-freeze.json', dict(protocol_sha256=digest(out/'protocol.json'),
                 frames=[r['key'] for r in results], cuts=sum(len(r['added_cuts']) for r in results),
                 selection='Predeclared candidate evaluated unchanged on the remaining six reviewed buildings.'))
    if len(results) != len(rows):
        raise ValueError('A frozen screenshot was not processed')
    save(out/'cases.json', results)
    save(out/'manifest.json', dict(run_id=out.name, method=METHOD, code=revision(), fixtures=manifest['fixtures'],
         reference_sha256=manifest['reference_sha256'], splits=manifest['splits'], settings=SETTINGS,
         source_manifests={s.relative_to(root).as_posix():digest(s/'manifest.json') for s in sources},
         line_checkpoint_sha256=line_meta['checkpoint_sha256'], version=manifest['version'],
         timing='No new inference. cached_processing_s measures only the additional refinement; shared preparation and depth voting are separate.',
         reproduce='python -m benchmark.ledge_cut_test --source-root . --out runs/ledge-cuts-test',
         artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.source_root.resolve(), args.out.resolve())
