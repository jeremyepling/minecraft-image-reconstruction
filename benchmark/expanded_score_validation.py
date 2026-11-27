"""Validate expanded capture targets, including their geometry-only alignment."""

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import file_digest, sha256
from io import BytesIO
import json
from math import prod
from operator import index
from pathlib import Path
from random import Random
from zipfile import ZipFile

import numpy as np

from benchmark.relative_geometry import best_translation


MAX_CORRELATION_CELLS = 4_000_000
SMALL_PAIR_COUNT = 8_192
COHORTS = {'visible-builds-v3': 144, 'decorative-builds-v1': 40}
MANIFEST_SHA256 = {
    'visible-builds-v3': '6e7b25fa56aa39d04a34bb3984551cf11441c11acb1a021b045d4f86e824aa28',
    'decorative-builds-v1': '09d5612ee175718b979c73363eb40fe05959d90f96aab894e82efa9548eaf064',
}
SOURCE_REVISION = '1ca88221bf2a29d4495b61a783b0c99273ee0634'
ASSETS = {'rgb.png', 'camera.json', 'labels.json', 'labels.npz', 'world-readback.json'}


def _fft_length(length):
    """Smallest practical smooth FFT length at or above the linear extent."""
    while True:
        rest = length
        for factor in (2, 3, 5):
            while rest % factor == 0:
                rest //= factor
        if rest == 1:
            return length
        length += 1


def best_translation_fast(predicted, target):
    """Find the same whole-build offset and overlap as the sparse reference scorer."""
    prediction = {tuple(map(index, (x, y, z))) for x, y, z in predicted}
    reference = {tuple(map(index, (x, y, z))) for x, y, z in target}
    at_origin = len(prediction & reference)
    ceiling = min(len(prediction), len(reference))
    if at_origin == ceiling:
        return (0, 0, 0), at_origin
    if len(prediction) * len(reference) <= SMALL_PAIR_COUNT:
        return best_translation(prediction, reference)

    prediction_axes = tuple(zip(*prediction))
    reference_axes = tuple(zip(*reference))
    pred_low = tuple(map(min, prediction_axes))
    pred_high = tuple(map(max, prediction_axes))
    ref_low = tuple(map(min, reference_axes))
    ref_high = tuple(map(max, reference_axes))
    pred_shape = tuple(high - low + 1 for low, high in zip(pred_low, pred_high))
    ref_shape = tuple(high - low + 1 for low, high in zip(ref_low, ref_high))
    full_shape = tuple(a + b - 1 for a, b in zip(pred_shape, ref_shape))
    if prod(full_shape) > MAX_CORRELATION_CELLS:
        return best_translation(prediction, reference)
    fft_shape = tuple(map(_fft_length, full_shape))
    if prod(fft_shape) > MAX_CORRELATION_CELLS:
        return best_translation(prediction, reference)

    # Reverse the prediction so linear convolution counts all target-minus-prediction votes.
    pred_grid = np.zeros(pred_shape, dtype=np.float64)
    ref_grid = np.zeros(ref_shape, dtype=np.float64)
    pred_indices = tuple([high - v for v in axis] for high, axis in zip(pred_high, prediction_axes))
    ref_indices = tuple([v - low for v in axis] for low, axis in zip(ref_low, reference_axes))
    pred_grid[pred_indices] = 1
    ref_grid[ref_indices] = 1
    axes = (0, 1, 2)
    spectrum = np.fft.rfftn(pred_grid, s=fft_shape, axes=axes)
    spectrum *= np.fft.rfftn(ref_grid, s=fft_shape, axes=axes)
    votes = np.fft.irfftn(spectrum, s=fft_shape, axes=axes)
    votes = votes[tuple(slice(n) for n in full_shape)]
    rounded = np.rint(votes)
    # Binary occupancy gives integer votes. Reject unexpected numerical error instead of guessing a tie.
    if not np.isfinite(rounded).all() or np.max(np.abs(votes - rounded)) >= 0.125:
        return best_translation(prediction, reference)
    best = int(rounded.max())
    if best == at_origin:
        return (0, 0, 0), best
    if best < at_origin or best > ceiling:
        return best_translation(prediction, reference)
    origin = tuple(low - high for low, high in zip(ref_low, pred_high))
    offsets = (tuple(int(v) + o for v, o in zip(i, origin)) for i in np.argwhere(rounded == best))
    winner = min(offsets, key=lambda xyz: (sum(v * v for v in xyz), xyz))
    dx, dy, dz = winner
    overlap = sum((x + dx, y + dy, z + dz) in reference for x, y, z in prediction)
    if overlap != best:
        return best_translation(prediction, reference)
    return winner, best


def _shift(points, offset):
    dx, dy, dz = offset
    return {(x + dx, y + dy, z + dz) for x, y, z in points}


def translation_checks():
    """Exercise FFT ties, negative offsets, the origin shortcut and sparse fallbacks."""
    pred = {(x, y, z) for x in range(8) for y in range(5) for z in range(3)}
    sparse = {(100_003 * x, -150_001 * x, x) for x in range(96)}
    cases = [
        ('empty both', [], [], ((0, 0, 0), 0)),
        ('empty target', pred, [], ((0, 0, 0), 0)),
        ('empty prediction', [], pred, ((0, 0, 0), 0)),
        ('duplicates', [(-2, 0, 0)] * 2, [(2, 0, 0)] * 2, ((4, 0, 0), 1)),
        ('known translation', pred, _shift(pred, (-17, 3, -19)), ((-17, 3, -19), len(pred))),
        ('lexicographic tie', pred, _shift(pred, (-15, 0, 0)) | _shift(pred, (15, 0, 0)),
         ((-15, 0, 0), len(pred))),
        ('smallest squared displacement', pred, _shift(pred, (-17, 0, 0)) | _shift(pred, (9, 7, 0)),
         ((9, 7, 0), len(pred))),
        ('origin tie below ceiling', pred | {(0, 0, 90)}, pred | {(0, 0, -90)} | _shift(pred, (45, 0, 0)),
         ((0, 0, 0), len(pred))),
        ('same-origin subset', pred, {(0, 0, 0)}, ((0, 0, 0), 1)),
        ('too-wide sparse fallback', sparse, _shift(sparse, (-4, 5, -3)), ((-4, 5, -3), len(sparse))),
        ('large absolute coordinates', _shift(pred, (10**22, -10**30, 0)),
         _shift(pred, (-10**22, -10**30, 5)), ((-2 * 10**22, 0, 5), len(pred))),
    ]
    rng = Random(168)
    for sample in range(48):
        prediction = {(rng.randrange(-7, 7), rng.randrange(-4, 5), rng.randrange(-8, 8))
                      for _ in range(96 + sample * 2)}
        reference = {(rng.randrange(-4, 12), rng.randrange(-5, 6), rng.randrange(-4, 10))
                     for _ in range(113 + sample)}
        cases.append((f'seeded random {sample}', prediction, reference, None))
    for name, predicted, target, expected in cases:
        legacy = best_translation(predicted, target)
        assert best_translation_fast(predicted, target) == legacy, name
        assert expected is None or legacy == expected, name
    return {'checks': len(cases), 'seed': 168, 'known_cases': [name for name, _, _, expected in cases if expected]}


def _verified_archive(root, record):
    path = root / record['file']
    with path.open('rb') as stream:
        assert path.stat().st_size == record['bytes'] and file_digest(stream, 'sha256').hexdigest() == record['sha256'], path
    return ZipFile(path)


def _xyz(block):
    return tuple(block[axis] for axis in 'xyz')


def _native_owners(labels, ids, depth, pose):
    blocks, faces = labels['blocks'], labels['faces']
    assert ids.shape == depth.shape == tuple(reversed(pose['image_size'])) == (900, 1440)
    assert ids.dtype == np.uint16 and depth.dtype == np.float32
    assert [block['id'] for block in blocks] == list(range(1, len(blocks) + 1))
    assert [face['id'] for face in faces] == list(range(1, len(faces) + 1))
    assert set(face['block'] for face in faces) == set(range(1, len(blocks) + 1))
    assert all(blocks[face['block'] - 1]['xyz_world'] == face['xyz_world'] for face in faces)
    counts = np.bincount(ids.ravel())
    assert counts[1:].tolist() == [face['visible_pixels'] for face in faces]
    assert np.all(counts[1:] > 0)
    foreground = np.flatnonzero(ids)
    assert np.array_equal(np.isfinite(depth), ids != 0) and np.all(np.isnan(depth[ids == 0]))
    assert np.all(depth[ids != 0] > 0)
    first = np.full(len(faces) + 1, ids.size)
    np.minimum.at(first, ids.ravel()[foreground], foreground)
    uniform = foreground[np.linspace(0, len(foreground) - 1, min(4096, len(foreground)), dtype=int)]
    y, x = np.unravel_index(np.union1d(first[1:], uniform), ids.shape)
    camera = labels['camera']
    uv1 = np.column_stack((x + .5, y + .5, np.ones(len(x))))
    rays = uv1 @ np.linalg.inv(camera['K']).T @ np.asarray(camera['world_to_camera_rotation'])
    hits = np.asarray(camera['eye_world']) + rays * depth[y, x, None]
    quads = np.asarray([face['corners_world'] for face in faces])[ids[y, x] - 1]
    # Axis-aligned quad bounds verify both its signed plane and partial model extent.
    lower, upper = quads.min(axis=1), quads.max(axis=1)
    assert np.all(hits >= lower - 2e-4) and np.all(hits <= upper + 2e-4)
    return int(len(foreground)), int(len(x))


def _native_optional(labels, ids):
    """Check horizontal-only targets using runs of four identical native IDs."""
    a, b, c, d = (ids[:, offset:ids.shape[1] - 3 + offset] for offset in range(4))
    wide = np.where((a == b) & (a == c) & (a == d), a, 0)
    a, b, c, d = (wide[offset:wide.shape[0] - 3 + offset] for offset in range(4))
    substantial = set(np.unique(a[(a == b) & (a == c) & (a == d)]))
    directions = [set() for _ in labels['blocks']]
    square = [False] * len(labels['blocks'])
    for face in labels['faces']:
        owner = face['block'] - 1
        directions[owner].add(face['normal_world'][1])
        square[owner] = square[owner] or face['id'] in substantial
    return {tuple(block['xyz_world']) for block, sides, patch in zip(labels['blocks'], directions, square)
            if sides in ({1}, {-1}) and not patch}


def _assert_metric(result, strict, facade):
    for name, (tp, fn, fp) in (('strict', strict), ('facade', facade)):
        assert result[name] == dict(correct=tp, missed=fn, extras=fp, f1=2 * tp / (2 * tp + fn + fp))


def _check_pack(pose, origin, truth, optional, hidden):
    from benchmark.camera_metrics import registration, truth_camera
    from benchmark.mask_blocks import decoded_pack
    from benchmark.typed_structures import aligned_sites, geometry_score, score
    from engine.export import JAVA_26_3_FORMAT, write_pack
    from engine.nbt import read_nbt

    # This known local frame swaps X/Z and reflects one axis. Coordinates are cell centers for the reflection.
    local_to_world = np.array([[0, 0, -1, origin[0]], [0, 1, 0, origin[1]],
                               [1, 0, 0, origin[2]], [0, 0, 0, 1]])
    estimated = registration(truth_camera(pose) @ local_to_world, pose)
    assert np.array_equal(estimated, local_to_world)
    local = np.floor((np.asarray(list(truth)) + .5 - local_to_world[:3, 3]) @ local_to_world[:3, :3]).astype(int)
    shift = (17, -4, 9)
    shifted = [tuple(map(int, pos)) for pos in local + shift]
    packed = dict(zip(shifted, truth.values(), strict=True))
    buffer = BytesIO()
    write_pack(buffer, {p: name.removeprefix('minecraft:') for p, name in packed.items()}, JAVA_26_3_FORMAT)
    decoded = decoded_pack(buffer)
    assert decoded == packed
    with ZipFile(buffer) as archive:
        for entry in archive.namelist():
            if entry.endswith('.nbt'):
                data = read_nbt(gzip.decompress(archive.read(entry)))
                assert all(set(state) == {'id'} for state in data['palette'])

    positions = aligned_sites(list(decoded), estimated, (0, 0, 0))
    assert set(positions) == _shift(truth, (-9, -4, 17))
    translation, overlap = best_translation_fast(positions, truth)
    assert translation == (9, 4, -17) and overlap == len(truth)
    registered = dict(zip(aligned_sites(list(decoded), estimated, translation), decoded.values(), strict=True))
    assert registered == truth
    target, required = len(truth), len(truth) - len(optional)
    _assert_metric(geometry_score(registered, truth, optional), (target, 0, 0), (required, 0, 0))
    _assert_metric(score(registered, truth, optional, registered.keys()), (target, 0, 0), (required, 0, 0))

    # One wrong required ID and one unsupported placement have distinct penalties.
    wrong = min(truth.keys() - optional)
    extra = min(hidden) if hidden else (min(p[0] for p in truth) - 1, origin[1], origin[2])
    assert extra not in truth
    changed = registered | {wrong: 'minecraft:air', extra: registered[wrong]}
    _assert_metric(geometry_score(changed, truth, optional), (target, 0, 1), (required, 0, 1))
    _assert_metric(score(changed, truth, optional, changed.keys()), (target - 1, 1, 2), (required - 1, 1, 2))
    if optional:
        thin = min(optional)
        changed[thin] = 'minecraft:air'
        _assert_metric(score(changed, truth, optional, changed.keys()), (target - 2, 2, 3), (required - 1, 1, 2))
        del changed[thin]
        _assert_metric(geometry_score(changed, truth, optional), (target - 1, 1, 1), (required, 0, 1))
        _assert_metric(score(changed, truth, optional, changed.keys()), (target - 2, 2, 2), (required - 1, 1, 2))


def _check_frame(root, archive, row, listed, capture, plan, catalog):
    from PIL import Image

    from benchmark.strip_visibility_audit import horizontal_slivers
    from calibration.world import GROUND
    from training.framing import measure_framing

    fid, split = row['id'], row['split']
    assert row['directory'] == 'frames/' + fid and row['ground_y'] == GROUND == 64
    assert set(row['files']) == ASSETS
    assert all(row[field] == listed[field] for field in ('layout', 'palette', 'split', 'family'))
    assert all(row[field] == capture[field] for field in ('layout', 'palette', 'split', 'page', 'angle', 'condition'))
    assert row['files']['rgb.png'] == listed['rgb_sha256']
    frozen = root / 'data' / split / row['directory']
    assets = {}
    for name, digest in row['files'].items():
        payload = archive.read(f"{row['directory']}/{name}")
        assert sha256(payload).hexdigest() == digest and (frozen / name).read_bytes() == payload, name
        assets[name] = payload
    archived_index = archive.read(f"{row['directory']}/frame.json")
    assert json.loads(archived_index) == row and (frozen / 'frame.json').read_bytes() == archived_index

    labels = json.loads(assets['labels.json'])
    pose = json.loads(assets['camera.json'])
    source = json.loads(assets['world-readback.json'])
    actual = {_xyz(block): block for block in source}
    assert len(actual) == len(source) and actual.keys() == plan.keys()
    assert all(block['block_id'].removeprefix('minecraft:') == plan[point] for point, block in actual.items())
    is_decorative = row['dataset'] == 'decorative-builds-v1'
    label_prefix = 'visible-decorative-block' if is_decorative else 'visible-block'
    assert labels['schema'] == f'{label_prefix}-labels/v1'
    assert labels['frame_id'] == pose['id'] == fid and pose['minecraft_version'] == '26.3'
    assert labels['camera']['eye_world'] == pose['eye_world']
    with Image.open(BytesIO(assets['rgb.png'])) as rgb:
        assert rgb.format == 'PNG' and list(rgb.size) == pose['image_size'] == labels['camera']['image_size']

    origin = labels['origin_world']
    assert origin == row['origin_world'] and origin[1] == GROUND + 1
    truth, kinds = {}, {}
    states = Counter()
    for block in labels['blocks']:
        point, identifier = tuple(block['xyz_world']), block['block_id']
        assert point not in truth and point[1] > GROUND and identifier.startswith('minecraft:')
        assert identifier != 'minecraft:air' and block['xyz_local'] == [v - o for v, o in zip(point, origin)]
        known = actual[point]
        assert identifier == 'minecraft:' + known['block_id'].removeprefix('minecraft:')
        assert block['state'] == known['state']
        truth[point] = identifier
        kinds[point] = block.get('geometry_kind', 'full_cube')
        if '[' in block['state']:
            states[kinds[point]] += 1
    hidden = {point for point in actual.keys() - truth.keys() if point[1] > GROUND}
    with np.load(BytesIO(assets['labels.npz']), allow_pickle=False) as raster:
        ids, depth = raster['face_id'], raster['depth_z']
        pixels, samples = _native_owners(labels, ids, depth, pose)
        assert measure_framing(pose, source, GROUND, ids) == row['framing'] and row['framing']['accepted']
        optional = _native_optional(labels, ids)
        assert optional == horizontal_slivers(labels, ids)

    totals = dict(frames=1, visible_blocks=len(truth), visible_faces=len(labels['faces']), visible_pixels=pixels,
                  native_depth_samples=samples, optional_targets=len(optional), required_targets=len(truth) - len(optional),
                  hidden_structure_appearances=len(hidden), frames_with_optional_targets=int(bool(optional)))
    assert all(totals[field] == row['counts'][field] for field in ('visible_blocks', 'visible_faces', 'visible_pixels'))
    type_counts = Counter(kinds.values())
    if is_decorative:
        assert type_counts == labels['block_kinds'] == row['counts']['visible_block_kinds']
        assert labels['overlapping_visible_full_blocks'] == row['counts']['visible_full_blocks_behind_decorations']
        assert labels['occluded_overlap_pixels'] == row['counts']['visible_overlap_occluded_pixels']
        totals.update(overlapping_visible_full_blocks=labels['overlapping_visible_full_blocks'],
                      occluded_overlap_pixels=labels['occluded_overlap_pixels'],
                      frames_with_foreground_overlap=int(labels['overlapping_visible_full_blocks'] > 0))
    _check_pack(pose, origin, truth, optional, hidden)
    missing_ids = Counter(kinds[point] for point, name in truth.items() if name not in catalog)
    return totals, type_counts, Counter(kinds[point] for point in optional), states, missing_ids


def verify_cohort(root, cohort, catalog):
    manifest_path = root / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = sha256(manifest_bytes).hexdigest()
    assert manifest_hash == MANIFEST_SHA256[cohort], manifest_path
    manifest = json.loads(manifest_bytes)
    assert manifest['dataset'] == cohort and manifest['minecraft_version'] == '26.3'
    assert manifest['schema_version'] == 1 and manifest['min_margin_fraction'] == .02
    assert set(manifest['shards']) == {'train', 'validation', 'test'}
    listed = {row['id']: row for row in manifest['frames']}
    assert len(listed) == len(manifest['frames']) == COHORTS[cohort]
    assert len({row['rgb_sha256'] for row in listed.values()}) == len(listed)
    with _verified_archive(root, manifest['capture_records']) as archive:
        assert archive.testzip() is None
        views = {row['id']: row for row in json.loads(archive.read('views.json'))}
        layout = json.loads(archive.read('layout.json'))
    assert views.keys() == listed.keys() and layout['ground_y'] == 64 and layout['version'] == '26.3'
    pages = defaultdict(dict)
    for block in layout['blocks']:
        page, point = block['page'], _xyz(block)
        assert point not in pages[page]
        pages[page][point] = block['block_id'].removeprefix('minecraft:')

    splits, verified, memberships = {}, {}, {}
    visible_types, optional_types, state_counts, missing_ids = Counter(), Counter(), Counter(), Counter()
    for split, shard in sorted(manifest['shards'].items()):
        totals = Counter()
        with _verified_archive(root, shard) as archive:
            index_bytes = archive.read('frames.jsonl')
            assert (root / 'data' / split / 'frames.jsonl').read_bytes() == index_bytes
            rows = [json.loads(line) for line in index_bytes.splitlines()]
            assert len(rows) == shard['frames'] and len({row['layout'] for row in rows}) == shard['layouts']
            expected_paths = {'frames.jsonl'}
            for row in rows:
                fid = row['id']
                assert fid not in verified and row['split'] == split and row['dataset'] == cohort
                for group in ('layout', 'family'):
                    assert memberships.setdefault((group, row[group]), split) == split
                try:
                    metrics, types, optional, states, out_of_catalog = _check_frame(
                        root, archive, row, listed[fid], views[fid], pages[row['page']], catalog)
                except (AssertionError, KeyError, ValueError) as error:
                    raise ValueError(f'{cohort}/{fid}: validation failed') from error
                totals.update(metrics)
                visible_types.update(types)
                optional_types.update(optional)
                state_counts.update(states)
                missing_ids.update(out_of_catalog)
                verified[fid] = row['files']
                expected_paths.update(f"{row['directory']}/{name}" for name in (*ASSETS, 'frame.json'))
            assert len(expected_paths) == len(archive.namelist()) and expected_paths == set(archive.namelist())
        splits[split] = dict(totals)
        print(f'{cohort}/{split}: validated {totals["frames"]} frames and packs', flush=True)
    assert verified.keys() == listed.keys()
    digest = sha256(json.dumps(verified, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    summary = sum((Counter(entry) for entry in splits.values()), Counter())
    return dict(manifest_sha256=manifest_hash, shards=manifest['shards'],
                capture_records=manifest['capture_records'], frame_asset_fingerprints_sha256=digest,
                frames_available=len(listed), splits=splits, totals=dict(summary),
                known_pack_roundtrip=dict(frames_checked=len(verified), strict_position_f1=1.0, facade_position_f1=1.0,
                                          strict_position_and_id_f1=1.0, facade_position_and_id_f1=1.0),
                visible_blocks_by_kind=dict(visible_types), optional_targets_by_kind=dict(optional_types),
                visible_blocks_without_default_id_candidate_by_kind=dict(missing_ids),
                visible_source_blocks_with_properties_by_kind=dict(state_counts))


def main():
    from engine.atlas import resource_id
    from engine.atlas_count import load_atlas

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Saved runs directory containing both cohorts')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    atlas = load_atlas(args.source / '26.3/atlas', '26.3')
    catalog = {resource_id(name) for alternatives in atlas.ids for name in alternatives}
    assert atlas.provenance['atlas_sha256'] == '49b477fbfe5f2171c2529e3ffe1722f09fee5a094de2b6b99831512b8601c950'
    assert atlas.provenance['placement_index_sha256'] == '32f95873f5b10e0fa36023c149b4277a4b1f7a59fb77940ede120f9292f37ac6'
    result = dict(source_revision=SOURCE_REVISION, java_version='26.3', default_atlas=atlas.provenance,
                  alignment_reference_checks=translation_checks(),
                  scores='Known visible targets supplied to the exporter for evaluator validation',
                  state_scope='Pack palettes carry block IDs alone. Source state properties are absent, so the ID score does not establish matching facing, half, connections, or other properties.',
                  cohorts={cohort: verify_cohort(args.source / cohort, cohort, catalog) for cohort in COHORTS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes((json.dumps(result, indent=2, sort_keys=True) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
