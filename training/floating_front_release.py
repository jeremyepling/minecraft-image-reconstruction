"""Package, stage and resolve the paired Floating front-view capture release."""

import argparse
import json
import re
from collections import defaultdict
from hashlib import sha256 as hash_bytes
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from training.floating_combined import REFERENCE, sha256


DATASET = 'floating-front-100-v1'
PIN = Path(__file__).with_name(f'{DATASET}.sha256')
COHORTS = {'front-full': 'full', 'front-decorative': 'decorative'}
CLOCK = {'noon': 6000, 'morning': 400, 'afternoon': 11400}
LIGHTING = (*CLOCK, 'all')
SHARDS = ('full-noon', 'full-morning', 'full-afternoon', 'decorative-noon')
STRUCTURES = f'{DATASET}-structures.zip'
SPLITS = ('train', 'validation', 'test')
CAPTURE_FILES = ('rgb.png', *REFERENCE)
RELEASE_FILES = (*CAPTURE_FILES, 'world-readback.json')
LAYOUTS = 100


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def digest_value(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def pinned_hash(provided=None):
    pinned = PIN.read_text(encoding='utf-8').strip() if PIN.is_file() else None
    if pinned is not None and provided is not None and pinned != provided:
        raise ValueError(f'{DATASET} release digest disagrees with the checkout pin')
    expected = provided if pinned is None else pinned
    if not digest_value(expected):
        raise ValueError(f'{DATASET} needs a pinned manifest SHA-256; provide --release-sha256 while preparing the release')
    return expected


def shard_name(key):
    return f'{DATASET}-{key}.zip'


def frame_id_valid(frame_id):
    return isinstance(frame_id, str) and re.fullmatch(r'front-[a-z0-9-]+', frame_id) is not None


def release_files(row):
    return (*RELEASE_FILES, *sorted(row['files'].keys() - set(RELEASE_FILES)))


def validate_rows(rows):
    by_layout = defaultdict(dict)
    by_shard = {key: [] for key in SHARDS}
    rgb_layouts = {}
    family_splits = {}
    for row in rows:
        key = f"{row['condition']}-{row['lighting']}"
        if (not frame_id_valid(row['id']) or key not in by_shard or row['split'] not in SPLITS
                or row['clock_ticks'] != CLOCK[row['lighting']]
                or not row['framing']['accepted'] or row['ground_y'] != 64
                or row['counts']['visible_blocks'] <= 0
                or not all(digest_value(row['files'].get(name)) for name in release_files(row))
                or any(name == 'frame.json' or not re.fullmatch(r'[a-z0-9][a-z0-9._-]+', name)
                       for name in row['files'])):
            raise ValueError(f"Invalid front capture metadata: {row['id']}")
        if rgb_layouts.setdefault(row['files']['rgb.png'], row['layout']) != row['layout']:
            raise ValueError(f"Identical RGB crosses layout boundaries: {row['id']}")
        if family_splits.setdefault(row['family'], row['split']) != row['split']:
            raise ValueError(f"Architecture family crosses dataset splits: {row['family']}")
        captures = by_layout[row['layout']]
        if key in captures:
            raise ValueError(f"Duplicate {key} capture for layout {row['layout']}")
        captures[key] = row
        by_shard[key].append(row)
    if len(by_layout) != LAYOUTS or len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Expected {LAYOUTS} layouts with unique frame IDs')
    for layout, captures in by_layout.items():
        if captures.keys() != by_shard.keys():
            raise ValueError(f'Missing paired captures for {layout}: {sorted(by_shard.keys() - captures.keys())}')
        noon = captures['full-noon']
        if captures['decorative-noon']['files']['rgb.png'] == noon['files']['rgb.png']:
            raise ValueError(f'Decorations made no visible difference: {layout}')
        for row in captures.values():
            fields = ('split', 'family', 'palette', 'size', 'angle', 'dimensions')
            if row['parent_id'] != noon['id'] or any(row[name] != noon[name] for name in fields):
                raise ValueError(f"{row['id']} disagrees with its clean noon parent")
            if row['condition'] == 'full' and (row['counts'] != noon['counts']
                    or row['files']['labels.npz'] != noon['files']['labels.npz']):
                raise ValueError(f"Visible labels changed across lighting: {row['id']}")
    return by_shard


def camera_geometry(row, payload):
    camera = json.loads(payload)
    if camera['id'] != row['id'] or camera['clock_ticks'] % 24000 != row['clock_ticks']:
        raise ValueError(f"Native camera frame or clock differs from release metadata: {row['id']}")
    return tuple(camera[key] for key in ('image_size', 'eye_world', 'view_rotation_projection'))


def check_camera_pairs(rows, payload_for):
    cameras = {row['id']: camera_geometry(row, payload_for(row, 'camera.json')) for row in rows}
    for row in rows:
        if cameras[row['id']] != cameras[row['parent_id']]:
            raise ValueError(f"Paired camera geometry changed: {row['id']}")


def check_lighting_labels(grouped, payload_for):
    def labels(row):
        value = json.loads(payload_for(row, 'labels.json'))
        if value.pop('frame_id') != row['id']:
            raise ValueError(f"Visible labels have the wrong frame ID: {row['id']}")
        return value

    noon = {row['id']: row for row in grouped['full-noon']}
    for key in ('full-morning', 'full-afternoon'):
        for row in grouped[key]:
            if labels(row) != labels(noon[row['parent_id']]):
                raise ValueError(f"Visible geometry or materials changed across lighting: {row['id']}")


def write_zip(archive, name, data):
    info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED if name.endswith(('.png', '.npz')) else ZIP_DEFLATED
    archive.writestr(info, data)


def capture_payload(root, row, name):
    path = root[row['id']] / name
    payload = path.read_bytes()
    if hash_bytes(payload).hexdigest() != row['files'][name]:
        raise ValueError(f'Capture checksum mismatch: {path}')
    return payload


def package(captures, out):
    """Publish a deterministic four-shard release after all complete pairs pass validation."""
    if out.exists():
        raise FileExistsError(f'Use a new release directory: {out}')
    captures = tuple(captures)
    roots = {}
    rows = []
    for capture in captures:
        frames = sorted((capture / 'frames').glob('*/frame.json'))
        if not frames:
            raise ValueError(f'No labeled front captures found in {capture}')
        for path in frames:
            row = json.loads(path.read_bytes())
            if row['id'] in roots or row['id'] != path.parent.name:
                raise ValueError(f'Duplicate frame or capture path disagrees with its ID: {path}')
            roots[row['id']] = path.parent
            rows.append(row)
    rows.sort(key=lambda row: row['id'])
    grouped = validate_rows(rows)

    def payload_for(row, name):
        return capture_payload(roots, row, name)

    check_camera_pairs(rows, payload_for)
    check_lighting_labels(grouped, payload_for)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'{DATASET}-', dir=out.parent) as temp:
        release = Path(temp) / DATASET
        release.mkdir()
        specs = {}
        for key, frames in grouped.items():
            path = release / shard_name(key)
            with ZipFile(path, 'w') as archive:
                for row in frames:
                    directory = f"frames/{row['id']}"
                    write_zip(archive, f'{directory}/frame.json', encode(row))
                    for name in release_files(row):
                        write_zip(archive, f'{directory}/{name}', payload_for(row, name))
            specs[key] = dict(file=path.name, frames=len(frames), sha256=sha256(path), bytes=path.stat().st_size)
        from training.front_structures import build_archive

        structures = build_archive(captures, release / STRUCTURES)
        checked_structures(release, structures)
        manifest = dict(dataset=DATASET, schema_version=1, minecraft_version='26.3', layouts=LAYOUTS,
                        clock_ticks=CLOCK, shards=specs, structures=structures, frames=rows)
        payload = encode(manifest)
        (release / 'manifest.json').write_bytes(payload)
        release.rename(out)
    return hash_bytes(payload).hexdigest()


def load_manifest(directory, expected):
    path = directory / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {DATASET} manifest: {path}')
    payload = path.read_bytes()
    if hash_bytes(payload).hexdigest() != expected:
        raise ValueError(f'{DATASET} manifest fingerprint changed: {path}')
    manifest = json.loads(payload)
    if (manifest['dataset'] != DATASET or manifest['schema_version'] != 1
            or manifest['minecraft_version'] != '26.3' or manifest['layouts'] != LAYOUTS
            or manifest['clock_ticks'] != CLOCK or set(manifest['shards']) != set(SHARDS)):
        raise ValueError(f'Unexpected front capture release index: {path}')
    grouped = validate_rows(manifest['frames'])
    for key, frames in grouped.items():
        spec = manifest['shards'][key]
        if (spec['file'] != shard_name(key) or spec['frames'] != len(frames)
                or not digest_value(spec['sha256'])):
            raise ValueError(f'Invalid {key} archive specification: {path}')
    if not valid_structure_spec(manifest['structures']):
        raise ValueError(f'Invalid structures archive specification: {path}')
    return payload, manifest, grouped


def valid_structure_spec(spec):
    return (spec['file'] == STRUCTURES and digest_value(spec['sha256']) and spec['bytes'] > 0
            and spec['layouts'] == LAYOUTS and spec['templates'] == 2 * LAYOUTS)


def checked_structures(directory, spec):
    if not valid_structure_spec(spec):
        raise ValueError(f'Invalid structures archive specification: {directory}')
    path = directory / STRUCTURES
    data = path.read_bytes()
    if len(data) != spec['bytes'] or hash_bytes(data).hexdigest() != spec['sha256']:
        raise ValueError(f'Structures archive fingerprint changed: {path}')
    return data


def write_read_only(path, payload):
    path.write_bytes(payload)
    path.chmod(0o444)


def verify_frame(directory, row):
    frame = directory / row['id']
    if (frame / 'frame.json').read_bytes() != encode(row):
        raise ValueError(f'Fixture frame metadata changed: {frame}')
    for name in CAPTURE_FILES:
        if sha256(frame / name) != row['files'][name]:
            raise ValueError(f'Fixture file checksum mismatch: {frame / name}')
    return frame


def stage(root, release, *, release_sha256=None):
    """Publish all verified scoring and source inputs together under the shared root."""
    expected = pinned_hash(release_sha256)
    target = root / 'frames' / DATASET
    if target.exists():
        _, manifest, _ = load_manifest(target, expected)
        checked_structures(target, manifest['structures'])
        for row in manifest['frames']:
            verify_frame(target, row)
        return len(manifest['frames'])
    payload, manifest, grouped = load_manifest(release, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'{DATASET}-', dir=target.parent) as temp:
        staging = Path(temp) / DATASET
        staging.mkdir()
        write_read_only(staging / STRUCTURES, checked_structures(release, manifest['structures']))
        for key, frames in grouped.items():
            path = release / shard_name(key)
            if sha256(path) != manifest['shards'][key]['sha256']:
                raise ValueError(f'Archive fingerprint changed: {path}')
            with ZipFile(path) as archive:
                for row in frames:
                    frame = staging / row['id']
                    frame.mkdir()
                    prefix = f"frames/{row['id']}"
                    metadata = archive.read(f'{prefix}/frame.json')
                    if metadata != encode(row):
                        raise ValueError(f'Capture metadata differs from release manifest: {frame}')
                    for name in release_files(row):
                        data = archive.read(f'{prefix}/{name}')
                        if hash_bytes(data).hexdigest() != row['files'][name]:
                            raise ValueError(f'Capture file checksum mismatch: {frame / name}')
                        if name in CAPTURE_FILES:
                            write_read_only(frame / name, data)
                    write_read_only(frame / 'frame.json', metadata)
        write_read_only(staging / 'manifest.json', payload)
        staging.rename(target)
    return sum(map(len, grouped.values()))


def cases(root, *, cohort, lighting=None, ids=None, release_sha256=None):
    condition = COHORTS[cohort]
    lighting = lighting or 'noon'
    if lighting not in LIGHTING or (condition == 'decorative' and lighting != 'noon'):
        raise ValueError(f'Unsupported lighting for {cohort}: {lighting}')
    expected = pinned_hash(release_sha256)
    directory = root.resolve() / 'frames' / DATASET
    _, manifest, _ = load_manifest(directory, expected)
    rows = [row for row in manifest['frames']
            if row['condition'] == condition and (lighting == 'all' or row['lighting'] == lighting)]
    if ids:
        wanted = set(ids)
        if unknown := wanted - {row['id'] for row in rows}:
            raise ValueError(f'Unknown {cohort} {lighting} views: {", ".join(sorted(unknown))}')
        rows = [row for row in rows if row['id'] in wanted]
    result = []
    for row in rows:
        frame = verify_frame(directory, row)
        sha = row['files']['rgb.png']
        case = {key: row[key] for key in ('id', 'split', 'layout', 'family', 'palette', 'size',
                                         'condition', 'lighting', 'clock_ticks', 'parent_id')}
        case.update(dataset=DATASET, rgb_sha256=sha, frame=str(frame), dataset_manifest_sha256=expected,
                    reference_sha256={name: row['files'][name] for name in REFERENCE},
                    variants={'original': dict(image=str(frame / 'rgb.png'), source_sha256=sha)})
        result.append(case)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures-root', type=Path, action='append', required=True,
                        help='Labeled capture batch directory (repeat for independent batches)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(f'{DATASET} manifest SHA-256: {package(args.captures_root, args.out)}')


if __name__ == '__main__':
    main()
