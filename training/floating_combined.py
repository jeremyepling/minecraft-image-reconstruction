"""Index Floating source images and visible labels without repackaging the releases."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from training.floating_cohort import RELEASES, RETIRED_PALETTES, frame_index


MANIFEST = Path(__file__).with_name('floating-combined-v1.json.gz')
V1 = 'visible-builds-v1'
V2 = 'visible-builds-v2'
COMPANION = 'visible-builds-v3-original-204'
BACKGROUND = 'floating-build-grid-01'
REFERENCE = ('camera.json', 'labels.json', 'labels.npz')
LIGHTING = {
    f'{V1}-early-morning': (400, '5d5c39e8f44a08981a860a658ad2124a7b3bfbebf1fbb2f1bc78df6e9070a7bb'),
    f'{V1}-morning': (2500, 'ad08793d6ca21b12e27fa98f4e0af08505ea33fc56ed8c854f0c9230fad48eba'),
    f'{V1}-afternoon': (9500, 'b9985f1873c119d7578803a5a3a3303bccab635da88ed6ffd540cca5e609e9a3'),
    f'{V1}-late-afternoon': (11600, '83512ddebf645f42074783d4a85698f4fb8cce79c7df79586e43421e128762d5'),
}
PINNED = dict(RELEASES) | {V1: 'a702f1d1a79c9e471680fd362993e14a4c2eeb20b4790d6b67b054b5f4bf0a56',
                           COMPANION: '0e321cd58bd9ca24353a0f575ced2b0f8c2195de33895e5ea56ebd8f71049803'}
PINNED.update({name: digest for name, (_, digest) in LIGHTING.items()})


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_release(root, name):
    directory = root / name
    file = directory / (f'{name}-manifest.json' if name == COMPANION else 'manifest.json')
    if sha256(file) != PINNED[name]:
        raise ValueError(f'Release manifest fingerprint changed: {file}')
    release = json.loads(file.read_bytes())
    parts = release['shards'] if 'shards' in release else {'reference': release['archive']}
    index = {row['id']: row for row in release['frames']}
    seen = set()
    for spec in parts.values():
        path = directory / spec['file']
        if sha256(path) != spec['sha256']:
            raise ValueError(f'Archive fingerprint changed: {path}')
        with ZipFile(path) as archive:
            rows = [json.loads(line) for line in archive.read('frames.jsonl').splitlines()]
            if len(rows) != spec['frames']:
                raise ValueError(f'Unexpected row count in {path}')
            members = set(archive.namelist())
            for row in rows:
                marker = row['id']
                hint = index[marker]
                rgb = hint.get('rgb_sha256', hint.get('files', {}).get('rgb.png'))
                if marker in seen or rgb != row['files']['rgb.png']:
                    raise ValueError(f'Duplicate frame ID or inconsistent RGB: {name}/{marker}')
                if any(f"{row['directory']}/{label}" not in members for label in (*REFERENCE, 'rgb.png')):
                    raise ValueError(f'Missing image or reference file: {name}/{marker}')
                seen.add(marker)
                yield row, f"{name}/{spec['file']}"
    if seen != index.keys():
        raise ValueError(f'Incomplete release: {name}')


def frame(row, archive, dataset, role):
    return dict(id=row['id'], dataset=dataset, split=row['split'], role=role, archive=archive,
                member=row['directory'], rgb_sha256=row['files']['rgb.png'], files=row['files'])


def same_reference(a, b):
    return all(a['files'][key] == b['files'][key] for key in REFERENCE)


def check_splits(rows):
    seen = defaultdict(set)
    for row in rows:
        if row['role'] == 'active':
            seen[row['family']].add(row['split'])
    leaks = sorted(family for family, splits in seen.items() if len(splits) != 1)
    if leaks:
        raise ValueError(f'Families cross active splits: {leaks}')


def load(path=MANIFEST):
    data = json.loads(gzip.decompress(path.read_bytes()))
    if data.get('dataset') != 'floating-combined-v1':
        raise ValueError(f'Unexpected Floating manifest: {path}')
    return data


def select_active_cases(cases, index=None):
    """Keep the caller's order while selecting canonical active RGBs by release and ID."""
    if index is None:
        index = {(row['dataset'], row['id']): row for row in load()['frames'] if row['role'] == 'active'}
    for case in cases:
        row = index.get((case['dataset'], case['id']))
        if row is not None:
            if any(case[key] != row[key] for key in ('rgb_sha256', 'split', 'layout', 'palette') if key in case):
                raise ValueError(f"Fixture differs from manifest: {case['id']}")
            yield case


def build(root):
    policy = frame_index()
    framing = json.loads(Path(__file__).with_name('framing-audit-v2.json').read_bytes())
    framing = {row['id']: row for row in framing['frames']}
    rows = []
    by_rgb = {}
    base_v2 = {}

    def admit(row):
        rgb = row['rgb_sha256']
        if rgb in by_rgb:
            raise ValueError(f"Duplicate source across combined tracks: {row['dataset']}/{row['id']}")
        by_rgb[rgb] = row
        rows.append(row)

    inventory = {}
    for name in RELEASES:
        capture_count = 0
        for row, archive in read_release(root, name):
            key = (name, row['id'])
            gate = framing[row['id']] if name == V2 else row['framing']
            if not gate['accepted'] or gate.get('rgb_sha256', row['files']['rgb.png']) != row['files']['rgb.png']:
                raise ValueError(f'Frame fails the whole-structure rule: {key}')
            if row['split'] == 'regression':
                role = 'regression'
                coverage = {}
                family = row['layout']
            else:
                eligibility = policy[key]
                if any(eligibility[k] != row[k] for k in ('palette', 'layout', 'split')):
                    raise ValueError(f'Eligibility and archive differ: {key}')
                if (eligibility['rgb_sha256'] != row['files']['rgb.png'] or
                        any(eligibility['reference_sha256'][r] != row['files'][r] for r in REFERENCE)):
                    raise ValueError(f'Source or reference hash differs from eligibility manifest: {key}')
                role = 'active' if eligibility['active'] else 'excluded'
                coverage = {k: eligibility[k] for k in ('visible_pixels', 'smooth_pixels')}
                family = row['family']
            entry = frame(row, archive, name, role)
            entry.update(layout=row['layout'], family=family, palette=row['palette'], angle=row['angle'],
                         framing=gate, **coverage)
            admit(entry)
            capture_count += 1
            if name == V2:
                base_v2[row['id']] = entry
        inventory[name] = dict(released=capture_count, manifest_sha256=PINNED[name])
    check_splits(rows)

    # V1 supplies aliases to the canonical, byte-identical V2 captures.
    repeated = 0
    excluded_v1 = []
    for row, archive in read_release(root, V1):
        parent = base_v2.get(row['id'])
        if parent is None:
            if framing[row['id']]['accepted']:
                raise ValueError(f"Admitted V1 capture is missing in V2: {row['id']}")
            excluded_v1.append(row['id'])
        elif parent['rgb_sha256'] == row['files']['rgb.png'] and same_reference(parent, row):
            repeated += 1
        else:
            raise ValueError(f"V1 and V2 source or reference differs: {row['id']}")
    inventory[V1] = dict(released=repeated + len(excluded_v1), aliases_of=V2, aliases=repeated,
                         rejected_closeups=sorted(excluded_v1), manifest_sha256=PINNED[V1])

    overlaps = 0
    for row, archive in read_release(root, COMPANION):
        existing = by_rgb.get(row['files']['rgb.png'])
        if existing:
            if not same_reference(existing, row):
                raise ValueError(f"Companion reference conflicts with canonical labels: {row['id']}")
            overlaps += 1
            continue
        entry = frame(row, archive, COMPANION, 'historical')
        entry.update(layout=row.get('benchmark_layout') or row.get('layout') or row['source_id'],
                     shape=row.get('benchmark_shape_sha256') or row.get('shape'),
                     cohort=row['benchmark_cohort'], label_source=row['label_source'])
        admit(entry)
    inventory[COMPANION] = dict(released=204, aliases=overlaps, manifest_sha256=PINNED[COMPANION])

    for name, (ticks, _) in LIGHTING.items():
        closeups = 0
        paired = 0
        for row, archive in read_release(root, name):
            if not row['whole_structure']:
                closeups += 1
                continue
            parent = base_v2[row['id']]
            if (row['noon_file_sha256']['rgb.png'] != parent['rgb_sha256'] or row['clock_ticks'] != ticks
                    or row['split'] != parent['split'] or row['files']['noon-camera.json'] != parent['files']['camera.json']
                    or any(row['files'][label] != parent['files'][label] for label in REFERENCE[1:])):
                raise ValueError(f"Lighting pair changes parent scene or visible labels: {name}/{row['id']}")
            role = 'excluded-lighting' if parent['role'] == 'excluded' else 'lighting'
            entry = frame(row, archive, name, role)
            entry.update(parent_dataset=V2, parent_id=parent['id'], clock_ticks=ticks,
                         visible_rgb_change_vs_noon=row['visible_rgb_change_vs_noon'])
            admit(entry)
            paired += 1
        inventory[name] = dict(released=paired + closeups, closeups=closeups, clock_ticks=ticks,
                               manifest_sha256=PINNED[name])

    fixtures = root / BACKGROUND / 'fixtures.json'
    for source in json.loads(fixtures.read_bytes())['cases']:
        parent = by_rgb[source['rgb_sha256']]
        if parent['role'] != 'active' or tuple(source['reference_sha256'][k] for k in REFERENCE) != tuple(
                parent['files'][k] for k in REFERENCE):
            raise ValueError(f"Background pair references a different or excluded capture: {source['id']}")
        for style in ('plain', 'sky'):
            variant = source['variants'][style]
            path = root / BACKGROUND / 'images' / source['id'] / f'{style}.png'
            if sha256(path) != variant['source_sha256']:
                raise ValueError(f'Fixture fingerprint changed: {path}')
            admit(dict(id=f"{source['id']}:{style}", dataset=BACKGROUND, split=parent['split'], role='background',
                       parent_dataset=parent['dataset'], parent_id=parent['id'], path=path.relative_to(root).as_posix(),
                       rgb_sha256=variant['source_sha256'], backdrop=style))
    inventory[BACKGROUND] = dict(native_anchors=6, derivative_images=12, kind='synthetic backgrounds')

    for row in rows:
        if row.get('palette') in RETIRED_PALETTES and row['role'] != 'excluded':
            raise ValueError(f"Retired palette cannot enter active work: {row['id']}")
    roles = Counter(row['role'] for row in rows)
    splits = Counter(row['split'] for row in rows if row['role'] == 'active')
    return dict(dataset='floating-combined-v1', roles=dict(roles), active_splits=dict(splits),
                sources=inventory, frames=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-root', required=True, type=Path)
    parser.add_argument('--out', default=MANIFEST, type=Path)
    args = parser.parse_args()
    manifest = build(args.archives_root)
    encoded = json.dumps(manifest, separators=(',', ':')).encode() + b'\n'
    args.out.write_bytes(gzip.compress(encoded, mtime=0))
    print(manifest['roles'], manifest['active_splits'])


if __name__ == '__main__':
    main()
