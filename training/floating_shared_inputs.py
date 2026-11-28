"""Stage and resolve the two active Floating cohorts from pinned capture releases."""

import argparse
from collections import defaultdict
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from training.floating_combined import MANIFEST, PINNED, REFERENCE, load, read_release, sha256


COHORTS = {
    'visible': ('visible-builds-v2', 'visible-builds-v3'),
    'decorative': ('decorative-builds-v1',),
}
CAPTURE_FILES = ('rgb.png', *REFERENCE)
RECEIPT = 'sha256.json'


def active_rows():
    return [row for row in load()['frames'] if row['role'] == 'active']


def hashes(row):
    return {name: row['files'][name] for name in CAPTURE_FILES}


def read(path):
    return json.loads(path.read_bytes())


def write_read_only(path, payload):
    path.write_bytes(payload)
    path.chmod(0o444)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def install(root, archives, dataset):
    """Publish one complete verified dataset directory in a single rename."""
    wanted = {row['id']: row for row in active_rows() if row['dataset'] == dataset}
    if not wanted:
        raise ValueError(f'No active Floating views in {dataset}')
    source = defaultdict(list)
    for record, archive in read_release(archives, dataset):
        if row := wanted.get(record['id']):
            if (row['archive'] != archive or row['member'] != record['directory']
                    or hashes(row) != hashes(record)):
                raise ValueError(f'Capture differs from the Combined V1 index: {dataset}/{row["id"]}')
            source[archive].append((row, record))
    if {row['id'] for group in source.values() for row, _ in group} != wanted.keys():
        raise ValueError(f'Missing active views in {dataset}')

    target = root / 'frames' / dataset
    if target.exists():
        raise FileExistsError(f'Fixture dataset already installed: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'{dataset}-', dir=target.parent) as temp:
        staging = Path(temp) / dataset
        staging.mkdir()
        frames = {}
        for archive, views in source.items():
            with ZipFile(archives / archive) as shard:
                for row, record in views:
                    frame = staging / row['id']
                    frame.mkdir()
                    files = hashes(row)
                    for name, expected in files.items():
                        payload = shard.read(f'{row["member"]}/{name}')
                        if hash_bytes(payload).hexdigest() != expected:
                            raise ValueError(f'Capture file checksum mismatch: {dataset}/{row["id"]}/{name}')
                        write_read_only(frame / name, payload)
                    # The scorer needs the published visible counts and the Combined framing decision.
                    metadata = dict(id=row['id'], files=files, framing=row['framing'],
                                    counts=record['counts'], ground_y=record['ground_y'])
                    payload = json_bytes(metadata)
                    write_read_only(frame / 'frame.json', payload)
                    frames[row['id']] = hash_bytes(payload).hexdigest()
        receipt = dict(dataset=dataset, release_sha256=PINNED[dataset], index_sha256=sha256(MANIFEST), frames=frames)
        write_read_only(staging / RECEIPT, json_bytes(receipt))
        staging.rename(target)
    return len(wanted)


def cases(root, *, cohort='visible', ids=None):
    datasets = COHORTS[cohort]
    root = root.resolve()
    rows = [row for row in active_rows() if row['dataset'] in datasets]
    if ids:
        requested = set(ids)
        found = {row['id'] for row in rows}
        if unknown := requested - found:
            raise ValueError(f'Unknown {cohort} views: {", ".join(sorted(unknown))}')
        rows = [row for row in rows if row['id'] in requested]
    index_sha = sha256(MANIFEST)
    receipts = {}
    result = []
    for row in rows:
        dataset = row['dataset']
        directory = root / 'frames' / dataset
        if dataset not in receipts:
            receipt = directory / RECEIPT
            if not receipt.is_file():
                raise FileNotFoundError(f'Missing pinned {dataset} fixture at {directory}')
            saved = read(receipt)
            if (saved['dataset'] != dataset or saved['release_sha256'] != PINNED[dataset]
                    or saved['index_sha256'] != index_sha):
                raise ValueError(f'Fixture provenance changed: {receipt}')
            receipts[dataset] = saved['frames']
        frame = directory / row['id']
        metadata = frame / 'frame.json'
        if sha256(metadata) != receipts[dataset].get(row['id']) or read(metadata)['files'] != hashes(row):
            raise ValueError(f'Fixture frame metadata changed: {metadata}')
        for name in CAPTURE_FILES:
            if not (frame / name).is_file():
                raise FileNotFoundError(f'Missing pinned capture file: {frame / name}')
        case = {key: row[key] for key in ('id', 'dataset', 'split', 'layout', 'family', 'palette', 'rgb_sha256')}
        case.update(frame=str(frame), dataset_manifest_sha256=PINNED[dataset],
                    reference_sha256={key: row['files'][key] for key in REFERENCE},
                    variants={'original': dict(image=str(frame / 'rgb.png'), source_sha256=row['rgb_sha256'])})
        result.append(case)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-root', type=Path, required=True)
    parser.add_argument('--fixture-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=tuple(name for group in COHORTS.values() for name in group),
                        required=True)
    args = parser.parse_args()
    count = install(args.fixture_root, args.archives_root, args.dataset)
    print(f'Installed {count} read-only {args.dataset} views in {args.fixture_root}')


if __name__ == '__main__':
    main()
