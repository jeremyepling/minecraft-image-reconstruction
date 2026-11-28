"""Produce the frozen Floating MoGe and Depth Anything fields on Eligible V1 RGB."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine import surface_fields


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent.parent / 'minecraft-image-reconstruction/runs/visible-builds-v1'
OUT = ROOT / 'runs/v1-fields'
V1_MANIFEST_SHA256 = 'a702f1d1a79c9e471680fd362993e14a4c2eeb20b4790d6b67b054b5f4bf0a56'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jobs():
    release = SOURCE / 'manifest.json'
    if digest(release) != V1_MANIFEST_SHA256:
        raise ValueError('Unexpected Visible Builds v1 release manifest')
    captures = json.loads(release.read_bytes())['frames']
    framing = json.loads((ROOT / 'training/framing-audit-v2.json').read_bytes())['frames']
    admitted = {row['id']: row['rgb_sha256'] for row in framing if row['accepted']}
    jobs = [row for row in captures if row['id'] in admitted and row['palette'] != 'orchid']
    if len(jobs) != 144 or any(admitted[row['id']] != row['rgb_sha256'] for row in jobs):
        raise ValueError('Unexpected V1 framing audit or source fingerprints')
    return jobs


def produce(frame):
    image = SOURCE / 'data' / frame['split'] / 'frames' / frame['id'] / 'rgb.png'
    source_sha = digest(image)
    if source_sha != frame['rgb_sha256']:
        raise ValueError(f'Unexpected V1 screenshot: {image}')
    directory = OUT / frame['id']
    directory.mkdir(parents=True, exist_ok=True)
    existing = directory / 'receipt.json'
    model_sha = {kind: spec[2] for kind, spec in surface_fields.MODELS.items()}
    if existing.is_file():
        receipt = json.loads(existing.read_bytes())
        if (receipt['source_sha256'] == source_sha and receipt['model_sha256'] == model_sha
                and receipt['preprocessing'] == surface_fields.SETTINGS
                and all(digest(directory / f'{kind}.npz') == sha for kind, sha in receipt['fields_sha256'].items())):
            return frame['id']
    fingerprints = {}
    with Image.open(image) as rgb:
        for kind in model_sha:
            value = surface_fields.infer(rgb, kind)
            target = directory / f'{kind}.npz'
            temporary = target.with_suffix('.part')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, value=value)
            temporary.replace(target)
            fingerprints[kind] = digest(target)
    receipt = dict(source_sha256=source_sha, model_sha256=model_sha,
                   fields_sha256=fingerprints, preprocessing=surface_fields.SETTINGS)
    temporary = existing.with_suffix('.part')
    temporary.write_text(json.dumps(receipt) + '\n', encoding='utf-8')
    temporary.replace(existing)
    return frame['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        for frame_id in pool.map(produce, jobs()):
            print(frame_id, flush=True)


if __name__ == '__main__':
    main()
