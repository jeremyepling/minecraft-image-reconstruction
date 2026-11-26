"""Score a frozen RGB sky detector against visible buildings in the 72 training views."""
import argparse
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from engine.sky_filter import sky_masks


def run(archive, out):
    shard = read(ROOT/'training/visible-builds-v1.json')['shards']['train']
    if digest(archive) != shard['sha256']:
        raise ValueError('Expected the original Visible Builds v1 training archive')
    rows = []
    totals = Counter()
    with ZipFile(archive) as z:
        for line in z.read('frames.jsonl').splitlines():
            key = json.loads(line)['directory']
            with Image.open(io.BytesIO(z.read(key+'/rgb.png'))) as photo:
                blue, cloud = sky_masks(photo.convert('RGB'))
            with np.load(io.BytesIO(z.read(key+'/labels.npz')), allow_pickle=False) as labels:
                building = labels['face_id'] > 0
            if blue.shape != building.shape or cloud.shape != building.shape:
                raise ValueError(f'Misaligned prediction for {key}')
            metrics = dict(building_pixels=int(building.sum()), blue_on_building=int((blue & building).sum()),
                           cloud_on_building=int((cloud & building).sum()), sky_marked=int((cloud | blue).sum()))
            rows.append(dict(frame=key, **metrics))
            totals.update(metrics)
    if len(rows) != shard['frames']:
        raise ValueError('Incomplete training shard')
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, dict(code=revision(), shard=shard, frames=rows, totals=dict(totals),
                   scope='RGB detector only. Recorded building silhouettes have pixel-center rasterization tolerance.'))
    print(dict(totals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'runs/visible-builds-v1/visible-builds-v1-train.zip')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.archive, args.out)
