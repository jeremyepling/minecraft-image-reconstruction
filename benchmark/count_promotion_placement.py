"""Measure the placement effect of the approved count fallback on the same detected outlines."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_blocks import registered
from benchmark.texture_count_round import FINDERS
from benchmark.wall_join_test import checked, read_gzip
from engine.visible_placement import place
from runtime.mode import image_only, mark_image_only

COUNTS = ROOT/'benchmark/reviews/phase-corners-01'


def replay(row, frames, outlines, hashes):
    key = row['key']
    folder = frames/('legacy-'+key.replace('/','-'))
    with Image.open(checked(folder/'rgb.png',row['sha256'])) as f:
        photo = f.convert('RGB')
    source,method = FINDERS['photo']
    path = row['predictions']
    parts = read_gzip(checked(source/path,outlines[path]))[method]
    predictions = read_gzip(checked(COUNTS/path,hashes[path]))
    prior, promoted = [], []
    for part, record in zip(parts,predictions,strict=True):
        if record['proposal'] != part['id']:
            raise ValueError('Mismatched proposal')
        base = dict(id=part['id'],quad=record['chart'],counts=record['methods']['current']['counts'])
        prior.append(base)
        selected = record['methods']['phase']['evidence'].get('self_repeat',base)
        promoted.append(dict(id=part['id'],quad=selected['quad'],counts=selected['counts']))
    old,status = place(photo,'26.3',parts,counts=dict(surfaces=prior))
    new = place(photo,'26.3',parts,counts=dict(surfaces=promoted))[0] if prior!=promoted and status['camera_status']=='supported' else old
    pose = read(folder/'camera.json')
    truth = {tuple(cell['xyz_world']) for cell in read(folder/'labels.json')['blocks']}
    if old or new:
        reg = registration(np.asarray(status['P']),pose)
        old,new = registered(old,reg),registered(new,reg)
    return dict(key=key,image_sha256=row['sha256'],camera=status['camera_status'],counts_changed=prior!=promoted,
                prior_correct=len(old&truth),prior_extra=len(old-truth),new_correct=len(new&truth),new_extra=len(new-truth),
                correct_gained=len((new-old)&truth),correct_lost=len((old-new)&truth),
                extras_added=len((new-old)-truth),extras_removed=len((old-new)-truth))


@image_only()
def run(frames,out,workers):
    m=read(COUNTS/'manifest.json')
    source,_=FINDERS['photo']
    outlines=read(checked(source/'manifest.json',m['source_manifest_sha256']))
    cases=read(checked(COUNTS/'cases.json',m['artifacts']['cases.json']))
    job=partial(replay,frames=frames,outlines=outlines['artifacts'],hashes=m['artifacts'])
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        rows=list(pool.map(job,cases))
    counts=Counter()
    for r in rows:
        counts.update({k:v for k,v in r.items() if isinstance(v,int)})
    out.parent.mkdir(exist_ok=True,parents=True)
    save(out,dict(code=revision(),counter_manifest_sha256=digest(COUNTS/'manifest.json'),frames=len(rows),
                  scope='Offline local-coordinate placements registered for scoring. Both inputs use identical photo-refined surfaces and camera code.',
                  totals=dict(counts),cases=rows))
    print(counts)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--frames',type=Path,default=ROOT/'runs/visible-builds-v1/data/regression/frames')
    cli.add_argument('--out',required=True,type=Path)
    cli.add_argument('--workers',type=int,default=4)
    args=cli.parse_args()
    run(args.frames,args.out,args.workers)
