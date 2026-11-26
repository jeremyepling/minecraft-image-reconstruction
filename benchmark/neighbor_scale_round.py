"""Replay the saved placement experiment changing only metric neighbor transfer."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.camera_round import read, digest
from benchmark.mask_blocks import SETTINGS, _footprints, _carriers, _blocks, _write_read, registered, _score
from benchmark.neighbor_scale import transfer, SETTINGS as TRANSFER
from engine.camera_grid import predict
from engine.masked_count import count
from engine.surface_sample import sample_surface


def run(job):
    case, suite, frames, out, hashes, baseline = job
    key = case['key']
    folder = frames/('legacy-'+key.replace('/','-'))
    pred = suite/case['predictions']
    assert digest(pred)==hashes[case['predictions']] and digest(folder/'rgb.png')==case['sha256']
    parts = json.loads(gzip.decompress(pred.read_bytes()))['gap']
    with Image.open(folder/'rgb.png') as source:
        im = source.convert('RGB')
    camera = predict(im,'26.3',SETTINGS['camera'])
    P = None if camera['P'] is None else np.asarray(camera['P'])
    masks = _footprints(parts,im.size,True)
    inputs = []
    if P is not None:
        for part in parts:
            sample = sample_surface(im,part)
            ns = count(sample)['counts']
            inputs.append(dict(id=part['id'],quad=sample['quad'] if sample else None,
                               pitch=dict(counts=ns if all(v is not None for v in ns) else None)))
    carriers = _carriers(im,inputs,masks,P)
    carriers, log = transfer(im,inputs,masks,carriers,P)
    blocks,stages,_ = _blocks(masks,carriers,P,{})
    path = out/'packs'/(key+'.zip')
    path.parent.mkdir(exist_ok=True,parents=True)
    emitted = _write_read(path,blocks)
    pose = read(folder/'camera.json')
    aligned = registered(emitted, registration(P,pose) if P is not None else np.eye(4))
    labels = read(folder/'labels.json')
    target = {tuple(b['xyz_world']) for b in labels['blocks']}
    old = set(map(tuple,baseline))
    changed = dict(gained=len((aligned-old)&target),lost=len((old-aligned)&target),
                   added_extra=len(aligned-old-target),removed_extra=len(old-aligned-target))
    return dict(key=key,split=case['cohort'],score=_score(aligned,target),delta=changed,
                transferred=log,stages=stages,coords=sorted(aligned),image_sha256=case['sha256'],
                mask_sha256=hashes[case['predictions']],labels_sha256=digest(folder/'labels.json'),
                pose_sha256=digest(folder/'camera.json'),pack_sha256=digest(path))


def main():
    p = argparse.ArgumentParser()
    for k in ['suite','frames','baseline','out']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    cases = read(args.suite/'cases.json')
    hashes = read(args.suite/'manifest.json')['artifacts']
    baseline = read(args.baseline)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run,[(r,args.suite,args.frames,args.out,hashes,baseline[r['key']]) for r in cases]))
    coords = {r['key']:r.pop('coords') for r in rows}
    def totals(field):
        return dict(sum((Counter(r[field]) for r in rows),Counter()))
    report = dict(frames=len(rows),producing=sum(bool(r['score']['serialized']) for r in rows),
                  score=totals('score'),delta=totals('delta'),stages=totals('stages'),
                  transfers=sum(len(r['transferred']) for r in rows),method=TRANSFER,
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  baseline_sha256=digest(args.baseline),suite_sha256=digest(args.suite/'manifest.json'))
    (args.out/'coords.json').write_text(json.dumps(coords))
    (args.out/'results.json').write_text(json.dumps(dict(report,results=rows),indent=2))
    print(report)

if __name__=='__main__':
    main()
