"""Cache image-only maps by frame digest; inference reads no fixtures or labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import lru_cache

import numpy as np
from PIL import Image
from benchmark.camera_round import ROOT, digest, read
from engine.surface_fields import MODELS, SETTINGS, infer


@lru_cache(maxsize=1)
def inference_signature():
    import onnxruntime, PIL
    code=[digest(ROOT/p) for p in ('engine/surface_fields.py','benchmark/depth_probe.py')]
    return hashlib.sha256(json.dumps([code,MODELS,SETTINGS,onnxruntime.__version__,PIL.__version__,np.__version__],
                            sort_keys=True).encode()).hexdigest()


def fields(source,sha,cache):
    if digest(source)!=sha:
        raise ValueError(f'Source digest mismatch: {source}')
    cache.mkdir(parents=True,exist_ok=True)
    file=cache/(sha+'.npz')
    fingerprint=sha+':'+inference_signature()
    if file.exists():
        with np.load(file,allow_pickle=False) as old:
            if str(old['signature'])==fingerprint:
                return old['normal'],old['depth']
    with Image.open(source) as src:
        im=Image.fromarray(np.asarray(src.convert('RGB')))
    normal,depth=(infer(im,k) for k in ('normal','depth'))
    with NamedTemporaryFile(dir=cache,suffix='.tmp',delete=False) as tmp:
        name=tmp.name
        np.savez(tmp,normal=normal,depth=depth,signature=fingerprint)
    os.replace(name,file)
    return normal,depth


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=ROOT/'runs/normal-fields')
    ap.add_argument('--dev-only',action='store_true')
    args=ap.parse_args()
    previous=ROOT/'benchmark/reviews/surfaces-01'
    for row in read(previous/'cases.json'):
        if not args.dev_only or row['cohort']=='dev':
            fields(previous/row['source'],row['sha256'],args.cache)
            print(row['id'],row['floor'],flush=True)
