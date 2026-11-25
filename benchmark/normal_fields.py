"""Cache image-only maps by frame digest; inference reads no fixtures or labels."""
import argparse
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import lru_cache

import numpy as np
from PIL import Image
from benchmark.camera_round import ROOT, digest, read
from engine.surface_fields import MODELS, SETTINGS, infer


@lru_cache(maxsize=1)
def inference_signatures():
    """Write the LF fingerprint and recognize equivalent older LF/CRLF worktrees."""
    import PIL
    try:
        runtime = version('onnxruntime')
    except PackageNotFoundError:
        runtime = '1.23.2'  # Allows reading the published suite's cache without an inference runtime.
    candidates = []
    for path in ('engine/surface_fields.py', 'benchmark/depth_probe.py'):
        lf = (ROOT / path).read_bytes().replace(b'\r\n', b'\n')
        candidates.append([hashlib.sha256(data).hexdigest() for data in (lf, lf.replace(b'\n', b'\r\n'))])
    signatures = []
    for code in product(*candidates):
        recipe = [code, MODELS, SETTINGS, runtime, PIL.__version__, np.__version__]
        signatures.append(hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest())
    return tuple(signatures)


def cached_fields(sha, cache, kinds=('normal', 'depth')):
    """Read only the requested fields, leaving the cache untouched on misses."""
    path = cache / (sha + '.npz')
    if path.is_file():
        with np.load(path, allow_pickle=False) as old:
            signature = str(old['signature'])
            if any(signature == sha + ':' + s for s in inference_signatures()):
                return {kind: old[kind] for kind in kinds}
    return None


def fields(source,sha,cache):
    if digest(source)!=sha:
        raise ValueError(f'Source digest mismatch: {source}')
    cache.mkdir(parents=True,exist_ok=True)
    file=cache/(sha+'.npz')
    fingerprint=sha+':'+inference_signatures()[0]
    ready = cached_fields(sha,cache)
    if ready is not None:
        return ready['normal'],ready['depth']
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
