"""Time the two shared networks once per unique frozen screenshot without replacing cached fields."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.normal_fields import cached_fields
from engine.surface_fields import MODELS, SETTINGS, infer
from runtime.mode import image_only


@image_only()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    base = ROOT/'benchmark/reviews/normals-01'
    frames = {}
    for row in read(base/'cases.json'):
        sha = row['sha256']
        if sha in frames:
            continue
        source = base/row['source']
        if digest(source)!=sha:
            raise ValueError(source)
        cached = cached_fields(sha,args.fields)
        if cached is None:
            raise ValueError(f'No cached numerical reference for {sha}')
        frames[sha] = {}
        with Image.open(source) as src:
            image = Image.fromarray(np.asarray(src.convert('RGB')))
        for kind in MODELS:
            t = perf_counter()
            values = infer(image,kind)
            frames[sha][kind] = dict(seconds=perf_counter()-t,
                max_abs_difference=float(np.max(abs(values-cached[kind]))))
    save(args.out,dict(code=revision(),environment='onnxruntime==1.23.2, CPU on capture PC',settings=SETTINGS,
         first_call_includes_session_load=True,unique_images=len(frames),
         seconds={kind:sum(r[kind]['seconds'] for r in frames.values()) for kind in MODELS},per_image=frames))


if __name__=='__main__':
    main()
