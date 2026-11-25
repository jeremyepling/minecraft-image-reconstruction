"""Save the frozen Depth Anything V2 values on the exact wall-voting lattice."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.normal_fields import cached_fields
from benchmark.normal_round import groups
from engine.surface_fields import MODELS, SETTINGS, infer
from runtime.mode import image_only


@image_only()
def prepare(cache,out):
    from engine.single_wall_gate import depth_grid
    prior=ROOT/'benchmark/reviews/perimeters-01'
    manifest=read(prior/'manifest.json')
    if digest(prior/'cases.json')!=manifest['artifacts']['cases.json']:
        raise ValueError('The frozen source for Depth Anything has changed')
    expected={r['sha256']:r['numerical_depth_sha256'] for r in read(prior/'cases.json')}
    cohorts,_,fixtures=groups()
    if fixtures!=manifest['fixtures']:
        raise ValueError('Fixture set differs from the depth run')
    (out/'fields').mkdir(parents=True,exist_ok=False)
    files,frames={},{}
    for cases in cohorts.values():
        for case in cases:
            sha=digest(case.path('image'))
            frames[f'{case.floor}/{case.id}']=sha
            name='fields/'+sha+'.npz'
            if name in files:
                continue
            with Image.open(case.path('image')) as src:
                image=src.convert('RGB')
            fields=cached_fields(sha,cache,kinds=('depth',))
            depth=fields['depth'] if fields is not None else infer(image,'depth')
            if hashlib.sha256(depth.tobytes()).hexdigest()!=expected[sha]:
                raise ValueError(f'Depth Anything field differs from the frozen source for {case.id}')
            np.savez_compressed(out/name,inverse_depth=depth_grid(depth,image.size))
            files[name]=digest(out/name)
    save(out/'manifest.json',dict(code=revision(),version='26.3',fixtures=fixtures,frames=frames,
         model=MODELS['depth'],inference_settings=SETTINGS,source_manifest=digest(prior/'manifest.json'),
         sampling='Native four-pixel lattice, [2::4,2::4]. Shared gate adapter applied to original inverse-depth field.',
         adapter_sha256=digest(ROOT/'engine/single_wall_gate.py'),artifacts=files,
         replay='python -m benchmark.single_wall_fields --out runs/single-wall-v2-fields'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=ROOT.parent/'normal-round/runs/normal-fields')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    prepare(args.cache,args.out)
