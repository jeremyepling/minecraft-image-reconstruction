"""Prepare frozen image-only fields for the plane-depth comparison."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.texture_camera_eval import DATA
from engine.surface_fields import MODELS, SETTINGS, infer, session


def moge_z(image):
    image = image.convert('RGB')
    image.thumbnail((SETTINGS['normal_max_side'],)*2,Image.Resampling.LANCZOS)
    rgb = (np.asarray(image,np.float32)/255.).transpose(2,0,1)[None]
    env,filename,_ = MODELS['normal']
    path=Path(os.environ.get(env) or Path(__file__).resolve().parents[1]/'.deps'/filename)
    points, = session('normal',path).run(['points'],dict(image=rgb,num_tokens=np.array(SETTINGS['normal_tokens'],np.int64)))
    return np.ascontiguousarray(points[0,:,:,2])


def field(image, kind):
    root = os.environ.get('MCIR_DEPTH_EXPERIMENT_CACHE')
    suffix = '.npy' if kind=='inverse' else '-points.npy'
    if root:
        path = Path(root)/(hashlib.sha256(image.tobytes()).hexdigest()+suffix)
        if path.exists():
            return np.load(path,allow_pickle=False)
    array = infer(image,'depth') if kind=='inverse' else moge_z(image)
    if root:
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.npy')
        np.save(tmp,array)
        tmp.replace(path)
    return array


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['inverse','points'],required=True)
    args=p.parse_args()
    for split in ('train','validation','test','regression'):
        for f in sorted((DATA/'data'/split/'frames').glob('*/rgb.png')):
            with Image.open(f) as opened:
                field(opened.convert('RGB'),args.kind)
        print(split,flush=True)
