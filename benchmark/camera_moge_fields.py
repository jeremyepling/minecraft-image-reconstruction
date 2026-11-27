"""Full MoGe-2 ONNX point map and centered intrinsics from one source screenshot."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, SOURCE, COHORTS, WORKTREES, digest, frames, read, save
from benchmark.moge_probe import SHA256, REVISION

RUN = ROOT / 'runs/camera-moge-surface-first-01'
BASE = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01'
CHECKPOINT = WORKTREES / 'normal-round/.deps/moge-2-vits.onnx'
MAX_SIDE = 448
TOKENS = 600


@lru_cache
def session():
    sys.path.append(str(WORKTREES / 'surface-round/runs/depth-probe/runtime'))
    import onnxruntime as ort
    if digest(CHECKPOINT) != SHA256:
        raise ValueError('MoGe-2 checkpoint checksum mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(CHECKPOINT), options, providers=['CPUExecutionProvider'])


def intrinsics(points, mask, size):
    # f*X = u*(Z + shift), f*Y = v*(Z + shift).  u,v in native centered pixels.
    # IRLS in projected pixel space reduces leverage of distant points near the horizon.
    h,w = mask.shape
    y,x = np.where(mask[::4,::4])
    y,x = 4*y,4*x
    if len(y)<16:
        raise ValueError('insufficient model-valid points')
    points = points[y,x].astype(float)
    uv = (np.column_stack(((x+.5)/w,(y+.5)/h))-.5)*size
    A = np.stack((points[:,:2], -uv), axis=2)
    B = uv*points[:,2,None]
    weights = np.ones(len(y))
    for _ in range(6):
        sol = np.linalg.lstsq((A*weights[:,None,None]).reshape(-1,2),
                              (B*weights[:,None]).ravel(),rcond=None)[0]
        depth = points[:,2]+sol[1]
        residual = np.linalg.norm(sol[0]*points[:,:2]/np.maximum(.001,depth[:,None])-uv,axis=1)
        weights = (depth>.001)*np.minimum(1.,np.sqrt(4/np.maximum(.01,residual)))/np.maximum(.01,depth)
    return sol


def make_field(entry):
    out = RUN / entry['dataset'] / entry['id']
    dest = out/'field.npz'
    if dest.exists():
        return
    source = Path(entry['frame'])/'rgb.png'
    if digest(source) != entry['rgb_sha256']:
        raise ValueError(source)
    with Image.open(source) as im:
        image = im.convert('RGB')
    size = image.size
    image.thumbnail((MAX_SIDE,MAX_SIDE),Image.Resampling.LANCZOS)
    batch = (np.asarray(image,dtype='f')/255.).transpose(2,0,1)[None]
    pts,norm,mask,scale = session().run(['points','normal','mask','scale'],
                                      dict(image=batch,num_tokens=np.array(TOKENS,dtype=np.int64)))
    mask = mask[0]>.5
    focal,shift = intrinsics(pts[0],mask,size)
    out.mkdir(parents=True,exist_ok=True)
    temp = out/'field.part.npz'
    np.savez_compressed(temp,points=pts[0].astype('f'),normal=norm[0].astype('f2'),mask=mask,
                        model_scale=scale, focal_px=focal,z_shift=shift,native_size=size)
    temp.replace(dest)
    save(out/'field.json',dict(source_sha256=entry['rgb_sha256'],checkpoint_sha256=SHA256,revision=REVISION,
         native_size=size, resized_size=image.size, crop=None, tokens=TOKENS, frame='camera +X right, +Y down, +Z forward',
         points='raw affine-invariant points; add z_shift to Z before projection',principal_point='native image center',
         native_xy='(field pixel index + .5) * native_size / resized_size',focal_px=float(focal),z_shift=float(shift)))


def failures():
    entries = [e for e in frames(SOURCE,COHORTS)
               if read(BASE/e['dataset']/e['id']/'build-manifest.json')['placement']['P'] is None]
    return sorted(entries,key=lambda e:(e['split']!='train',e['id']))


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,_ in enumerate(pool.map(make_field,failures()),1):
            if i%10==0:
                print(i,flush=True)
