"""Compute Apple Depth Pro boundary evidence from RGB, with source-pixel sampling."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from runtime.mode import image_only


SOURCE = 'https://github.com/apple/ml-depth-pro'
SOURCE_REV = '9e65e4dbe9568d23c546fcec53302b10445e109e'
WEIGHTS_URL = 'https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt'
WEIGHTS_SHA = '3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
SETTINGS = dict(precision='fp16 on CUDA, fp32 on CPU', encoder_side=1536, focal='inferred from RGB',
                resize='official infer: bilinear to 1536 square and back, align_corners=False, no padding',
                field='1 / predicted native depth, float32', grid_stride=4, grid_first_xy=[2,2],
                grid_size='floor(H/4) rows by floor(W/4) columns')


def load_model(source, checkpoint, device='cuda'):
    """Initialize the frozen Apple network. The optional upstream HEIF reader is unused for RGB."""
    import torch
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_REV:
        raise ValueError('Apple Depth Pro source revision differs from the frozen run')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no']):
        raise ValueError('Apple Depth Pro tracked source is modified')
    if digest(checkpoint)!=WEIGHTS_SHA:
        raise ValueError('Apple Depth Pro checkpoint differs from the published weights')
    # Load the official model module and its relative network imports. Its package initializer
    # also registers an HEIF/EXIF reader, whereas this interface accepts already decoded pixels.
    folder = source/'src/depth_pro'
    spec = importlib.util.spec_from_file_location('_apple_depth_pro_model',folder/'depth_pro.py',
                                                  submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = replace(module.DEFAULT_MONODEPTH_CONFIG_DICT,checkpoint_uri=str(checkpoint))
    target = torch.device(device)
    precision = torch.float16 if target.type=='cuda' else torch.float32
    model,transform = module.create_model_and_transforms(config,device=target,precision=precision)
    return model.eval(),transform


def sample_inverse_depth(native):
    """Keep the exact sensor pixel at (4x+2,4y+2) in every complete 4x4 cell."""
    height,width = native.shape
    if min(height,width)<4:
        raise ValueError('A depth field needs at least one 4x4 sample')
    samples = native[2:(height//4)*4:4,2:(width//4)*4:4]
    if not np.isfinite(samples).all() or np.any(samples<=0):
        raise ValueError('Depth Pro returned nonfinite or nonpositive depth')
    return np.reciprocal(samples,dtype=np.float32)


def predict_field(image, model, transform):
    """Return only a source-aligned numerical field. No known camera is supplied."""
    import torch
    with torch.inference_mode():
        depth = model.infer(transform(image.convert('RGB')),f_px=None)['depth'].float().cpu().numpy()
    if depth.shape!=(image.height,image.width):
        raise ValueError('Depth Pro did not restore the native screenshot dimensions')
    return sample_inverse_depth(depth)


def write_field(dest, field):
    np.savez_compressed(dest,inverse_depth=field)
    return digest(dest)


@image_only()
def run(out, source, checkpoint, device):
    from importlib.metadata import version
    import torch
    source_run = ROOT/'benchmark/reviews/recovery-summary-01/manifest.json'
    manifest = read(source_run)
    index = ROOT/'benchmark/reviews/normals-01/cases.json'
    images = {r['key']:index.parent/r['source'] for r in read(index)}
    frames = manifest['frames']
    if len(frames)!=56 or len(images)!=56 or set(images)!={r['key'] for r in frames}:
        raise ValueError('The frozen screenshot inventory changed')
    out.mkdir(parents=True,exist_ok=False)
    (out/'fields').mkdir()
    torch.set_num_threads(4)
    start = perf_counter()
    model,transform = load_model(source,checkpoint,device)
    load_time = perf_counter()-start
    results,artifacts = {},{}
    for row in frames:
        key,sha = row['key'],row['sha256']
        if digest(images[key])!=sha:
            raise ValueError(f'Screenshot changed for {key}')
        if sha in results:
            continue
        with Image.open(images[key]) as opened:
            rgb = opened.convert('RGB')
        start = perf_counter()
        field = predict_field(rgb,model,transform)
        elapsed = perf_counter()-start
        path = 'fields/'+sha+'.npz'
        artifacts[path]=write_field(out/path,field)
        results[sha]=dict(image_size=list(rgb.size),grid_shape=list(field.shape),
                          field_sha256=hashlib.sha256(field.tobytes()).hexdigest(),file=path,
                          preprocess_inference_and_sampling_seconds=elapsed)
        print(f'{len(results):02d}  {key}  {elapsed:.3f}s',flush=True)
    if len(results)!=52:
        raise ValueError('Expected 52 distinct images on the 56-frame suite')
    save(out/'manifest.json',dict(run_id='depth-pro-fields-01',code=revision(),source_url=SOURCE,
        source_revision=SOURCE_REV,checkpoint_url=WEIGHTS_URL,checkpoint_sha256=WEIGHTS_SHA,
        checkpoint_verification='https://huggingface.co/apple/DepthPro/blob/c9e4772/depth_pro.pt',
        frame_index_sha256=digest(index),source_run_sha256=digest(source_run),
        fixtures=manifest['fixtures'],version=manifest['version'],frames=frames,settings=SETTINGS,
        device=device,hardware=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU',
        packages={p:version(p) for p in ('torch','torchvision','timm','numpy','Pillow')},cuda=torch.version.cuda,
        timing=dict(load_and_checkpoint_validation_seconds=load_time,
                    unique_rgb_inference_and_sampling_seconds=sum(r['preprocess_inference_and_sampling_seconds'] for r in results.values()),
                    scope='Synchronized through host copy of each native map; excludes encoding, scoring and repeated captures.'),
        fields=results,artifacts=artifacts,runner_sha256=digest(Path(__file__)),
        reproduce='python -m benchmark.depth_pro_fields --source PATH_TO_APPLE_CHECKOUT --checkpoint PATH_TO_WEIGHTS --out NEW_DIRECTORY'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'.deps/ml-depth-pro')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'.deps/depth_pro.pt')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/('depth-pro-'+uuid4().hex[:12]))
    parser.add_argument('--image',type=Path,help='Single RGB image; --out must name a new .npz file.')
    args=parser.parse_args()
    if args.image is None:
        run(args.out,args.source,args.checkpoint,args.device)
    else:
        if args.out.suffix!='.npz' or args.out.exists():
            parser.error('A single image needs a new --out FILE.npz')
        model,transform=load_model(args.source,args.checkpoint,args.device)
        with Image.open(args.image) as image:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            write_field(args.out,predict_field(image,model,transform))
