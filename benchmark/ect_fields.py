"""Generate the numerical ECT edge-cause fields used by the cached boundary experiment."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.camera_round import digest, read, save
from benchmark.wall_join_test import capture_index
from runtime.mode import image_only


SOURCE_REV = 'd9d50ad2d2573384d2640d1d02da289079b70b57'
TIMM_REV = '5b28ef410062af2a2ab1f27bf02cf33e2ba28ca2'
WEIGHTS = '25129414417674aeb424ad7c7167fead308937b0e8c80786287edf91b055288c'
WEIGHT_URL = 'https://drive.google.com/file/d/10b8fqd2cea-Ca2dPYXSkD_mLmgWxK5si/view'
CAUSES = ('depth', 'normal', 'reflectance', 'illumination')
SIZE = (768, 480)


def verify_checkout(root, revision):
    head = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip()
    if head != revision or subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD','--']).returncode != 0:
        raise ValueError(f'Expected unmodified upstream source {revision} at {root}')


def load_network(source, timm_source, weights):
    import torch
    verify_checkout(source, SOURCE_REV)
    verify_checkout(timm_source, TIMM_REV)
    if digest(weights) != WEIGHTS:
        raise ValueError('ECT checkpoint checksum mismatch')
    sys.path.insert(0,str(timm_source))
    import timm
    if Path(timm.__file__).resolve().parents[1] != timm_source.resolve():
        raise ValueError('Expected timm 0.4.5 from the pinned source checkout')
    # The research files import an interactive shell and logger, unused by their inference path.
    # Make a temporary inference copy to avoid those training/debug dependencies.
    skip = {'from IPython import embed','from loguru import logger','from turtle import forward'}
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)/'model'
        folder.mkdir()
        for name in ('edge_model','base_model','blocks','vit','decoder'):
            body = (source/'model'/(name+'.py')).read_text(encoding='utf8')
            (folder/(name+'.py')).write_text('\n'.join(line for line in body.splitlines() if line.strip() not in skip), encoding='utf8')
        sys.path.insert(0,scratch)
        try:
            from model.edge_model import EdgeCerberus
            create = timm.create_model
            # The complete released checkpoint overwrites the initial ImageNet weights.
            with patch.object(timm,'create_model',lambda name, **kw: create(name, **dict(kw, pretrained=False))):
                network = EdgeCerberus(backbone='vitb_rn50_384')
        finally:
            sys.path.remove(scratch)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    network.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()}, strict=True)
    return network.eval().to('cuda')


def infer(network, image):
    import torch
    from torch.nn.functional import interpolate
    from torchvision.transforms.functional import normalize, to_tensor
    if image.width*SIZE[1] != image.height*SIZE[0]:
        raise ValueError('Frozen source aspect ratio differs')
    image_tensor = to_tensor(image.resize(SIZE,Image.Resampling.BILINEAR)).to('cuda')
    image_tensor = normalize(image_tensor, [0.485,0.456,0.406], [0.229,0.224,0.225])
    with torch.inference_mode():
        outputs = network(image_tensor[None])
        if len(outputs) != 5 or any(tuple(o.shape[-2:]) != SIZE[::-1] for o in outputs):
            raise ValueError('Unexpected ECT output geometry')
        features = torch.cat(outputs[1:],dim=1)
        aligned = interpolate(features, size=(image.height//4,image.width//4),mode='bilinear',align_corners=False)
        if not torch.isfinite(aligned).all() or aligned.min()<0 or aligned.max()>1:
            raise ValueError('ECT returned invalid cause probabilities')
        array = (aligned[0]*255).round().byte().cpu().numpy()
    return dict(zip(CAUSES, array, strict=True))


@image_only()
def run(args):
    import torch
    start = perf_counter()
    network = load_network(args.upstream.resolve(),args.timm_source.resolve(),args.weights)
    load_s = perf_counter()-start
    captures = capture_index(args.source_root/'benchmark/data')
    base = args.source_root/'benchmark/reviews/single-wall-depth-01'
    manifest = read(base/'manifest.json')
    if digest(base/'cases.json') != manifest['artifacts']['cases.json']:
        raise ValueError('Changed baseline roster')
    rows = read(base/'cases.json')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'fields').mkdir(exist_ok=True)
    artifacts, timings = {}, {}
    for row in rows:
        key, sha = row['key'], row['sha256']
        rel = 'fields/'+sha+'.npz'
        if rel in artifacts:
            continue
        path = captures[key].path('image')
        if digest(path) != sha:
            raise ValueError(f'Wrong screenshot for {key}')
        with Image.open(path) as opened:
            rgb = opened.convert('RGB')
        start = perf_counter()
        data = infer(network,rgb)
        timings[sha] = perf_counter()-start
        np.savez_compressed(args.out/rel,**data)
        artifacts[rel] = digest(args.out/rel)
        if len(artifacts)==1 or len(artifacts)%10==0:
            print(len(artifacts), key, {k: (int(v.min()),int(v.max())) for k,v in data.items()}, flush=True)
    save(args.out/'manifest.json',dict(run_id=args.out.name, upstream='https://github.com/Daniellli/ECT',
         code_revision=SOURCE_REV, timm_revision=TIMM_REV, weights_url=WEIGHT_URL, checkpoint_sha256=WEIGHTS,
         torch=torch.__version__, device=torch.cuda.get_device_name(), version=manifest['version'], fixtures=manifest['fixtures'],
         reference_sha256=manifest['reference_sha256'], roster_sha256=digest(base/'cases.json'),
         settings=dict(input='single RGB screenshot resized bilinearly, ImageNet normalization, fp32 inference',input_wh=SIZE,
                       causes=CAUSES, sampling='bilinear image-centered downsampling to source stride 4, normalized uint8 probabilities'),
         inference_s=timings, load_s=load_s, artifacts=artifacts, code_sha256=digest(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('source-root','upstream','timm-source','weights','out'):
        parser.add_argument('--'+arg,type=Path,required=True)
    run(parser.parse_args())
