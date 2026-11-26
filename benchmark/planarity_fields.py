"""Run the released four-head Pixel-wise Planarity model on each distinct frozen RGB capture."""
import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.wall_join_test import capture_index, checked
from engine.planarity_source import MODEL_SHA, MODEL_REV, UPSTREAM, MOGE, UTILS3D, SETTINGS, load_model, infer_image
from runtime.mode import image_only


@image_only()
def run(source_root, upstream, utils3d, checkpoint, output, device):
    import torch
    inputs = source_root/'benchmark/reviews/single-wall-depth-01'
    base = read(inputs/'manifest.json')
    rows = read(checked(inputs/'cases.json',base['artifacts']['cases.json']))
    captures = capture_index(source_root/'benchmark/data')
    output.mkdir(parents=True, exist_ok=False)
    (output/'fields').mkdir()
    unique = {}
    for row in rows:
        image_path = checked(captures[row['key']].path('image'), row['sha256'])
        unique[row['sha256']] = image_path
    if len(rows) != 56 or len(unique) != 52:
        raise ValueError('Frozen frame suite differs')
    model, segmentation, load_seconds = load_model(upstream,utils3d,checkpoint,device)
    files = {}
    timings = {}
    for index, (sha,path) in enumerate(unique.items(), 1):
        with Image.open(path) as photo:
            rgb = photo.convert('RGB')
        arrays, timings[sha] = infer_image(rgb,model,segmentation,device,SETTINGS['segmentations'])
        name = 'fields/'+sha+'.npz'
        np.savez_compressed(output/name, **arrays)
        files[name] = digest(output/name)
        if index == 1 or index%10 == 0:
            print(f'{index}/{len(unique)} unique frames; {timings[sha]["inference_s"]:.3f}s inference', flush=True)
    save(output/'manifest.json',dict(model='Pixel-wise Planarity MoGe-2 ViT-L, four-head release',
        model_url='https://huggingface.co/alpayozkan/pxwplanar-moge2-planarity', model_revision=MODEL_REV,
        checkpoint_sha256=MODEL_SHA, upstream=UPSTREAM, moge=MOGE, utils3d=UTILS3D, device=device,
        torch_version=torch.__version__, device_name=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        versions={name:importlib.metadata.version(name) for name in
                  ('numpy','pillow','scipy','torch','torchvision','connected-components-3d')},
        seed=0, inference_script_sha256=digest(Path(__file__)),
        model_loader_sha256=digest(ROOT/'engine/planarity_source.py'), settings=SETTINGS,
        fixtures=base['fixtures'], load_seconds=load_seconds,
        timing=timings, artifacts=files, inputs='One RGB. Resize preserves source aspect ratio; no crop or padding. '
        'Native cell centers at 2 + 4n are mapped to the raw field by nearest neighbor.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--utils3d', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda','cpu'))
    args = parser.parse_args()
    run(args.source_root.resolve(), args.upstream.resolve(), args.utils3d.resolve(), args.checkpoint.resolve(),
        args.out.resolve(), args.device)
