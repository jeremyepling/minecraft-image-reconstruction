"""Run the released four-head Pixel-wise Planarity model on each distinct frozen RGB capture."""
import argparse
import importlib.util
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.wall_join_test import capture_index, checked
from runtime.mode import image_only


MODEL_SHA = '305325096a80f541e430ccbb1b6581af67b3ef9d3c92f8d9384dc7aaae9f3952'
MODEL_REV = 'c09bd3361e64ff0dec473d6adc9456472e698d97'
UPSTREAM = '6bdd3a5d334bd716faae515a48e4408336d29b47'
MOGE = '8147f21883be489543e47e7a4d6134833ef992c7'
UTILS3D = '3fab839f0be9931dac7c8488eb0e1600c236e183'
SETTINGS = dict(max_width=800, tokens=1600, stride=4, planarity=.3, relative_depth=.025,
                segmentations={'labels':(5,8), 'thin':(5,4), 'relaxed':(20,8)})


def git_sha(path):
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=path, text=True).strip()


def pack_labels(labels, normal, points, native):
    # Preserve the exact center of each evaluation cell in source-image coordinates.
    width, height = native
    x = ((np.arange(2, width, 4) + .5) * labels.shape[1] / width).astype(int)
    y = ((np.arange(2, height, 4) + .5) * labels.shape[0] / height).astype(int)
    labels = labels[np.ix_(y, x)]
    ids, inv = np.unique(labels, return_inverse=True)
    if ids[0] != 0:
        inv = inv + 1
    labels = inv.reshape(labels.shape).astype(np.uint16)
    ids = labels.ravel()
    normals = np.nan_to_num(normal[np.ix_(y, x)].reshape(-1, 3), posinf=0, neginf=0)
    lengths = np.maximum(1., np.bincount(ids)).astype(float)
    vectors = np.stack([np.bincount(ids, weights=v) for v in normals.T], axis=1)
    vectors /= np.maximum(1e-7, np.linalg.norm(vectors,axis=1,keepdims=True))
    xyz = np.nan_to_num(points[np.ix_(y, x)].reshape(-1, 3), posinf=0, neginf=0)
    centers = np.stack([np.bincount(ids, weights=v)/lengths for v in xyz.T], axis=1)
    return labels, vectors.astype('float32'), centers.astype('float32')


@image_only()
def run(source_root, upstream, utils3d, checkpoint, output, device):
    import torch
    for path, expected in ((upstream, UPSTREAM), (upstream/'MoGe', MOGE), (utils3d, UTILS3D)):
        if git_sha(path) != expected:
            raise ValueError(f'Wrong upstream revision in {path}')
    checked(checkpoint, MODEL_SHA)
    sys.path[:0] = [str(upstream), str(utils3d)]
    from pxwplanar.moge_backend import MoGeModel
    # The pure segmentation module only needs cc3d, torch and numpy. Loading it directly
    # avoids importing upstream evaluation, training and reporting dependencies.
    filename = upstream/'pxwplanar/shared/segmentation/plan2seg.py'
    spec = importlib.util.spec_from_file_location('upstream_planarity_segmentation', filename)
    segmentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segmentation)
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
    start = perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = MoGeModel(**weights['model_config']).eval()
    model.load_state_dict(weights['model'])
    del weights
    model = model.to(device)
    if device == 'cuda':
        torch.cuda.synchronize()
    load_seconds = perf_counter() - start
    files = {}
    timings = {}
    for index, (sha,path) in enumerate(unique.items(), 1):
        with Image.open(path) as photo:
            rgb = photo.convert('RGB')
        width = min(SETTINGS['max_width'], rgb.width)
        resized = rgb.resize((width,round(rgb.height*width/rgb.width)), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.array(resized)).permute(2,0,1).to(device).float().div_(255)
        start = perf_counter()
        with torch.inference_mode():
            heads = model.infer(tensor, num_tokens=SETTINGS['tokens'], force_projection=True, apply_mask=True)
        depth = heads['depth'].cpu().numpy()
        normal = heads['normal'].cpu().numpy()
        points = heads['points'].cpu().numpy()
        planarity = heads['planarity'].squeeze().cpu().numpy()
        if depth.shape != planarity.shape or normal.shape != (*depth.shape,3):
            raise ValueError('Unexpected model output dimensions')
        valid = heads['mask'].cpu().numpy() & np.isfinite(depth)
        depth = np.where(valid, depth, 0.)
        inference_s = perf_counter() - start
        start = perf_counter()
        arrays = {}
        for key, (angle, neighbors) in SETTINGS['segmentations'].items():
            labels, _ = segmentation.compute_planar_segments(valid & (planarity > SETTINGS['planarity']),
                normal, depth, np.deg2rad(angle), SETTINGS['relative_depth'], neighbors, device)
            arrays[key], arrays[key+'_normals'], arrays[key+'_centers'] = pack_labels(labels, normal, points, rgb.size)
        segment_s = perf_counter() - start
        name = 'fields/'+sha+'.npz'
        np.savez_compressed(output/name, **arrays, image_size=rgb.size)
        files[name] = digest(output/name)
        timings[sha] = dict(inference_s=inference_s, segmentation_s=segment_s)
        if index == 1 or index%10 == 0:
            print(f'{index}/{len(unique)} unique frames; {inference_s:.3f}s inference', flush=True)
    save(output/'manifest.json',dict(model='Pixel-wise Planarity MoGe-2 ViT-L, four-head release',
        model_url='https://huggingface.co/alpayozkan/pxwplanar-moge2-planarity', model_revision=MODEL_REV,
        checkpoint_sha256=MODEL_SHA, upstream=UPSTREAM, moge=MOGE, utils3d=UTILS3D, device=device,
        torch_version=torch.__version__, device_name=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        versions={name:importlib.metadata.version(name) for name in
                  ('numpy','pillow','scipy','torch','torchvision','connected-components-3d')},
        seed=0, inference_script_sha256=digest(Path(__file__)), settings=SETTINGS,
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
