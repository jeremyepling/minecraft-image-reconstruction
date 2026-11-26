"""Pinned Pixel-wise Planarity inference for one RGB image. Optional dependencies load in the model process."""
import importlib.util
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from engine.atlas import digest


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
    x = ((2.5+4*np.arange(width//4)) * labels.shape[1] / width).astype(int)
    y = ((2.5+4*np.arange(height//4)) * labels.shape[0] / height).astype(int)
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


def load_model(upstream, utils3d, checkpoint, device):
    import torch
    for path, expected in ((upstream, UPSTREAM), (upstream/'MoGe', MOGE), (utils3d, UTILS3D)):
        dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=path)
        if git_sha(path) != expected or dirty:
            raise ValueError(f'Expected the unmodified pinned upstream revision in {path}')
    if digest(checkpoint) != MODEL_SHA:
        raise ValueError('Pixel-wise Planarity checkpoint does not match its release')
    sys.path[:0] = [str(upstream), str(utils3d)]
    from pxwplanar.moge_backend import MoGeModel
    # The pure segmentation module only needs cc3d, torch and numpy. Loading it directly
    # avoids importing upstream evaluation, training and reporting dependencies.
    filename = upstream/'pxwplanar/shared/segmentation/plan2seg.py'
    spec = importlib.util.spec_from_file_location('upstream_planarity_segmentation', filename)
    segmentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segmentation)
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
    return model,segmentation,load_seconds


def infer_image(rgb, model, segmentation, device, keys):
    import torch
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
    for key in keys:
        angle, neighbors = SETTINGS['segmentations'][key]
        labels, _ = segmentation.compute_planar_segments(valid & (planarity > SETTINGS['planarity']),
            normal, depth, np.deg2rad(angle), SETTINGS['relative_depth'], neighbors, device)
        arrays[key], arrays[key+'_normals'], arrays[key+'_centers'] = pack_labels(labels, normal, points, rgb.size)
    segment_s = perf_counter() - start
    return dict(**arrays, image_size=rgb.size),dict(inference_s=inference_s,segmentation_s=segment_s)
