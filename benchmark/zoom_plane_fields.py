"""Infer overlapping RGB crops and put their plane masks on the full screenshot's sampling grid."""
import argparse
import importlib.metadata
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.wall_join_test import checked
from engine import planarity_source
from engine.planarity_crops import TILING, compute_crops, registered_tiles
from engine.planarity_entry import LABELS, RELEASE, selected_settings
from runtime.mode import image_only


def global_geometry(rgb, model, device):
    """Use the same global MoGe camera and sampling convention as planarity_source.infer_image."""
    import torch
    width = min(planarity_source.SETTINGS['max_width'], rgb.width)
    smaller = rgb.resize((width, round(rgb.height * width / rgb.width)), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.array(smaller)).permute(2, 0, 1).to(device).float().div_(255)
    with torch.inference_mode():
        heads = model.infer(tensor, num_tokens=planarity_source.SETTINGS['tokens'],
                            force_projection=True, apply_mask=True)
    normal = heads['normal'].cpu().numpy()
    points = heads['points'].cpu().numpy()
    indices = np.ix_(*planarity_source.sample_indices(rgb.size, normal.shape))
    return normal[indices], points[indices]


def compute_frame(image, model, segmentation, device):
    start = perf_counter()
    normal, points = global_geometry(image, model, device)
    elapsed = perf_counter() - start
    fields, timing = compute_crops(image, model, segmentation, device, normal, points, LABELS)
    return fields, dict(global_geometry_s=elapsed, **timing)


def render_crop_proposals(photo, fields_path, box=None):
    """Show actual eligible crop boundaries in source coordinates; different crop views may overlap."""
    photo = photo.convert('RGB')
    with np.load(fields_path, allow_pickle=False) as cache:
        data = dict(cache)
    if tuple(data['image_size']) != photo.size:
        raise ValueError('Wrong screenshot dimensions for crop evidence')
    color = np.asarray(photo).copy()
    for i, labels, _, _ in registered_tiles(data):
        edge = np.zeros_like(labels, bool)
        edge[1:] |= labels[1:] != labels[:-1]
        edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edge &= labels != 0
        native = np.asarray(Image.fromarray(edge).resize(photo.size, Image.Resampling.NEAREST))
        color[native] = (255, 205 - 12*i, 45 + 20*i)
    overlay = Image.fromarray(color)
    return overlay.crop(box) if box is not None else overlay


@image_only()
def run(source, out, upstream, utils3d, checkpoint, device):
    import torch
    rows = read(source)
    if len(rows) != 52 or len({r['sha256'] for r in rows}) != 52:
        raise ValueError('Wrong unique RGB roster')
    out.mkdir(parents=True, exist_ok=False)
    (out/'fields').mkdir()
    model, segmentation, load_s = planarity_source.load_model(upstream, utils3d, checkpoint, device)
    times, files = {}, {}
    for index, row in enumerate(rows, 1):
        sha = row['sha256']
        with Image.open(checked(Path(row['path']),sha)) as photo:
            arrays, times[sha] = compute_frame(photo.convert('RGB'), model, segmentation, device)
        name = f'fields/{sha}.npz'
        np.savez_compressed(out/name, **arrays)
        files[name] = digest(out/name)
        if index == 1 or index % 10 == 0:
            print(f'{index}/52 distinct screenshots', flush=True)
    save(out/'manifest.json', dict(**RELEASE, settings=TILING, model_settings=selected_settings(planarity_source.SETTINGS),
        source_sha256=digest(Path(__file__)), loader_sha256=digest(ROOT/'engine/planarity_source.py'),
        crop_mapper_sha256=digest(ROOT/'engine/planarity_crops.py'),
        source_frames=sorted(times), device=device, torch=torch.__version__,
        device_name=torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
        versions={n:importlib.metadata.version(n) for n in ('numpy','pillow','scipy','torch','torchvision','connected-components-3d')},
        load_s=load_s, timing=times, artifacts=files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('roster','out','upstream','utils3d','checkpoint'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    opt = parser.parse_args()
    run(opt.roster, opt.out, opt.upstream, opt.utils3d, opt.checkpoint, opt.device)
