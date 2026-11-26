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
from engine.planarity_entry import LABELS, RELEASE, selected_settings
from runtime.mode import image_only


STRIDE = planarity_source.SETTINGS['stride']
TILING = dict(grid=[3, 3], crop_fraction=.5, stride=STRIDE, artificial_edge='discard touching region',
              geometry='Average full-image camera normals and points under registered tile labels',
              order='row-major; for each parent retain one accepted split with the most supporting samples')


def crop_boxes(size):
    """Nine half-frame windows. Align edges to the same 4-pixel cell lattice as the original labels."""
    if any(v % STRIDE or v < 4 * STRIDE for v in size):
        raise ValueError('Crop test requires image dimensions aligned to the full-frame lattice')
    w, h = (v // STRIDE for v in size)
    width, height = (w + 1) // 2, (h + 1) // 2
    return [tuple(STRIDE * v for v in (x, y, x + width, y + height))
            for y in (0, (h - height) // 2, h - height)
            for x in (0, (w - width) // 2, w - width)]


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
    x = ((2.5 + STRIDE * np.arange(rgb.width // STRIDE)) * normal.shape[1] / rgb.width).astype(int)
    y = ((2.5 + STRIDE * np.arange(rgb.height // STRIDE)) * normal.shape[0] / rgb.height).astype(int)
    return normal[np.ix_(y, x)], points[np.ix_(y, x)]


def mean_geometry(labels, normal, points):
    ids = labels.ravel()
    valid = np.isfinite(normal).all(axis=2) & np.isfinite(points).all(axis=2) & (points[..., 2] > 0)
    select = valid.ravel()
    count = np.maximum(1, np.bincount(ids[select], minlength=int(ids.max()) + 1))
    def means(field):
        values = field.reshape(-1, 3)[select]
        return np.stack([np.bincount(ids[select], weights=c, minlength=len(count)) for c in values.T], axis=1) / count[:, None]
    normals = means(normal)
    normals /= np.maximum(1e-8, np.linalg.norm(normals, axis=1, keepdims=True))
    return normals.astype('float32'), means(points).astype('float32')


def compute_frame(image, model, segmentation, device):
    start = perf_counter()
    normal, points = global_geometry(image, model, device)
    timing = dict(global_geometry_s=perf_counter()-start, crop_inference_s=0., crop_segmentation_s=0.)
    boxes = crop_boxes(image.size)
    tiles = []
    fields = dict(image_size=image.size, boxes=boxes)
    for i, box in enumerate(boxes):
        data, elapsed = planarity_source.infer_image(image.crop(box), model, segmentation, device, [LABELS])
        tile = data[LABELS]
        tiles.append(tile)
        x0, y0, x1, y1 = [v // STRIDE for v in box]
        fields[f'normals_{i}'], fields[f'centers_{i}'] = mean_geometry(tile, normal[y0:y1, x0:x1], points[y0:y1, x0:x1])
        timing['crop_inference_s'] += elapsed['inference_s']
        timing['crop_segmentation_s'] += elapsed['segmentation_s']
    fields['tile_labels'] = np.stack(tiles)
    return fields, timing


def registered_tiles(data, complete_only=True):
    """Yield (index, full-grid label image, normals, centers); behind a crop edge is unknown."""
    width, height = map(int, data['image_size'])
    boxes = crop_boxes((width, height))
    if not np.array_equal(data['boxes'], boxes) or len(data['tile_labels']) != len(boxes):
        raise ValueError('Cached tile grid differs from the frozen geometry')
    for i, (box, tile) in enumerate(zip(boxes, data['tile_labels'])):
        x0, y0, x1, y1 = [v // STRIDE for v in box]
        if tile.shape != (y1 - y0, x1 - x0):
            raise ValueError('Inconsistent crop transform')
        if complete_only:
            rims = [edge for edge, cut in ((tile[0], y0>0), (tile[-1], box[3]<height),
                                          (tile[:, 0], x0>0), (tile[:, -1], box[2]<width)) if cut]
            excluded = np.unique(np.concatenate(rims))
            tile = np.where(np.isin(tile, excluded), 0, tile)
        canvas = np.zeros((height//STRIDE, width//STRIDE), dtype=tile.dtype)
        canvas[y0:y1, x0:x1] = tile
        yield i, canvas, data[f'normals_{i}'], data[f'centers_{i}']


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
