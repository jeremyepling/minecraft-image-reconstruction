"""Evaluate two already-defined line finders against physical patch boundaries in one frame."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_labels import rasterize
from benchmark.camera_metrics import truth_camera
from benchmark.mlsd_probe import sha256
from engine.structural_lines import detect


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def visible_surface_ids(pose, faces):
    w, h = pose['image_size']
    ids = np.zeros((h, w), dtype=np.int32)
    nearest = np.zeros((h, w))
    keys = {}
    Q = truth_camera(pose)
    for face in faces:
        normal = tuple(face['normal'])
        axis = next(i for i, value in enumerate(normal) if value)
        corners = np.asarray(face['world_corners'])
        plane = keys.setdefault((normal, corners[0, axis]), len(keys)+1)
        raster = rasterize(corners @ Q[:, :3].T + Q[:, 3], w, h)
        if raster is not None:
            box, inside, inverse_depth = raster[:3]
            nearer = inside & (inverse_depth > nearest[box])
            ids[box][nearer] = plane
            nearest[box][nearer] = inverse_depth[nearer]
    return ids


def dilated(mask, radius):
    """Native pixel centers within Euclidean distance radius of a true mask pixel."""
    h, w = mask.shape
    pad = np.pad(mask, radius)
    out = np.zeros_like(mask)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx*dx + dy*dy <= radius*radius:
                y, x = radius+dy, radius+dx
                out |= pad[y:y+h, x:x+w]
    return out


def score(segments, image_size, roi, boundary, band, radius):
    lines = Image.new('1', image_size)
    pen = ImageDraw.Draw(lines)
    for s in segments:
        pen.line([tuple(np.rint(s[k]).astype(int)) for k in ('p', 'q')], fill=1, width=1)
    pred = np.asarray(lines)
    inside = pred & roi
    hit = np.count_nonzero(inside & band)
    recovered = np.count_nonzero(boundary & dilated(inside, radius))
    count = int(inside.sum())
    return dict(segments=len(segments), all_line_px=int(pred.sum()), evaluated_line_px=count,
                line_px_within_tolerance=int(hit), line_precision=float(hit/count) if count else None,
                truth_boundary_px=int(boundary.sum()), recovered_boundary_px=int(recovered),
                boundary_recall=float(recovered/boundary.sum()) if boundary.any() else None)


def run(fixture, image_id, segments_path, out):
    frame = next(f for f in read(fixture/'index.json')['frames'] if f['id'] == image_id)
    screenshot = fixture/frame['image']
    supplied = read(segments_path)
    if supplied['image_sha256'] != sha256(screenshot):
        raise ValueError('M-LSD segments were generated from a different framebuffer')
    with Image.open(screenshot) as raw:
        source = raw.convert('RGB')
    existing = detect(source)
    # Evaluator-only data are read after both segment predictions have been frozen.
    ids = visible_surface_ids(read(fixture/frame['pose']), read(fixture/frame['labels'])['faces'])
    foreground = ids != 0
    pad = np.pad(ids, 1, mode='edge')
    h, w = ids.shape
    boundary = np.zeros_like(foreground)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        boundary |= foreground & (ids != pad[1+dy:h+1+dy, 1+dx:w+1+dx])
    radius = 5
    roi, band = dilated(foreground, radius), dilated(boundary, radius)
    methods = {'mlsd-tiny-512-fp16': supplied['segments'], 'structural_lines.detect': existing}
    result = dict(frame=image_id, fixture=fixture.name, image_sha256=supplied['image_sha256'],
                  model_sha256=supplied['model_sha256'], tolerance_native_px=radius,
                  scope='One development observation. Unique rasterized 1-pixel lines in the visible building footprint and 5-pixel exterior band.',
                  truth='Foreground pixel adjacent (4-neighborhood) to another visible face normal/plane offset or air, at native resolution. Coplanar block seams excluded.',
                  roi_px=int(roi.sum()), results={k:score(v,source.size,roi,boundary,band,radius) for k,v in methods.items()})
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['results'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'segments', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    run(args.fixture, args.id, args.segments, args.out)
