"""Render the canopy failure from frozen transfer inputs and decoded placement results."""
import argparse
from io import BytesIO
from itertools import product
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_metrics import truth_camera
from benchmark.texture_camera_sheet import font, read
from benchmark.visible_depth_sheet import CORRECT, EXTRA, HIDDEN, render
from benchmark.visible_depth_transfer import OUT, structure
from engine.geometry import project
from training.visible_labels import visible_labels


SOURCE = 'train-canopy-09-right'
KEY = 'shape-benchmark-26.3--' + SOURCE


def overlay(image, parts):
    result = image.copy()
    colors = [(144, 97, 221), (36, 200, 207), (243, 160, 93), (100, 163, 239)]
    for i, part in enumerate(parts):
        support = part.get('support')
        if support:
            mask = Image.new('L', tuple(support['size']))
            pen = ImageDraw.Draw(mask)
            for y, left, right in support['runs']:
                pen.line((left, y, right - 1, y), fill=130)
            result.paste(colors[i % len(colors)], (0, 0, *image.size),
                         mask.resize(image.size, Image.Resampling.NEAREST))
    return result


def build(transfer, out):
    prefix = 'fixtures/shape-benchmark-26.3/captures/' + SOURCE
    with ZipFile(transfer / 'native-transfer-v1.zip') as archive:
        image = Image.open(BytesIO(archive.read(prefix + '.png'))).convert('RGB')
        pose = json.loads(archive.read(prefix + '.json'))
        raw = json.loads(archive.read(prefix + '-blocks.json'))
        parts = json.loads(archive.read(f'transfer/upstream/{KEY}/surfaces.json'))['planes']
    world = {tuple(b[k] for k in 'xyz'): b for b in raw}
    labels, _, _ = visible_labels(pose, world, (0, 0, 0))
    truth = {tuple(b['xyz_world']) for b in labels['blocks']}
    occupied = set(structure(raw, 64))
    P = truth_camera(pose)
    eye = pose['eye_world']
    old, new = [set(map(tuple, read(transfer / 'results' / method / KEY / 'cells.json')['world']))
                for method in ('prior', 'production')]
    corners = np.asarray(list(product((0, 1), repeat=3)))
    points = (np.array(sorted(truth | old | new))[:, None, :] + corners).reshape(-1, 3)
    xy = project(P, points[points @ P[2, :3] + P[2, 3] > 0])
    bounds = (*np.maximum(0, np.floor(xy.min(0) - 10)).astype(int),
              *np.minimum(image.size, np.ceil(xy.max(0) + 10)).astype(int))

    tiles = [image.crop(bounds), overlay(image, parts).crop(bounds),
             render(truth, P, eye, bounds, truth, occupied, True),
             render(old, P, eye, bounds, truth, occupied), render(new, P, eye, bounds, truth, occupied)]
    sheet = Image.new('RGB', (1164, 736), '#101921')
    pen = ImageDraw.Draw(sheet)
    pen.text((22, 16), 'Canopy | right view', font=font(34), fill='white')
    pen.text((22, 61), 'Native Minecraft 26.3 capture  /  held-out geometry', font=font(17), fill='#acbdc9')
    names = ['Screenshot', 'Image-derived surfaces', 'Visible reference', 'Previous placement', 'Combined placement']
    for i, (name, tile) in enumerate(zip(names, tiles, strict=True)):
        x = 22 + i % 3 * 379
        y = 111 + i // 3 * 300
        pen.text((x, y), name, font=font(21), fill='white')
        tile.thumbnail((365, 251), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x + (365 - tile.width) // 2, y + 38 + (251 - tile.height) // 2))
    x, y = 780, 411
    pen.text((x, y), 'Visible block coordinates', font=font(21), fill='white')
    metrics = [('Recovered', len(old & truth), len(new & truth), CORRECT),
               ('Extra', len(old - truth), len(new - truth), EXTRA),
               ('Hidden extra', len((old - truth) & occupied), len((new - truth) & occupied), HIDDEN)]
    for i, (name, before, after, color) in enumerate(metrics):
        line = y + 84 + i * 42
        pen.rectangle((x, line + 5, x + 11, line + 16), fill=color)
        pen.text((x + 20, line), name, font=font(18), fill='white')
        for col, val in enumerate((before, after)):
            pen.text((x + 170 + col * 88, line), str(val), font=font(19), fill='white')
    for col, title in enumerate(('Previous', 'Combined')):
        pen.text((x + 170 + col * 88, y + 48), title, font=font(15), fill='#acbdc9')
    pen.text((x, y + 230), f'{len(truth)} visible coordinates in the reference', font=font(17), fill='#acbdc9')
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92, subsampling=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer', type=Path, default=OUT)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'reviews/visible-depth-01/canopy.jpg')
    args = parser.parse_args()
    build(args.transfer, args.out)
