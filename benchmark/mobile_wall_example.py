"""Draw the same screenshot crop before and after the MobileSAM wall partition."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from benchmark.camera_round import read
from benchmark.mobile_region_probe import METHOD, WALLS
from benchmark.normal_round import groups


def draw_example(out, source, key):
    rows = [r for split in ('development', 'regression') for r in read(source/split/'cases.json')]
    row = next(r for r in rows if r['key'] == key)
    before = next(r for r in read(WALLS/'cases.json') if r['key'] == key)
    with gzip.open(WALLS/before['predictions'], 'rt', encoding='utf8') as f:
        parents = json.load(f)[METHOD]
    with gzip.open(source/row['predictions'], 'rt', encoding='utf8') as f:
        children = json.load(f)
    event = row['events'][0]
    parent = next(p for p in parents if p['id'] == event['parent'])
    children = [c for c in children if c['id'] in event['children']]
    by_id = {f'{c.floor}/{c.id}': c for group in groups()[0].values() for c in group}
    with Image.open(by_id[key].path('image')) as f:
        original = f.convert('RGB')
    def mask(p):
        support = p['support']
        result = np.zeros(support['size'][::-1], np.uint8)
        for y, a, b in support['runs']:
            result[y, a:b] = 255
        return Image.fromarray(result).resize(original.size, Image.Resampling.NEAREST)
    def overlay(parts):
        colors = ['#f7cf41', '#4ee1f2', '#eb78de', '#8ced65', '#ff996e']
        result = original.copy()
        for p, color in zip(parts, colors):
            m = mask(p)
            outer = np.asarray(m.filter(ImageFilter.MaxFilter(3)))
            inner = np.asarray(m.filter(ImageFilter.MinFilter(3)))
            result.paste(color, mask=Image.fromarray(outer-inner))
        return result
    pad = 24
    left, top, right, bottom = mask(parent).getbbox()
    crop = (max(0, left-pad), max(0, top-pad), min(original.width, right+pad), min(original.height, bottom+pad))
    size = (390, round(390*(crop[3]-crop[1])/(crop[2]-crop[0])))
    canvas = Image.new('RGB', (3*(size[0]+12)+4, size[1]+64), '#101924')
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=13)
    for i, (im, title) in enumerate(zip([original, overlay([parent]), overlay(children)],
                                         ['Original screenshot', 'Wall Separator', 'After MobileSAM split'])):
        left = 8+i*(size[0]+12)
        d.text((left, 7), title, font=font, fill='white')
        canvas.paste(im.crop(crop).resize(size, Image.Resampling.LANCZOS), (left, 35))
    d.text((8, size[1]+41), f'{key}  |  Same camera and crop. Colors outline the predictions.', font=small, fill='white')
    canvas.save(out, quality=93)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('benchmark/reviews/mobile-walls-01'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key', default='light_gray_concrete/holdout-galleries-masonry-low-left')
    args = parser.parse_args()
    draw_example(args.out, args.source, args.key)
