"""Compare ordinary RGB builds and show the actually recovered Watchpost silhouette."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font
from engine.pillar_seams import native_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rgb', type=Path)
    parser.add_argument('--before', type=Path, default=Path('runs/watch-control'))
    parser.add_argument('--after', type=Path, default=Path('runs/watch-candidate'))
    parser.add_argument('--out', type=Path, default=Path('benchmark/reviews/visible-skyline-01'))
    args = parser.parse_args()
    baseline = read(args.before/'build-manifest.json')
    candidate = read(args.after/'build-manifest.json')
    assert baseline['image_sha256'] == candidate['image_sha256']
    with Image.open(checked(args.rgb, baseline['image_sha256'])) as src:
        image = src.convert('RGB')
    old = read(args.before/'surfaces.json')['planes']
    new = read(args.after/'surfaces.json')['planes']
    assert [p['id'] for p in old] == [p['id'] for p in new]
    additions = np.zeros((image.height, image.width), bool)
    for a, b in zip(old, new):
        if a == b:
            continue
        m, n = native_mask(a, image.size), native_mask(b, image.size)
        assert np.all(n[m])
        additions |= n & ~m
    ys, xs = np.nonzero(additions)
    assert len(ys) == 225
    box = (int(xs.min()) - 12, int(ys.min()) - 16, int(xs.max()) + 13, int(ys.max()) + 17)
    thumb = image.copy()
    ImageDraw.Draw(thumb).rectangle(box, outline='#ffd262', width=4)
    highlighted = np.array(image)
    highlighted[additions] = (255, 83, 93)
    panels = [(thumb, 'Whole screenshot'), (image.crop(box), 'Photographed roofline'),
              (Image.fromarray(highlighted).crop(box), '225 recovered wall pixels in red')]
    board = Image.new('RGB', (1104, 240), '#152530')
    draw = ImageDraw.Draw(board)
    for i,(panel, title) in enumerate(panels):
        x = 10 + 366*i
        draw.text((x, 10), title, font=font(17), fill='white')
        pic = ImageOps.contain(panel, (350, 180), Image.Resampling.NEAREST)
        board.paste(pic, (x + (350-pic.width)//2, 40+(180-pic.height)//2))
    args.out.mkdir(parents=True, exist_ok=True)
    board.save(args.out/'watchpost.jpg', quality=95)
    packs = [decoded_pack(folder/'reconstruction.zip') for folder in (args.before, args.after)]
    assert packs[0] == packs[1]
    assert (args.before/'counts.json').read_bytes() == (args.after/'counts.json').read_bytes()
    assert (args.before/'placements.json').read_bytes() == (args.after/'placements.json').read_bytes()
    data = dict(rgb_sha256=baseline['image_sha256'], added_px=len(ys), placed_blocks=len(packs[0]),
                stage_sha256={name:[digest(f/name) for f in (args.before,args.after)]
                              for name in ('surfaces.json','counts.json','placements.json','reconstruction.zip')})
    save(args.out/'cli.json',data)


if __name__ == '__main__':
    main()
