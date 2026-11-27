"""Show how much of the photographed Facade wall belongs to its matched mask."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.texture_camera_sheet import font
from engine.pillar_seams import native_mask


def overlay(image, mask, color):
    pixels = np.array(image)
    inside = np.pad(mask, 1)
    core = inside[1:-1, :-2] & inside[1:-1, 2:] & inside[:-2, 1:-1] & inside[2:, 1:-1]
    pixels[mask] = (.67 * pixels[mask] + .33 * np.array(color)).astype('uint8')
    pixels[mask & ~core] = color
    return Image.fromarray(pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surfaces', required=True, type=Path, help='surfaces.json from the ordinary RGB run')
    parser.add_argument('--out', type=Path, default=Path('benchmark/reviews/strip-orientation-01'))
    args = parser.parse_args()
    manifest = read(args.surfaces.with_name('build-manifest.json'))
    key = next(row['representative']['id'] for row in
               read(Path('benchmark/reviews/facade-placement-01/worst/ranking.json'))['layouts'] if row['name'] == 'Facade')
    run = Path('../visible-error-budget/runs/primary-production')
    base = next(r for r in read(checked(run/'cases.json', read(run/'summary.json')['cases_sha256'])) if r['id'] == key)
    split, frame, prior = next(case for case in framed.cases((base['split'],)) if case[1]['id'] == key)
    image, parts, _, _, _, labels, ids, *_ = read_view(run, split, frame, prior, base)
    assert manifest['image_sha256'] == base['rgb_sha256']
    after = read(args.surfaces)['planes']
    previous = {p['id']: p for p in parts}
    disappeared = [native_mask(p, image.size) for k,p in previous.items() if k not in {a['id'] for a in after}]
    new = [native_mask(p, image.size) for p in after if p['id'] not in previous]
    assert len(disappeared) == 2 and len(new) == 1
    current = new[0]
    assert np.array_equal(current, disappeared[0] | disappeared[1])
    before = max(disappeared, key=lambda m: m.sum())
    other = next(m for m in disappeared if m is not before)
    lut = np.zeros(len(labels['faces']) + 1, dtype='int32')
    for face in labels['faces']:
        lut[face['id']] = face['surface_id']
    surfaces = lut[ids]
    sid = int(np.bincount(surfaces[current])[1:].argmax() + 1)
    target = surfaces == sid
    def quantify(mask):
        common = int((mask & target).sum())
        return dict(mask_px=int(mask.sum()), correct_px=common, reference_px=int(target.sum()),
                    missing_px=int((target & ~mask).sum()), background_px=int((mask & (ids == 0)).sum()),
                    neighboring_surface_px=int((mask & (surfaces != sid) & (ids != 0)).sum()),
                    recall=common / int(target.sum()), precision=common / int(mask.sum()))
    stats = dict(frame=key, rgb_sha256=base['rgb_sha256'], surfaces_sha256=digest(args.surfaces),
                 surface_id=sid, before=quantify(before), after=quantify(current),
                 union_added_pixels=int((current & ~(before | other)).sum()))
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out/'wall-extent.json', stats)
    ys, xs = np.where((surfaces > 0) | current)
    crop = (max(0, int(xs.min()) - 12), max(0, int(ys.min()) - 12),
            min(image.width, int(xs.max()) + 13), min(image.height, int(ys.max()) + 13))
    prior_view = overlay(overlay(image, before, (254, 181, 74)), other, (193, 126, 250))
    pictures = [(prior_view, 'Original split', 'Gold and violet mark separate masks'),
                (overlay(image, current, (87, 222, 180)), 'Corrected outline', 'Green marks the combined wall'),
                (overlay(image, target, (106, 194, 252)), 'Recorded visible face', 'Blue marks the scoring reference')]
    board = Image.new('RGB', (1100, 310), '#152530')
    draw = ImageDraw.Draw(board)
    draw.text((16, 9), 'Facade  ·  same photograph in all three panels', font=font(22), fill='white')
    for i, (panel, heading, caption) in enumerate(pictures):
        x = i * 366 + 10
        draw.text((x, 45), heading, font=font(18), fill='white')
        thumb = ImageOps.contain(panel.crop(crop), (350, 195), Image.Resampling.LANCZOS)
        board.paste(thumb, (x + (350 - thumb.width) // 2, 72 + (195 - thumb.height) // 2))
        draw.text((x, 276), caption, font=font(14), fill='#d1e1ea')
    board.save(args.out/'wall-extent.jpg', quality=94)
    print(stats)


if __name__ == '__main__':
    main()
