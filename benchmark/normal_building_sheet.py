"""Show the same eight whole-building views from the frozen normal-region experiment."""
import argparse
from fractions import Fraction
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, read
from benchmark.camera_sheet import font
from benchmark.normal_sheet import case_data
from benchmark.surface_sheet import scene_box


REFERENCE = ROOT / 'benchmark/reviews/normals-01'
METHOD = 'normal-regions'
BUILDINGS = ('shopfront', 'wing', 'courtyard', 'portico', 'workshop', 'crossing', 'galleries', 'registry')
PAPER, INK, MUTED = '#faf8f3', '#172d33', '#465b61'
MATCH, EXTRA, MISSED = '#28efba', '#ff5b96', '#ffcc47'
PANEL = (680, 400)
MARGIN, GAP, HEADER, ROW = 32, 14, 225, 474


def selected_views():
    """Pick once on the published B run so future sheets retain the original camera views."""
    def rank(row):
        scores = row['methods'][METHOD]['outlines']
        f1 = Fraction(2 * scores['matched'], max(1, scores['eligible'] + scores['proposed']))
        return -f1, -scores['matched'], row['floor'] != 'light_gray_concrete', row['id']

    cases = read(REFERENCE / 'cases.json')
    return [min((c for c in cases if c['layout'] == name and not c['id'].endswith('-close-front')),
                key=rank) for name in BUILDINGS]


def normal_recipe(run):
    manifest = read(run / 'manifest.json')
    settings = manifest['settings']['inference']
    return dict(model=manifest['model_sha256']['normal'], settings={
        k: settings[k] for k in ('provider', 'normal_max_side', 'normal_tokens', 'normal_resize')})


def save_previews(cache, path):
    from benchmark.moge_probe import normal_image
    from benchmark.normal_fields import fields

    with ZipFile(path, 'w') as archive:
        archive.writestr(ZipInfo('recipe.json'), json.dumps(normal_recipe(REFERENCE), sort_keys=True))
        for row in selected_views():
            normals, _ = fields(REFERENCE / row['source'], row['sha256'], cache)
            png = BytesIO()
            normal_image(normals).save(png, format='PNG', optimize=True)
            archive.writestr(ZipInfo(row['sha256'] + '.png'), png.getvalue())


def crop_for_view(image, truth, planes):
    # Keep the entire building and every visible predicted outline inside the common crop.
    x0, y0, x1, y1 = scene_box(truth, image)
    for plane in planes:
        xy = np.asarray(plane['quad'])
        x0, y0 = np.minimum([x0, y0], xy.min(axis=0) - 12)
        x1, y1 = np.maximum([x1, y1], xy.max(axis=0) + 12)
    w, h = image.size
    x0, x1 = np.clip([x0, x1], 0, w)
    y0, y1 = np.clip([y0, y1], 0, h)
    ratio = PANEL[0] / PANEL[1]
    width = max(x1 - x0, (y1 - y0) * ratio)
    height = width / ratio
    if width > w or height > h:
        return (0, 0, w, h)
    left = np.clip((x0 + x1 - width) / 2, 0, w - width)
    top = np.clip((y0 + y1 - height) / 2, 0, h - height)
    return tuple(round(value) for value in (left, top, left + width, top + height))


def stroke(draw, points, color, width):
    draw.line(points, fill='#172328', width=width + 2, joint='curve')
    draw.line(points, fill=color, width=width, joint='curve')


def dashed(draw, quad):
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        length = np.linalg.norm(end - start)
        if length < 1:
            continue
        unit = (end - start) / length
        for offset in np.arange(0, length, 13):
            stroke(draw, [tuple(start + offset * unit), tuple(start + min(length, offset + 8) * unit)],
                   EXTRA, 3)


def checked_overlay(image, truth, planes, scores):
    eligible = {i for i, patch in truth['patches'].items() if patch['eligible']}
    matches = scores['matches']
    hit = {m['proposal_index'] for m in matches}
    missed = eligible - {m['patch_id'] for m in matches}
    if (len(planes), len(hit), len(eligible), len(missed)) != (
            scores['proposed'], scores['matched'], scores['eligible'], len(eligible) - len(hit)):
        raise ValueError('Saved match counts disagree with the outlines or reference surfaces')
    mask = Image.fromarray(np.isin(truth['id_map'], list(missed))).resize(image.size, Image.Resampling.NEAREST)
    y, x = np.ogrid[:image.height, :image.width]
    opacity = np.where(np.asarray(mask), np.where((x + y) % 11 < 3, 200, 38), 0).astype(np.uint8)
    image = image.copy()
    image.paste(MISSED, (0, 0, *image.size), Image.fromarray(opacity))
    draw = ImageDraw.Draw(image)
    for i in sorted(hit):
        points = [tuple(p) for p in planes[i]['quad']]
        stroke(draw, points + points[:1], MATCH, 3)
    for i, plane in enumerate(planes):
        if i not in hit:
            dashed(draw, np.asarray(plane['quad']))
    return image


def draw_text(canvas, xy, value, size, *, bold=False, fill=INK, anchor='lt'):
    ImageDraw.Draw(canvas).text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def legend(canvas, x, title, kind):
    draw = ImageDraw.Draw(canvas)
    color = {'match': MATCH, 'extra': EXTRA, 'missed': MISSED}[kind]
    draw.rounded_rectangle((x, 116, x + 43, 142), radius=3, fill='#34474c')
    if kind == 'missed':
        for dx in range(4, 30, 7):
            draw.line((x + dx, 137, x + dx + 12, 120), fill=color, width=3)
    elif kind == 'match':
        draw.line((x + 5, 129, x + 38, 129), fill=color, width=4)
    else:
        for dx in (5, 17, 29):
            draw.line((x + dx, 129, x + dx + 8, 129), fill=color, width=4)
    draw_text(canvas, (x + 52, 117), title, 24)


def render(run, previews, out):
    current = {case['key']: case for case in read(run / 'cases.json')}
    views = selected_views()
    width = 2 * MARGIN + 3 * PANEL[0] + 2 * GAP
    canvas = Image.new('RGB', (width, HEADER + ROW * len(views) + 8), PAPER)
    xs = [MARGIN + i * (PANEL[0] + GAP) for i in range(3)]
    draw_text(canvas, (MARGIN, 19), 'What B saw in each building', 46, bold=True)
    draw_text(canvas, (MARGIN, 77), 'B’s best whole-building view from the original run. All three panels use the same crop.', 25, fill=MUTED)
    legend(canvas, MARGIN, 'Matched a surface', 'match')
    legend(canvas, MARGIN + 320, 'Extra outline', 'extra')
    legend(canvas, MARGIN + 581, 'Surface B missed', 'missed')
    titles = ('Original screenshot', 'Which way each surface faces', 'B’s rectangles, checked against Minecraft')
    for x, title in zip(xs, titles):
        draw_text(canvas, (x, 173), title, 29, bold=True)
    print('Building | saved frame | found / visible | extra outlines')
    for i, original in enumerate(views):
        row = current[original['key']]
        if row['sha256'] != original['sha256']:
            raise ValueError('A selected screenshot changed since the reference run')
        image, truth, predictions = case_data(run, row)
        planes = predictions[METHOD]['model']['planes']
        scores = row['methods'][METHOD]['outlines']
        overlay = checked_overlay(image, truth, planes, scores)
        box = crop_for_view(image, truth, planes)
        # The normals were inferred on a smaller copy of this same screenshot.
        with Image.open(BytesIO(previews.read(row['sha256'] + '.png'))) as preview:
            directions = preview.convert('RGB').resize(image.size, Image.Resampling.BILINEAR)
        y = HEADER + i * ROW
        ImageDraw.Draw(canvas).line((MARGIN, y - 8, width - MARGIN, y - 8), fill='#c4cfcb', width=2)
        _, palette, elevation, side = row['id'].rsplit('-', 3)
        floor = row['floor'].replace('_', ' ')
        label = f"{row['layout'].title()}   ·   {palette}, {elevation} {side}, {floor} floor"
        draw_text(canvas, (MARGIN, y + 7), label, 27, bold=True)
        found, eligible, proposed = (scores[k] for k in ('matched', 'eligible', 'proposed'))
        extra = proposed - found
        result = f"{found} of {eligible} found    ·    {eligible - found} missed    ·    {extra} extra"
        draw_text(canvas, (width - MARGIN, y + 9), result, 25, anchor='rt')
        for x, panel in zip(xs, (image, directions, overlay)):
            crop = panel.crop(box)
            # Preserve aspect ratio if a wide outlying proposal required the entire frame.
            scale = min(PANEL[0] / crop.width, PANEL[1] / crop.height)
            size = tuple(max(1, round(v * scale)) for v in crop.size)
            target = (x + (PANEL[0] - size[0]) // 2, y + 48 + (PANEL[1] - size[1]) // 2)
            canvas.paste(crop.resize(size, Image.Resampling.LANCZOS), target)
        print(row['key'], f'{found}/{eligible}', extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=94, subsampling=0, optimize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=REFERENCE)
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs/normal-fields')
    parser.add_argument('--previews', type=Path, default=REFERENCE / 'direction-previews.zip')
    parser.add_argument('--save-previews', action='store_true',
                        help='Recreate the eight original color maps from the frozen inference cache')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/normal-building-sheet.jpg')
    args = parser.parse_args()
    if args.save_previews:
        save_previews(args.cache, args.previews)
    with ZipFile(args.previews) as previews:
        if json.loads(previews.read('recipe.json')) != normal_recipe(args.run):
            raise ValueError('These surface directions were produced with different model settings')
        render(args.run, previews, args.out)
