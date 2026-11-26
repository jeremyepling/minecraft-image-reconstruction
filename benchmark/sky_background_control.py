"""Score predicted surface pixels against independent high-confidence sky evidence."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_labels import screen_matrix
from benchmark.camera_round import digest, read, revision, save
from benchmark.crop_chart_coverage import SOURCE, SOURCE_MANIFEST
from benchmark.normal_round import groups
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.sky_filter import filter_surface
from engine.surface_perimeter import _native_alpha
from runtime.mode import image_only


EXAMPLES = {
    'light_gray_concrete/holdout-workshop-masonry-low-left': 'mask-1',
    'light_gray_concrete/holdout-workshop-plaster-low-left': 'mask-5',
    'light_gray_concrete/holdout-crossing-masonry-low-left': 'mask-1-single-436-0',
}


def occupied(parts, size):
    mask = np.zeros(size[::-1], bool)
    for part in parts:
        alpha, (x, y, x1, y1) = _native_alpha(part['support'], size)
        mask[y:y1, x:x1] |= np.asarray(alpha) > 0
    return mask


def diagnostic(capture, image):
    # Scoring only. Looking upward on these flat worlds avoids confusing floor with sky.
    pose, ref = recorded(capture)
    actual = _native_reference(pose, ref, read(capture.path('labels'))['faces']) > 0
    ray = np.linalg.inv(screen_matrix(pose)[:, :3])[1]
    y, x = np.ogrid[:image.height, :image.width]
    sky = (ray[0]*(x+.5) + ray[1]*(y+.5) + ray[2] > .035) & ~actual
    r, g, b = np.asarray(image, dtype=np.int16).transpose(2, 0, 1)
    cloud = sky & (r >= 140) & (g >= 150) & (g > r+5) & (b > g+10) & (b < 250)
    blue = sky & (g > r+20) & (b == 255) & (b > g+30)
    return dict(upward_background=sky, cloud_proxy=cloud, clear_blue_proxy=blue, building=actual)


def count(mask, maps):
    return {name: int(np.count_nonzero(mask & m)) for name, m in maps.items()}


def overlay(image, maps, selected):
    rgb = np.array(image)
    rgb[selected & maps['cloud_proxy']] = [255, 40, 90]
    rgb[selected & maps['clear_blue_proxy']] = [255, 195, 10]
    return Image.fromarray(rgb)


@image_only()
def run(out):
    source = read(checked(SOURCE/'manifest.json', SOURCE_MANIFEST))
    checked(ARCHIVE, source['reference_sha256'])
    cases = read(checked(SOURCE/'cases.json', source['artifacts']['cases.json']))
    cohorts, _, fixtures = groups()
    captures = {f'{c.floor}/{c.id}': c for group in cohorts.values() for c in group}
    if fixtures != source['fixtures'] or len(cases) != 56 or {r['key'] for r in cases} != captures.keys():
        raise ValueError('Different fixture set')
    records, examples = [], []
    for row in cases:
        capture = captures[row['key']]
        with Image.open(checked(capture.path('image'), row['sha256'])) as src:
            image = src.convert('RGB')
        name = row['predictions']
        parts = read_gzip(checked(SOURCE/name, source['artifacts'][name]))['split']
        after, _ = filter_surface(image, parts)
        maps = diagnostic(capture, image)
        before_mask, after_mask = (occupied(p, image.size) for p in (parts, after))
        rec = dict(frame=row['key'], sha256=row['sha256'], baseline=count(before_mask, maps),
                   filtered=count(after_mask, maps), removed=count(before_mask & ~after_mask, maps),
                   newly_claimed=count(after_mask & ~before_mask, maps))
        if row['key'] in EXAMPLES:
            pid = EXAMPLES[row['key']]
            part = next(p for p in parts if p['id'] == pid)
            region = occupied([part], image.size)
            _, box = _native_alpha(part['support'], image.size)
            left, top, right, bottom = box
            box = (max(0,left-36), max(0,top-36), min(image.width,right+36), min(image.height,bottom+36))
            rec['example'] = dict(original_id=pid, baseline=count(region, maps),
                                  residual_within_original=count(after_mask & region, maps), crop_xyxy=box)
            imgs = (image, overlay(image, maps, region), overlay(image, maps, region & after_mask))
            examples.append((row['key'], [im.crop(box) for im in imgs]))
        records.append(rec)
    page = Image.new('RGB', (1110, len(examples)*256+50), 'white')
    draw = ImageDraw.Draw(page)
    draw.text((10,4), 'Sky in the old proposal footprint. Pink = cloud-colored pixels; gold = clear blue.', fill='black')
    for i, label in enumerate(('Source','Before','After')):
        draw.text((i*370+8,27), label, fill='black')
    for i, (frame, panels) in enumerate(examples):
        y = 50+256*i
        draw.text((8,y),frame,fill='black')
        for j, im in enumerate(panels):
            im.thumbnail((365,224))
            page.paste(im,(j*370+5,y+20))
    out.mkdir(parents=True, exist_ok=False)
    page.save(out/'examples.jpg',quality=95)
    totals = {method:{name:sum(r[method][name] for r in records) for name in maps}
              for method in ('baseline','filtered','removed','newly_claimed')}
    save(out/'report.json',dict(run_id=out.name, code=revision(), baseline_manifest_sha256=SOURCE_MANIFEST,
         reference_sha256=source['reference_sha256'], fixtures=fixtures, version=source['version'],
         protocol='Score upward rays outside the native recorded building. Color subsets are conservative proxies for clouds and clear blue, not exhaustive cloud labels. Counts sum 56 frame entries.',
         totals=totals, frames=records, examples_sha256=digest(out/'examples.jpg')))
    print(totals)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    run(cli.parse_args().out)
