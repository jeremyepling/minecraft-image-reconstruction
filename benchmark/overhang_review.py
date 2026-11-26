"""Show raw evidence and saved predictions on the same eight surface-review views."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.camera_labels import rasterize
from benchmark.camera_sheet import label
from benchmark.mask_round import load_lines
from benchmark.moge_probe import normal_image
from benchmark.normal_fields import cached_fields
from benchmark.normal_round import groups
from benchmark import overhang_matches, overhang_misses, overhang_seams
from benchmark.perimeter_sheet import load_parts
from benchmark.surface_detection_maps import detection_maps
from benchmark.surface_detection_sheet import COLORS, focus_box, paint
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


REVIEW = ROOT / 'benchmark/reviews/perimeters-01'
TRACE = ROOT / 'benchmark/reviews/mask-four-01/loss-trace/surfaces.json.gz'
LINES = ROOT / 'benchmark/data/mask-lines-v1'
LEFT, COL, GAP, HEADER, ROW = 206, 342, 10, 205, 316
STAGES = [
    ('Source view', 'White box locates the crop.'),
    ('Original pixels', 'Look for shade, steps or a recess.'),
    ('Recorded surface', 'Geometry used for scoring.'),
    ('Current finder', 'One predicted region in cyan.'),
    ('Depth split', 'An alternative saved prediction.'),
    ('Earlier B', 'Its original fitted rectangle.'),
    ('MoGe-2', 'Similar hues mean similar direction.'),
    ('Relative depth', 'Lighter means closer in this crop.'),
    ('ScaleLSD', 'Raw line segments on the screenshot.'),
    ('DeepLSD', 'Raw line segments on the screenshot.'),
]


def focus(references, patches, pairings):
    eligible = [r for r in references if r['eligible']]
    undersides = [r for r in eligible if patches[r['id']]['normal'][1] == -1]
    candidates = undersides or eligible
    missed = [r for r in candidates if r['status'] == 'missed']
    if missed:
        for alternative in ('depth-split', 'current-b'):
            rescued = [r for r in missed if pairings[r['id']][alternative]['proposal'] is not None]
            if rescued:
                return max(rescued, key=lambda r: (r['pixels'], -r['id']))
        return max(missed, key=lambda r: (r['pixels'], -r['id']))
    return min(candidates, key=lambda r: (r['match_iou'], r['id']))


def detail_box(ids, rid, size):
    """Enlarge a representative stretch of a long thin face so its thickness is readable."""
    ys, xs = np.where(ids == rid)
    major = max(np.ptp(xs)+1, np.ptp(ys)+1)
    if major < 200 or len(xs) / major > 24:
        return focus_box(ids, rid, size)
    horizontal = np.ptp(xs) >= np.ptp(ys)
    along, across = (xs, ys) if horizontal else (ys, xs)
    widths = np.bincount(along)
    thick = np.where(widths >= np.percentile(widths[widths > 0], 75))[0]
    at = thick[len(thick)//2]
    middle = int(np.median(across[along == at]))
    cx, cy = (at, middle) if horizontal else (middle, at)
    x = int(np.clip(cx - 68, 0, size[0] - 136))
    y = int(np.clip(cy - 45, 0, size[1] - 90))
    return x, y, x+136, y+90


def tile(page, image, column, row, captions):
    x, y = LEFT + column * (COL + GAP), HEADER + row * ROW
    height = ROW - 77
    panel = ImageOps.contain(image, (COL, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(page)
    draw.rectangle((x, y, x + COL - 1, y + height - 1), fill='#e2e7e4')
    page.paste(panel, (x + (COL - panel.width)//2, y + (height - panel.height)//2))
    for i, msg in enumerate(captions):
        label(draw, (x, y + height + 8 + i * 25), msg, size=17)


def proposal_tile(image, pred, proposal_index, pairing, box):
    found = pairing['proposal'] is not None
    if proposal_index is None:
        return image.crop(box), ['Missed', 'No guess touches this surface.']
    iou = pairing['iou'] if found else pairing['best_iou']
    title = f"{'Found' if found else 'Missed'}   ·   {iou:.0%} outline overlap"
    detail = 'Matched to the recorded surface.' if found else 'Closest outline from this method.'
    x, y, xx, yy = box
    if not np.any(pred[y:yy,x:xx] == proposal_index):
        detail = 'No part of this outline in the crop.'
    painted = paint(image, pred, {proposal_index: COLORS['finder']})
    return painted.crop(box), [title, detail]


def local_depth(depth, size, box):
    full = Image.fromarray(depth).resize(size, Image.Resampling.BILINEAR)
    local = np.asarray(full.crop(box))
    low, high = np.percentile(local, [2, 98])
    scaled = np.clip((local - low) / max(high - low, 1.e-8), 0, 1)
    return Image.fromarray(np.rint(255 * scaled).astype('uint8')).convert('RGB')


@image_only()
def render(fields, out):
    manifest = read(REVIEW / 'manifest.json')
    original = overhang_matches.ORIGINAL
    original_rows = {r['key']:r for r in read(original/'cases.json')}
    match_views = read(out/'results.json')['views']
    seams = {r['frame']:{s['reference']:s for s in r['undersides']} for r in read(out/'lines.json')['frames']}
    expected_trace = read(original/'manifest.json')['artifact_sha256']['loss-trace/surfaces.json.gz']
    if digest(TRACE) != expected_trace:
        raise ValueError('Changed orientation trace')
    for filename in ('cases.json', 'surface-detection.json'):
        if digest(REVIEW / filename) != manifest['artifacts'][filename]:
            raise ValueError(f'Changed review file: {filename}')
    if digest(ARCHIVE) != manifest['reference_sha256']:
        raise ValueError('Changed reference maps')
    with gzip.open(TRACE, 'rt', encoding='utf8') as file:
        directions = {r['frame']: {s['patch']: s['orientation_fraction'] for s in r['surfaces']}
                      for r in json.load(file)}
    rows = {r['key']: r for r in read(REVIEW / 'cases.json')}
    sheet = read(REVIEW / 'surface-detection.json')
    line_manifests = {m: read(LINES / m / 'manifest.json') for m in ('scalelsd', 'deeplsd')}
    cohorts, _, fixtures = groups()
    if fixtures != manifest['fixtures']:
        raise ValueError('Changed fixtures')
    cases = {f'{c.floor}/{c.id}': c for cohort in cohorts.values() for c in cohort}
    views = sheet['views']
    page = Image.new('RGB', (LEFT + len(views) * (COL + GAP), HEADER + len(STAGES) * ROW), '#f8f7f1')
    draw = ImageDraw.Draw(page)
    label(draw, (18, 15), 'Where could the finder separate two surfaces?', 37, bold=True)
    label(draw, (20, 70), 'Same eight views as the detection sheet. Each column follows one recorded surface. The lower four rows show evidence available to the finder.', 22)
    label(draw, (20, 104), 'Focus favors visible undersides, then misses recovered by depth splitting or B. A thin face is enlarged at a representative stretch. Percentages grade the whole surface.', 19)
    for i, (title, note) in enumerate(STAGES):
        y = HEADER + i * ROW
        label(draw, (12, y + 8), title, 21, bold=True)
        for j, line in enumerate(textwrap.wrap(note, width=21)):
            label(draw, (12, y + 47 + 23*j), line, 16)
    details = []
    for col, view in enumerate(views):
        key = view['frame']
        case, row = cases[key], rows[key]
        sha = row['sha256']
        if digest(case.path('image')) != sha or digest(REVIEW / row['predictions']) != manifest['artifacts'][row['predictions']]:
            raise ValueError(f'Changed image or prediction for {key}')
        with Image.open(case.path('image')) as opened:
            image = opened.convert('RGB')
        maps = cached_fields(sha, fields)
        if maps is None or hashlib.sha256(maps['depth'].tobytes()).hexdigest() != row['numerical_depth_sha256']:
            raise ValueError(f'Provide the compatible saved model fields for {key}')
        _, ref = recorded(case)
        pairings = {r['id']:r['methods'] for r in match_views[key]['references']}
        target = focus(view['references'], ref['patches'], pairings)
        rid = target['id']
        normal_y = ref['patches'][rid]['normal'][1]
        face = {-1: 'underside', 0: 'vertical face', 1: 'top face'}[normal_y]
        current = sheet['method']
        evidence = {}
        for method in (current, 'depth-split'):
            parts = load_parts(REVIEW, row, None, None, method)
            actual, pred, _ = detection_maps(case, parts, row['scores'][method])
            box = detail_box(actual, rid, image.size)
            pairing = pairings[rid][method]
            chosen = pairing['proposal'] or pairing['best_proposal']
            number = next((i for i,p in enumerate(parts,1) if p['id']==chosen),None)
            panel, captions = proposal_tile(image, pred, number, pairing, box)
            tile(page, panel, col, 3 if method == current else 4, captions)
            evidence[method] = pairing
        paired_b = pairings[rid]['current-b']
        evidence['current-b'] = paired_b
        chosen = paired_b['proposal'] or paired_b['best_proposal']
        with gzip.open(original/original_rows[key]['predictions'], 'rt', encoding='utf8') as file:
            old = json.load(file)['current-b']
        ids_b = np.zeros(actual.shape, dtype=np.uint8)
        if chosen is not None:
            region = next(p for p in old if p['id']==chosen)
            quad = np.column_stack((region['quad'], np.ones(4)))
            extent, mask, *_ = rasterize(quad, *image.size)
            ids_b[extent][mask] = 1
        panel, captions = proposal_tile(image, ids_b, 1 if chosen is not None else None, paired_b, box)
        tile(page, panel, col, 5, captions)
        x = LEFT + col * (COL + GAP)
        label(draw, (x, HEADER - 66), view['layout'].title(), 25, bold=True)
        label(draw, (x, HEADER - 32), f"R{rid}   ·   {face}", 19)
        scene = image.copy()
        scene_draw = ImageDraw.Draw(scene)
        scene_draw.rectangle(box, outline='#1b363a', width=5)
        scene_draw.rectangle(box, outline='white', width=2)
        counts = view['counts']
        tile(page, scene.crop(view['scene_xyxy']), col, 0,
             [f"{counts['found']} found / {counts['scored']} scored", f"{counts['visible']} visible in the picture"])
        tile(page, image.crop(box), col, 1, ['Clean source crop'])
        recorded_crop = paint(image, actual, {rid: COLORS[target['status']]}).crop(box)
        tile(page, recorded_crop, col, 2, [f"R{rid}   ·   {face}", f"{target['pixels']:,} visible source pixels"])
        direction = normal_image(maps['normal']).resize(image.size, Image.Resampling.BILINEAR)
        agreement = directions[key][rid]
        tile(page, direction.crop(box), col, 6, [f"{agreement:.0%} of target samples point", 'within 30° of its true direction.'])
        tile(page, local_depth(maps['depth'], image.size, box), col, 7, ['Brightness scaled within this crop.', 'Depth has no block units.'])
        for i, (m, meta) in enumerate(line_manifests.items()):
            lines = load_lines(LINES, m, sha, image.size, meta)['segments']
            overlay = image.copy()
            painter = ImageDraw.Draw(overlay)
            for segment in lines:
                painter.line(segment, fill='#ff62a6', width=2)
            captions = ['Pink lines are model outputs.']
            if rid in seams[key]:
                edges = seams[key][rid]['neighbors']
                tested = sum(e['tested'] for e in edges)
                if tested:
                    fraction = sum(e[m] for e in edges) / tested
                    captions.append(f'{fraction:.0%} of shared seam samples traced.')
            tile(page, overlay.crop(box), col, 8+i, captions)
        details.append(dict(frame=key,sha256=sha,reference=rid,surface=face,crop_xyxy=list(box),
                            counts=counts,orientation_fraction=agreement,predictions=evidence))
    page.save(out/'evidence.jpg', quality=94, subsampling=0)
    save(out/'focus.json', dict(views=details, input_review='perimeters-01', inputs=dict(
        perimeter_manifest=digest(REVIEW/'manifest.json'), orientation_trace=digest(TRACE),
        raw_lines={m:digest(LINES/m/'manifest.json') for m in line_manifests})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, default=ROOT.parent/'normal-round/runs/normal-fields')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with image_only():
        overhang_matches.run(args.out)
        overhang_misses.audit(args.out/'misses.json', args.fields)
        overhang_seams.audit(args.out/'lines.json')
        render(args.fields, args.out)
        files = ('evidence.jpg', 'summary.json', 'results.json', 'lines.json', 'misses.json', 'focus.json')
        save(args.out/'manifest.json', dict(run_id=args.out.name, kind='Saved-prediction diagnostic',
             code=revision(), reference_inputs=read(args.out/'results.json')['source_manifests'],
             raw_inputs=read(args.out/'focus.json')['inputs'], fixtures=read(REVIEW/'manifest.json')['fixtures'],
             splits='20 development and 36 previously reviewed regression pictures; no tuning in this audit.',
             methods=list(overhang_matches.METHODS), settings='Saved predictor settings and checkpoints are pinned by the two reference input manifests.',
             scoring='One-to-one surface overlap of at least 0.50. Independently count traceable seam samples within 3 native pixels and 15 degrees.',
             scores='summary.json', per_frame='results.json', artifacts={f:digest(args.out/f) for f in files},
             reproduce='python -m benchmark.overhang_review --out runs/overhang-repeat'))
        print((args.out/'evidence.jpg').resolve())
