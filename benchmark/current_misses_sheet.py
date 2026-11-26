"""Draw four missed surfaces using saved predictions, original pixels and recorded geometry."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.camera_sheet import font, label
from benchmark.mask_metrics import footprint
from benchmark.missed_surface_index import build_index
from benchmark.normal_round import groups
from benchmark.recovery_overview import assessment, checked
from benchmark.single_wall_sheet import load_gzip, locate
from benchmark.split_sheet import _detail
from benchmark.surface_detection_sheet import COLORS, focus_box, paint
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


REVIEWS = ROOT / 'benchmark/reviews'
SOURCE = REVIEWS / 'single-wall-depth-01'
MOBILE = REVIEWS / 'mobile-regions-01'
METHOD = 'single-v2'
# These four cases illustrate scored errors. They are never read by the predictor.
EXAMPLES = [
    ('light_gray_concrete/dev-shopfront-plaster-high-right', 5, 'Wall around the bays'),
    ('smooth_stone/dev-wing-plaster-high-right', 2, 'Wide plaster wall'),
    ('light_gray_concrete/holdout-workshop-masonry-low-left', 13, 'Front of the stone ledge'),
    ('light_gray_concrete/holdout-galleries-masonry-low-left', 13, 'Underside of the beam'),
]
ROWS = [
    ('Find the spot', 'Original screenshot', 'The white box locates the close-up.'),
    ('Misses in view', 'Reference', 'Amber marks every scored surface missed in this picture.'),
    ('Original pixels', 'Close-up', 'Compare the bare texture and shadows with the outlines below.'),
    ('Actual surface', 'Reference', 'Amber marks the visible parts of one recorded surface.'),
    ('Closest guess', 'Prediction', 'Cyan shows the single guess with the best overlap.'),
    ('What survived', 'Diagnostic', 'Purple shows evidence selected after scoring.'),
]
NAVY, INK, MUTED = '#172d33', '#183539', '#455e60'
PAPER, FRAME, PURPLE = '#f8f6ef', '#e5e9e3', '#ce97ff'
W, LEFT, COL, GAP = 1968, 212, 424, 12
TOP, STEP, VIEW_H = 286, 335, 261


def overlap(mask, target):
    return float((mask & target).sum() / (mask | target).sum())


def evidence(row, rid, ref, masks, sam, mobile_hashes, run_hashes):
    target = ref['id_map'] == rid
    layout = row['layout']
    if layout == 'shopfront':
        names = ('mask-1-wall-1', 'mask-3')
        shape = masks[names[0]] | masks[names[1]]
        score = overlap(shape, target)
        return shape, (f'Two pieces together · {score:.0%} overlap',
                       'The wall continues around the bays.'), dict(parts=names, overlap=score)
    if layout == 'galleries':
        shape = np.logical_or.reduce(list(masks.values())) & target
        coverage = float(shape.sum() / target.sum())
        return shape, (f'{coverage:.0%} of the underside was touched',
                       'Purple marks target pixels claimed by any guess.'), dict(target_coverage=coverage)
    candidate = sam['pool_matches'][str(rid)]
    index = candidate['proposal']
    base = 'proposals/' + row['sha256']
    checked(MOBILE / (base + '.npz'), mobile_hashes[base + '.npz'])
    with np.load(MOBILE / (base + '.npz'), allow_pickle=False) as data:
        shape = np.unpackbits(data['packed'][index], axis=1, count=int(data['shape'][2])).astype(bool)
    score = overlap(shape, target)
    if abs(score - candidate['iou']) > 1e-6:
        raise ValueError(f'Mismatched model outline in {row["key"]}')
    info = dict(mobile_proposal=index, overlap=score)
    if layout == 'wing':
        record = load_gzip(MOBILE, base + '.json.gz', mobile_hashes)['candidates'][index]
        fraction = record['building_fraction']
        info['inside_starting_outline'] = fraction
        note = f'{fraction:.0%} lay inside the starting building outline.'
    else:
        votes = load_gzip(SOURCE, row['boundary_evidence'], run_hashes)['events'][METHOD]
        vote, = [v for v in votes if v['proposal'] == index and v['parent'] == 'mask-0-mobile-1']
        if vote['accepted']:
            raise ValueError('The illustrated cut was accepted')
        agrees = round(vote['sample_count'] * vote['sign_agreement'])
        note = f'Only {agrees} of {vote["sample_count"]} depth samples agreed.'
        info['vote'] = vote
    return shape, (f'Earlier outline · {score:.0%} overlap', note), info


def scene_box(native, detail, size):
    x0, y0, x1, y1 = Image.fromarray(native > 0).getbbox()
    pad = max(24, (x1 - x0) // 12)
    return (max(0, min(detail[0], x0) - pad), max(0, min(detail[1], y0) - pad),
            min(size[0], max(detail[2], x1) + pad), min(size[1], max(detail[3], y1) + pad))


def column(case, row, rid, sam, mobile_hashes, hashes):
    checked(case.path('image'), row['sha256'])
    with Image.open(case.path('image')) as im:
        photo = im.convert('RGB')
    _, ref = recorded(case)
    parts = load_gzip(SOURCE, row['predictions'], hashes)[METHOD]
    result, native = assessment(case, ref, parts, set(row['scopes'][METHOD]['matched']))
    missed = {i for i, p in ref['patches'].items() if p['eligible']} - result['found']
    closest = result['lookup'][rid]
    if rid not in missed or not closest['index']:
        raise ValueError(f'Expected a missed target with a visible guess in {row["key"]}')
    masks = {p['id']: footprint(p, ref['id_map'].shape, photo.size, use_mask=True) for p in parts}
    shape, diagnosis, info = evidence(row, rid, ref, masks, sam, mobile_hashes, hashes)
    target = ref['id_map'] == rid
    coverage = float((target & masks[closest['proposal']]).sum() / target.sum())
    box = tuple(map(int, focus_box(native, rid, photo.size)))
    # Zoom into the place where the cyan trace stops, along a beam over 600 pixels wide.
    if row['layout'] == 'galleries':
        box = (796, 429, 984, 546)
    scene = scene_box(native, box, photo.size)
    crop = photo.crop(box)
    diagnostic_mask = Image.fromarray(shape.astype(np.uint8)).resize(photo.size, Image.Resampling.NEAREST)
    diagnostic = paint(crop, np.asarray(diagnostic_mask.crop(box)), {1: PURPLE})
    guess, visible = _detail(photo, parts[closest['index'] - 1], box)
    if not visible:
        raise ValueError('Closest guess missing from panel')
    amber = COLORS['missed']
    found, eligible = len(result['found']), len(missed) + len(result['found'])
    counts = result['grading']['counts']
    view = row['key'].split(row['layout'] + '-', 1)[1].replace('-', ' ')
    panels = [
        (locate(photo, box).crop(scene), f'{found} found / {eligible} scored in this view',
         f'{counts["visible"]} visible · {counts["small"]} below cutoff · {counts["proposed"]} guesses'),
        (paint(photo, native, {i: amber for i in missed}).crop(scene),
         f'{len(missed)} missed in this view', f'{view} · The rows below follow R{rid}.'),
        (crop, 'Original screenshot detail', 'Same crop in every row below.'),
        (paint(photo, native, {rid: amber}).crop(box), f'Recorded surface R{rid}', 'Visible pixels from the saved geometry.'),
        (guess, f'{closest["overlap"]:.0%} overlap with R{rid}', f'This guess covered {coverage:.0%} of the target.'),
        (diagnostic, *diagnosis),
    ]
    record = dict(frame=row['key'], sha256=row['sha256'], stage='Surface Finder', reference=rid,
                  crop_xyxy=box, scene_xyxy=scene, closest=closest, target_coverage=coverage, diagnostic=info)
    return panels, record


def lines(draw, xy, message, width, size, color, bold=False):
    x, y = xy
    words = []
    face = font(size, bold)
    for word in message.split():
        trial = ' '.join([*words, word])
        if words and draw.textlength(trial, font=face) > width:
            label(draw, (x, y), ' '.join(words), size, color, bold)
            y += size + 6
            words = []
        words.append(word)
    label(draw, (x, y), ' '.join(words), size, color, bold)


def draw_sheet(columns, totals):
    page = Image.new('RGB', (W, TOP + len(ROWS) * STEP + 115), PAPER)
    d = ImageDraw.Draw(page)
    d.rectangle((0, 0, W, 166), fill=NAVY)
    label(d, (28, 23), 'Where the Surface Finder still misses', 48, '#ffffff', True)
    label(d, (30, 89), f'{totals["found"]} of {totals["eligible"]} visible surfaces found'
          f'   /   {totals["missed"]} missed across 56 views', 27, '#e4eee8')
    label(d, (31, 133), 'Each column follows one missed surface from the original screenshot to the closest guess.', 20, '#bcd1ce')
    for j, (_, rid, title) in enumerate(EXAMPLES):
        x = LEFT + j * (COL + GAP)
        name = columns[j][1]['frame'].split('-', 2)[1].title()
        label(d, (x, 193), name, 33, INK, True)
        label(d, (x, 244), f'R{rid}  /  {title}', 22, MUTED)
    for k, (title, kind, hint) in enumerate(ROWS):
        y = TOP + k * STEP
        d.line((24, y, W - 24, y), fill='#bac9c3', width=2)
        label(d, (25, y + 32), kind, 16, MUTED, True)
        lines(d, (25, y + 69), title, LEFT - 35, 24, INK, True)
        lines(d, (25, y + 146), hint, LEFT - 44, 18, MUTED)
        for j, (panels, _) in enumerate(columns):
            x = LEFT + j * (COL + GAP)
            image, first, second = panels[k]
            d.rectangle((x, y + 12, x + COL - 1, y + 11 + VIEW_H), fill=FRAME)
            view = ImageOps.contain(image, (COL, VIEW_H), Image.Resampling.LANCZOS)
            page.paste(view, (x + (COL - view.width) // 2, y + 12 + (VIEW_H - view.height) // 2))
            label(d, (x, y + VIEW_H + 20), first, 21, INK, True)
            lines(d, (x, y + VIEW_H + 47), second, COL, 16, MUTED)
    y = TOP + len(ROWS) * STEP
    d.rectangle((0, y, W, page.height), fill=NAVY)
    label(d, (28, y + 24), 'Overlap measures shared area divided by combined area across the full picture. A match needs 50% and a unique guess.',
          21, '#ffffff')
    label(d, (28, y + 65), 'For feedback, name the building, its R number and the edge you would follow, such as the lower edge of Workshop R13.',
          20, '#c5d9d2')
    return page


@image_only()
def render(out):
    manifest = read(SOURCE / 'manifest.json')
    mobile = read(MOBILE / 'manifest.json')
    hashes = manifest['artifacts']
    checked(SOURCE / 'cases.json', hashes['cases.json'])
    checked(MOBILE / 'cases.json', mobile['artifacts']['cases.json'])
    checked(ARCHIVE, manifest['reference_sha256'])
    cohorts, _, fixtures = groups()
    cases = {f'{c.floor}/{c.id}': c for cohort in cohorts.values() for c in cohort}
    rows = read(SOURCE / 'cases.json')
    by_key = {r['key']: r for r in rows}
    sam = {r['key']: r for r in read(MOBILE / 'cases.json')}
    if (len(rows) != 56 or set(by_key) != set(cases) or set(by_key) != set(sam)
            or fixtures != manifest['fixtures'] or fixtures != mobile['fixtures']
            or manifest['reference_sha256'] != mobile['reference_sha256']):
        raise ValueError('Frozen inputs differ')
    frames, totals, index = build_index(rows, cases, SOURCE, hashes, sam)
    columns = [column(cases[key], by_key[key], rid, sam[key], mobile['artifacts'], hashes)
               for key, rid, _ in EXAMPLES]
    page = draw_sheet(columns, totals)
    out.mkdir(parents=True, exist_ok=True)
    page.save(out / 'missed-surfaces.jpg', quality=94, subsampling=0, optimize=True)
    (out / 'pictures.md').write_text(index, encoding='utf8')
    save(out / 'frames.json', frames)
    save(out / 'manifest.json', dict(run_id='current-misses-01', code=revision(), method='wall-recovery-single-v2',
         source_manifests={p.parent.name: digest(p) for p in (SOURCE/'manifest.json', MOBILE/'manifest.json')},
         fixtures=fixtures, reference_sha256=digest(ARCHIVE), totals=totals, examples=[c[1] for c in columns],
         settings='Saved production masks, official one-to-one scoring. Four illustrative errors selected after grading. '
                  'The diagnostic row uses reference geometry for selection or coverage. No inference or tuning.',
         dimensions=page.size, reproduce='python -m benchmark.current_misses_sheet --out runs/current-misses-review',
         artifacts={f: digest(out/f) for f in ('missed-surfaces.jpg', 'frames.json', 'pictures.md')}))
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/current-misses-review')
    render(parser.parse_args().out)
