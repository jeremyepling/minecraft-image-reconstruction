"""Draw the fixed eight-view Surface Finder review with separate, registered close-ups."""
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, read, save, digest
from benchmark.camera_sheet import label
from benchmark.current_misses_sheet import lines, scene_box
from benchmark.recovery_overview import COL, GAP, GOLD, LEFT, TOP, assessment, checked, tile
from benchmark.single_wall_sheet import color_predictions, locate
from benchmark.split_sheet import _detail
from benchmark.surface_detection_sheet import COLORS, INK, MUTED, PAPER, focus_box, paint
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


METHODS = ('current', 'ledge', 'join')
NAMES = dict(current='Current finder', ledge='Ledge cut', join='Wall-piece join')


def stages():
    rows = [('source', 'Source building', 'Original', 'Clean building crop from the fixed review view.', 238),
            ('actual', 'Visible surfaces', 'Reference', 'Gold outlines show scored surfaces. Gray marks surfaces below the size cutoff.', 238)]
    rows.extend((m, NAMES[m], 'Prediction', 'Guesses across this building. Pink marks guesses without a match.', 242)
                for m in METHODS)
    rows.extend(('grade-' + m, NAMES[m], 'Scored reference', 'Green marks surfaces found. Amber marks misses.', 242)
                for m in METHODS)
    rows.extend([('locate', 'Find the close-up', 'Original', 'The white box locates the detail below. Each caption identifies its view.', 247),
                 ('crop', 'Original pixels', 'Close-up', 'This crop stays fixed for the remaining rows in its column.', 247),
                 ('target', 'Actual surface', 'Reference close-up', 'Gold marks only the selected surface. Scores use its full outline.', 247)])
    rows.extend(('detail-' + m, NAMES[m], 'Prediction close-up', 'Cyan shows the paired guess, or the closest guess for a miss.', 247)
                for m in METHODS)
    return rows


def describe(row):
    floor, frame = row['key'].split('/')
    variant = frame.split(row['layout'] + '-', 1)[1].replace('-', ' ')
    return variant + ' / ' + floor.replace('_', ' ')


def picture(out, row, case):
    checked(case.path('image'), row['sha256'])
    with Image.open(case.path('image')) as source:
        photo = source.convert('RGB')
    with gzip.open(out / row['predictions'], 'rt', encoding='utf8') as stream:
        parts = json.load(stream)
    _, ref = recorded(case)
    graded = {}
    for method in METHODS:
        graded[method], native = assessment(case, ref, parts[method], set(row['scopes'][method]['matched']))
    return photo, ref, native, graded


def changes(row):
    baseline = set(row['scopes']['current']['matched'])
    return {m: baseline ^ set(row['scopes'][m]['matched']) for m in METHODS[1:]}


def choose_detail(candidates, fixed):
    # Use grades only to choose an illustration after every method has produced its predictions.
    def priority(row):
        changed = changes(row)
        different = len(changed['ledge'] | changed['join'])
        baseline = row['scores']['current']
        return (different > 0, baseline['matched'] < baseline['eligible'], row['key'] == fixed,
                different, baseline['eligible'] - baseline['matched'])
    return max(sorted(candidates, key=lambda r: r['key']), key=priority)


def choose_target(data, row):
    eligible = [p for p in data['current']['grading']['references'] if p['eligible']]
    changed = changes(row)
    def rank(p):
        rid = p['id']
        return (rid in changed['ledge'] | changed['join'], rid not in data['current']['found'],
                p['pixels'], -rid)
    rid = max(eligible, key=rank)['id']
    if rid in changed['ledge'] | changed['join']:
        return rid, 'Changed match'
    return rid, 'Current miss' if rid not in data['current']['found'] else 'Matched example'


def overview_panels(put, full, row, view):
    photo, ref, native, graded = full
    scene = tuple(view['scene_xyxy'])
    eligible = {i for i, patch in ref['patches'].items() if patch['eligible']}
    put('source', photo.crop(scene), 'Fixed review view', describe(row))
    actual = paint(photo, native, {i: GOLD if i in eligible else COLORS['small'] for i in ref['patches']})
    put('actual', actual.crop(scene), f'{len(ref["patches"])} visible / {len(eligible)} scored',
        f'{len(ref["patches"]) - len(eligible)} below the size cutoff')
    summary = {}
    for m in METHODS:
        result = graded[m]
        found = result['found']
        counts = result['grading']['counts']
        put(m, color_predictions(photo, result).crop(scene),
            f'{len(found)} found / {len(eligible)} scored', f'{counts["proposed"]} guesses · {counts["unpaired"]} extra')
        truth = paint(photo, native, {p['id']: COLORS[p['status']] for p in result['grading']['references']})
        base = graded['current']['found']
        change = 'Current finder' if m == 'current' else f'{len(found-base)} gained · {len(base-found)} lost vs current'
        put('grade-' + m, truth.crop(scene), f'{len(eligible - found)} missed in this view', change)
        summary[m] = dict(found=sorted(found), missed=sorted(eligible - found), counts=counts)
    return dict(frame=row['key'], sha256=row['sha256'], scene_xyxy=scene, methods=summary)


def detail_panels(put, full, row, overview, pinned):
    photo, _, native, graded = full
    if pinned:
        if pinned['sha256'] != row['sha256'] or pinned['frame'] != row['key']:
            raise ValueError('Pinned close-up uses a different screenshot')
        rid = pinned['reference']
        reason = 'Revisited miss'
        box, scene = tuple(pinned['crop_xyxy']), tuple(pinned['scene_xyxy'])
    else:
        rid, reason = choose_target(graded, row)
        box = tuple(map(int, focus_box(native, rid, photo.size)))
        scene = scene_box(native, box, photo.size)
    same = row['key'] == overview
    view_note = 'Same view as overview' if same else 'Different view for close-up'
    put('locate', locate(photo, box).crop(scene), view_note, describe(row))
    put('crop', photo.crop(box), f'{reason} · R{rid}', 'Same original pixels in the four rows below')
    put('target', paint(photo, native, {rid: GOLD}).crop(box), f'Recorded surface R{rid}', 'Gold marks its visible footprint')
    paired = {}
    base_found = rid in graded['current']['found']
    for m in METHODS:
        result = graded[m]
        near = result['lookup'][rid]
        i = near['index']
        image, shown = _detail(photo, result['parts'][i - 1] if i else None, box)
        found = near['found']
        state = 'Found' if found else 'Missed'
        comparison = ('Current finder' if m == 'current' else
                      'Newly found' if found and not base_found else 'Match lost' if base_found and not found else
                      'Still found' if found else 'Still missed')
        put('detail-' + m, image, f'{state} R{rid} · {near["overlap"]:.0%} overlap', comparison)
        paired[m] = dict(**near, visible_in_crop=shown)
    return dict(frame=row['key'], sha256=row['sha256'], different_view=not same, focus_reference=rid,
                focus_reason=reason, crop_xyxy=box, scene_xyxy=scene, methods=paired)


@image_only()
def render(out, rows, captures, table):
    out = Path(out)
    reviews = ROOT / 'benchmark/reviews'
    roster_file = reviews / 'recovery-01/sheet-views.json'
    misses_file = reviews / 'current-misses-01/manifest.json'
    old_manifest = read(roster_file.parent / 'manifest.json')
    checked(roster_file, old_manifest['artifacts'][roster_file.name])
    checked(ARCHIVE, old_manifest['reference_sha256'])
    roster = read(roster_file)['views']
    by_key = {r['key']: r for r in rows}
    pinned = {by_key[p['frame']]['layout']: p for p in read(misses_file)['examples']}
    if (len(roster) != 8 or len({v['layout'] for v in roster}) != 8 or len(pinned) != 4 or
            set(by_key) != set(captures) or len(by_key) != len(rows) or len(rows) != 56):
        raise ValueError('Review inputs differ from the eight fixed buildings and 56 frames')
    specs = stages()
    width = LEFT + len(roster) * (COL + GAP) + GAP
    height = TOP + sum(h + 82 for _, _, _, _, h in specs) + 9
    page = Image.new('RGB', (width, height), PAPER)
    draw = ImageDraw.Draw(page)
    label(draw, (16, 16), 'Did the ledge cut or the wall-piece join recover more surfaces?', 39, bold=True)
    totals = {m: table['all'][m]['matched'] for m in METHODS}
    scored = table['all']['current']['eligible']
    line = '   /   '.join(f'{NAMES[m]} {totals[m]} of {scored}' for m in METHODS)
    label(draw, (18, 74), line + '     ·     56 saved views', 22)
    legends = [(GOLD, 'Actual surface'), (COLORS['finder'], 'Prediction'), (COLORS['extra'], 'Extra guess'),
               (COLORS['found'], 'Found'), (COLORS['missed'], 'Missed'), (COLORS['small_found'], 'Small match')]
    for index, (color, title) in enumerate(legends):
        x = 20 + 221 * index
        draw.rectangle((x, 122, x + 18, 140), fill=color, outline=MUTED)
        label(draw, (x + 26, 121), title, 18)
    label(draw, (1390, 121), 'A match needs at least 50% overlap across the whole surface. Each guess can match once.', 18)
    starts = {}
    y = TOP
    for key, title, kind, note, h in specs:
        starts[key] = (y, h)
        draw.line((14, y, width - 14, y), fill='#abc0b7', width=2)
        label(draw, (15, y + 17), kind.upper(), 15, MUTED, True)
        lines(draw, (15, y + 52), title, LEFT - 28, 23, INK, True)
        lines(draw, (15, y + 116), note, LEFT - 37, 17, MUTED)
        y += h + 82
    views = []
    for index, view in enumerate(roster):
        layout, key = view['layout'], view['frame']
        row = by_key[key]
        if view['sha256'] != row['sha256'] or layout != row['layout']:
            raise ValueError(f'Fixed review view changed for {key}')
        x = LEFT + index * (COL + GAP)
        label(draw, (x, TOP - 82), layout.title(), 27, bold=True)
        cohort = 'Development' if row['cohort'] == 'development' else 'Reviewed regression'
        label(draw, (x, TOP - 44), cohort, 16, '#12655c' if cohort == 'Development' else '#69558e', True)
        def put(key, image, first, second):
            y, h = starts[key]
            tile(page, x, y, h, image, first, second)
        data = picture(out, row, captures[key])
        scene = overview_panels(put, data, row, view)
        pin = pinned.get(layout)
        detail = (by_key[pin['frame']] if pin else
                  choose_detail([r for r in rows if r['layout'] == layout], key))
        if detail['key'] != key:
            data = picture(out, detail, captures[detail['key']])
        selection = detail_panels(put, data, detail, key, pin)
        views.append(dict(layout=layout, cohort=row['cohort'], overview=scene, close_up=selection))
    path = out / 'comparison.jpg'
    page.save(path, quality=93, subsampling=0, optimize=True)
    save(out / 'sheet-views.json', dict(views=views, methods=NAMES, fixed_views_sha256=digest(roster_file),
         prior_misses_sha256=digest(misses_file), reference_sha256=digest(ARCHIVE), matched=totals, eligible=scored,
         definition='Scores and gained/lost status use full surfaces, including regions beyond an enlarged crop.',
         selection='Four pinned regions from the prior misses review. For other buildings prefer changed match IDs, '
                   'then baseline misses and the fixed overview view. Select only after scoring.',
         image=dict(file=path.name, width=width, height=height, sha256=digest(path))))
    return path
