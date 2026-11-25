"""Build the portrait comparison sheet from the published predictions, with no inference."""
import argparse
import gzip
import json
from pathlib import Path
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.camera_sheet import label
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.overhang_review import detail_box
from benchmark.split_sheet import _detail, _lookup_scores
from benchmark.surface_detection_maps import detection_maps
from benchmark.surface_detection_sheet import BG, COLORS, INK, MUTED, PAPER, paint
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


REVIEWS = ROOT / 'benchmark/reviews'
BASE = 'current'
JUNCTION = 'junction-then-underside'
MOBILE = 'mobile-walls'
COMBINED = 'mobile-combined'
NAMES = {BASE:'Wall Separator', 'wall-and-underside':'Wall + underside', 'junction-cuts':'Junction splits',
         JUNCTION:'Junction + underside', MOBILE:'Wall + MobileSAM', COMBINED:'All three refinements'}
NOTES = {BASE:'Separates walls using relative depth.',
         'wall-and-underside':'Adds narrow underside regions to the wall masks.',
         'junction-cuts':'Splits walls at supported line junctions.',
         JUNCTION:'Splits walls at junctions, then looks for narrow undersides.',
         MOBILE:'Splits walls with compatible MobileSAM region proposals.',
         COMBINED:'Adds MobileSAM splits to the junction and underside result.'}
LEFT, COL, GAP, TOP = 236, 395, 12, 248
GOLD = '#fdbf4b'


def checked(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Changed artifact: {path}')
    return actual


def index_run(root, mobile=False):
    cases, hashes = {}, {}
    for split in (('development', 'regression') if mobile else ('',)):
        folder = root / split
        manifest = read(folder / 'manifest.json')
        cases_sha = checked(folder / 'cases.json', manifest['artifacts']['cases.json'])
        name = f'{split}/' if split else ''
        hashes[name + 'cases.json'] = cases_sha
        hashes[name + 'manifest.json'] = digest(folder / 'manifest.json')
        for row in read(folder / 'cases.json'):
            if row['key'] in cases:
                raise ValueError(f'Duplicate frame in {root}: {row["key"]}')
            sha = row['prediction_sha256'] if mobile else manifest['artifacts'][row['predictions']]
            cases[row['key']] = (row, sha)
    return dict(root=root, cases=cases, hashes=hashes)


def predictions(run, frame, sha, method):
    row, mask_sha = run['cases'][frame]
    if row['sha256'] != sha:
        raise ValueError(f'Source differs in {method} for {frame}')
    path = run['root'] / row['predictions']
    checked(path, mask_sha)
    with gzip.open(path, 'rt') as f:
        masks = json.load(f)
    if method in (MOBILE, COMBINED):
        parts, matched = masks, row['assignments'].values()
    elif method == JUNCTION:
        parts, matched = masks[method], row['methods'][method]['found']
    else:
        parts, matched = masks[method], row['scopes'][method]['matched']
    return parts, set(matched), dict(run=run['root'].name, path=row['predictions'], sha256=mask_sha)


def suite_counts(runs):
    roster = {k:r['sha256'] for k,(r,_) in runs[BASE]['cases'].items()}
    counts = {}
    for method, run in runs.items():
        rows = [r for r,_ in run['cases'].values()]
        if {r['key']:r['sha256'] for r in rows} != roster:
            raise ValueError(f'Suite frame IDs or source hashes differ for {method}')
        if method in (MOBILE, COMBINED):
            scores = [r['split'] for r in rows]
        elif method == JUNCTION:
            scores = [r['methods'][method]['totals'] for r in rows]
        else:
            scores = [r['scores'][method] for r in rows]
        counts[method] = dict(frames=len(rows), found=sum(s['matched'] for s in scores),
                              eligible=sum(s['eligible'] for s in scores))
    return counts


def assessment(case, reference, parts, expected):
    scores, _ = measure(parts, reference, use_mask=True)
    native, mapped, grading = detection_maps(case, parts, scores)
    found = {r['id'] for r in grading['references'] if r['status'] == 'found'}
    if found != expected:
        raise ValueError(f'Saved match IDs differ for {case.id}')
    by_surface = _lookup_scores(parts, reference, grading['references'])
    return dict(parts=parts, mapped=mapped, grading=grading, found=found, lookup=by_surface), native


def choose_focus(methods, reference, previous):
    baseline = methods[BASE]
    refs = [r for r in baseline['grading']['references'] if r['eligible']]
    def priority(r):
        rid = r['id']
        newly = sum(rid in m['found'] for m in methods.values()) if rid not in baseline['found'] else 0
        merged = (r['coverage'] >= .5 and r['best_prediction_matched_ref'] not in (None, rid))
        return (3 if newly else 2 if rid not in baseline['found'] and merged else
                1 if rid not in baseline['found'] else 0, reference['patches'][rid]['normal'][1] == 0,
                newly, rid == previous, r['pixels'], -rid)
    best = max(refs, key=priority)
    role = ['Already found', 'Still missed', 'Merged in baseline', 'Recovered surface'][priority(best)[0]]
    return best['id'], role


def row_specs(methods):
    rows = [('source', 'Source screenshot', 'Original', 'The complete saved frame, with its original camera.', 234),
            ('scene', 'Source building', 'Original', 'The same building crop used in every overview.', 268),
            ('reference', 'Recorded surfaces', 'Reference', 'Gold shows scored surfaces. Gray marks surfaces below the scoring size. The box locates the close-up.', 268)]
    rows.extend((m, NAMES[m], 'Predictions', NOTES[m], 268) for m in methods)
    rows.extend([('crop', 'Original pixels', 'Close-up', 'One fixed crop per column. Look for a depth step, recess or gap.', 268),
                 ('target', 'Recorded target', 'Reference close-up', 'Gold marks the single surface being compared in the rows below.', 268)])
    rows.extend(('detail-' + m, NAMES[m], 'Prediction close-up',
                 'Cyan shows the matching region, or the closest guess on a miss. Changes compare with Wall Separator.',
                 268) for m in methods)
    return rows


def tile(page, x, y, h, image, first, second=''):
    draw = ImageDraw.Draw(page)
    y += 12
    draw.rectangle((x, y, x + COL - 1, y + h - 1), fill=BG)
    resized = ImageOps.contain(image, (COL, h), Image.Resampling.LANCZOS)
    page.paste(resized, (x + (COL-resized.width)//2, y + (h-resized.height)//2))
    label(draw, (x, y+h+8), first, 18, bold=True)
    label(draw, (x, y+h+34), second, 16, fill=MUTED)


def ids(values):
    return ', '.join(f'R{n}' for n in sorted(values))


def delta_label(gained, lost):
    changes = []
    if gained:
        changes.append('gained ' + ids(gained))
    if lost:
        changes.append('lost ' + ids(lost))
    return 'In this crop, ' + ' / '.join(changes) if changes else 'Same matched IDs in this crop'


def draw_column(page, x, starts, view, case, methods, runs):
    key, sha = view['frame'], view['sha256']
    checked(case.path('image'), sha)
    with Image.open(case.path('image')) as opened:
        photo = opened.convert('RGB')
    _, ref = recorded(case)
    results, files = {}, {}
    for m in methods:
        parts, expected, files[m] = predictions(runs[m], key, sha, m)
        results[m], native = assessment(case, ref, parts, expected)
    rid, reason = choose_focus(results, ref, view['focus_reference'])
    scene = tuple(view['scene_xyxy'])
    box = tuple(int(v) for v in detail_box(native, rid, photo.size))
    inside = set(np.unique(native[box[1]:box[3], box[0]:box[2]]).tolist()) - {0}
    eligible = {i for i,p in ref['patches'].items() if p['eligible']}
    baseline = results[BASE]['found']
    draw = ImageDraw.Draw(page)
    label(draw, (x, TOP-84), view['layout'].title(), 27, bold=True)
    variant = key.split(view['layout'] + '-')[-1].replace('-', ' ')
    label(draw, (x, TOP-47), variant + '  /  ' + key.split('/')[0].replace('_',' '), 16, fill=MUTED)
    def put(row, image, first, second=''):
        y, h = starts[row]
        tile(page, x, y, h, image, first, second)
    put('source', photo, 'Saved source', f'{photo.width} × {photo.height} pixels')
    put('scene', photo.crop(scene), f'{len(ref["patches"])} visible surfaces', f'{len(eligible)} pass the scoring size cutoff')
    truth = paint(photo, native, {i:GOLD if i in eligible else COLORS['small'] for i in ref['patches']})
    locator = ImageDraw.Draw(truth)
    locator.rectangle(box, outline=INK, width=5)
    locator.rectangle(box, outline='white', width=2)
    put('reference', truth.crop(scene), f'{len(eligible)} scored  /  {len(ref["patches"])-len(eligible)} small',
        f'White box locates the close-up of R{rid}')
    put('crop', photo.crop(box), reason, 'Same native crop for all detail rows')
    put('target', paint(photo, native, {rid:GOLD}).crop(box), f'R{rid}  /  recorded target',
        'The score uses the entire surface.')
    summary = {}
    for m, result in results.items():
        found = result['found']
        counts = result['grading']['counts']
        statuses = {p['id']:COLORS['extra'] if p['status']=='extra' else
                    COLORS['small_found'] if p['status']=='small_found' else COLORS['finder']
                    for p in result['grading']['predictions']}
        outlines = paint(photo, result['mapped'], statuses)
        put(m, outlines.crop(scene), f'{len(found)} found / {len(eligible)} scored  ·  {len(eligible-found)} missed',
            f'{counts["proposed"]} full-frame guesses  ·  {counts["unpaired"]} unpaired')
        focus = result['lookup'][rid]
        pi = focus['index']
        close, visible = _detail(photo, result['parts'][pi-1] if pi else None, box)
        gained, lost = (found-baseline)&inside, (baseline-found)&inside
        if rid in found:
            state = 'New match' if rid not in baseline else 'Found'
        else:
            state = 'Lost match' if rid in baseline else 'Missed'
        put('detail-' + m, close, f'{state} R{rid}  ·  {focus["overlap"]:.0%} overlap',
            'Wall Separator baseline' if m==BASE else delta_label(gained, lost))
        summary[m] = dict(found=sorted(found), missed=sorted(eligible-found), counts=counts, focus=focus,
                          focus_outline_in_crop=visible, gained_in_crop=sorted(gained), lost_in_crop=sorted(lost))
    return dict(frame=key, layout=view['layout'], sha256=sha, scene_xyxy=scene, crop_xyxy=box,
                focus_reference=rid, focus_reason=reason, reference_ids_in_crop=sorted(inside),
                visible=len(ref['patches']), eligible=sorted(eligible), methods=summary, prediction_files=files)


@image_only()
def write_sheet(args):
    recovery = index_run(args.recovery)
    junction = index_run(args.junction)
    mobile = index_run(args.mobile, mobile=True)
    runs = {BASE:recovery, 'wall-and-underside':recovery, 'junction-cuts':recovery, JUNCTION:junction, MOBILE:mobile}
    if args.combined:
        runs[COMBINED] = index_run(args.combined, mobile=True)
    methods = list(runs)
    totals = suite_counts(runs)
    roster = read(args.recovery / 'sheet-views.json')['views']
    manifest = read(args.recovery / 'manifest.json')
    checked(args.recovery / 'sheet-views.json', manifest['artifacts']['sheet-views.json'])
    checked(ARCHIVE, manifest['reference_sha256'])
    cohorts, _, fixtures = groups()
    if fixtures != manifest['fixtures'] or len(roster)!=8 or len({v['layout'] for v in roster})!=8:
        raise ValueError('The fixed eight review views or their fixtures changed')
    lookup = {f'{c.floor}/{c.id}':c for cohort in cohorts.values() for c in cohort}
    rows = row_specs(methods)
    width = LEFT + len(roster)*(COL+GAP) + GAP
    height = TOP + sum(h+82 for _,_,_,_,h in rows) + 10
    page = Image.new('RGB', (width, height), PAPER)
    draw = ImageDraw.Draw(page)
    label(draw, (16, 19), 'Which visible surfaces did each method find?', 41, bold=True)
    label(draw, (18, 77), 'Eight fixed views  /  Java 26.3  /  Clear weather at noon.  To count a match, at least half the area covered by either outline must be shared. Each surface can match once.', 21)
    legends = [(GOLD,'Recorded surface'),(COLORS['finder'],'Matched prediction'),(COLORS['extra'],'Unpaired guess'),
               (COLORS['small_found'],'Small surface paired')]
    x = 20
    for color, title in legends:
        draw.rectangle((x, 123, x+18, 141), fill=color, outline=MUTED)
        label(draw, (x+28, 121), title, 18, bold=True)
        x += 260
    starts = {}
    y = TOP
    for key, title, kind, note, h in rows:
        starts[key] = (y,h)
        draw.line((14, y, width-14, y), fill='#a9bdb5', width=2)
        label(draw, (16, y+17), kind, 17, fill=MUTED)
        for j,line in enumerate(wrap(title, width=17)):
            label(draw, (16, y+51+j*29), line, 23, bold=True)
        for j,line in enumerate(wrap(note, width=22)):
            label(draw, (16, y+126+j*24), line, 16, fill=MUTED)
        if key in totals:
            total = totals[key]
            label(draw, (16, y+245), f'Across {total["frames"]} pictures', 16, fill=MUTED)
            label(draw, (16, y+272), f'{total["found"]} of {total["eligible"]} found', 20, bold=True)
        y += h+82
    frames = [draw_column(page, LEFT+i*(COL+GAP), starts, v, lookup[v['frame']], methods, runs)
              for i,v in enumerate(roster)]
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / 'comparison.jpg'
    page.save(dest, quality=93, subsampling=0, optimize=True)
    inputs = {v['root'].name:v['hashes'] for v in runs.values()}
    inputs[args.recovery.name]['sheet-views.json'] = manifest['artifacts']['sheet-views.json']
    command = 'python -m benchmark.recovery_overview'
    if args.combined:
        command += ' --combined benchmark/reviews/' + args.combined.name
    save(args.out/'manifest.json', dict(run_id='recovery-overview-01', code=revision(), source_runs=inputs,
        version=manifest['version'], fixtures=fixtures, time_and_weather=manifest['capture'],
        reference_sha256=manifest['reference_sha256'], frame_selection='Same eight views as recovery-01.',
        focus_selection='For illustration after scoring, prioritize a new match, then a persistently merged or missed surface. Prefer vertical faces, broader recovery across candidates, then previous focus and visible area.',
        crop_changes='Compared with Wall Separator. A changed eligible ID is in the crop if any of its native reference pixels are visible there.',
        methods={m:NAMES[m] for m in methods}, displayed_methods=methods, suite_counts=totals, views=frames, image=dict(file=dest.name, sha256=digest(dest),
            width=width, height=height), reproduce=command))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', type=Path, default=REVIEWS/'recovery-01')
    parser.add_argument('--junction', type=Path, default=REVIEWS/'junction-underside-01')
    parser.add_argument('--mobile', type=Path, default=REVIEWS/'mobile-walls-01')
    parser.add_argument('--combined', type=Path)
    parser.add_argument('--out', type=Path, default=REVIEWS/'recovery-overview-01')
    write_sheet(parser.parse_args())


if __name__ == '__main__':
    main()
