"""Assemble per-frame decisions, frozen inputs and a traced Twinspire overlay."""
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.placement_grid_choice import CONTROL, REVIEW, RUN, COHORTS, fixed, grade, read, save
from benchmark.typed_structures import pooled_scores
from benchmark.placement_grid_audit import audit
from benchmark.expanded_baseline import SOURCE, digest, frames


def counts(rows):
    result = dict(views=len(rows), camera_failures=sum(not r['camera_usable'] for r in rows),
                  charts=sum(len(r['options']) for r in rows),
                  offered_combinations=sum(sum(len(v['selections']) for v in r['variants']) for r in rows),
                  distinct_placement_inputs=sum(len(r['variants']) for r in rows),
                  distinct_typed_exports=sum(len({v['result']['decoded_sha256'] for v in r['variants']}) for r in rows))
    result['required_missing_from_all_choices'] = sum(r['required_missing_from_all_choices'] for r in rows)
    for arm in ('baseline', 'original', 'source', 'oracle'):
        obj = {k: pooled_scores([r[arm][k] for r in rows]) for k in ('geometry', 'scores')}
        obj['paired'] = {}
        for kind, key in [('geometry', 'strict'), ('geometry','facade'), ('scores','strict'), ('scores','facade')]:
            diff = [r[arm][kind][key]['f1'] - r['baseline'][kind][key]['f1'] for r in rows]
            obj['paired'][kind + '_' + key] = dict(improved=sum(d > 0 for d in diff),
                                                   regressed=sum(d < 0 for d in diff),
                                                   tied=sum(d == 0 for d in diff))
        result[arm] = obj
    return result


def twin_overlay(case):
    entry = next(e for e in fixed.fixed_frames() if e['id'] == case['id'])
    with Image.open(entry['rgb']) as photo:
        src = photo.convert('RGB')
    all_points = np.array([q for r in case['options'] for q in (r['accepted'], r['original'])])
    lo = np.maximum(0, np.floor(all_points.min(axis=(0,1))-28)).astype(int)
    hi = np.minimum(src.size, np.ceil(all_points.max(axis=(0,1))+28)).astype(int)
    cropbox = (*lo.tolist(), *hi.tolist())
    width, height = (hi - lo).tolist()
    sheet = Image.new('RGB', (width * 2, height + 76), '#132029')
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 14)
    draw = ImageDraw.Draw(sheet)
    for index, (corner, title, color) in enumerate((
            ('accepted', 'Accepted retry corners', '#ffd46b'), ('original', 'Original corners', '#57dfeb'))):
        panel = src.copy()
        ink = ImageDraw.Draw(panel)
        for r in case['options']:
            points = [tuple(pt) for pt in r[corner]]
            ink.line(points + points[:1], fill=color, width=2)
            for x,y in points:
                ink.ellipse((x-2,y-2,x+2,y+2), fill=color)
            center = np.mean(points, axis=0)
            ink.text(tuple(center), '×'.join(map(str,r['dimensions']['blocks'])), font=small, fill='white',
                     stroke_width=2, stroke_fill='black', anchor='mm')
        sheet.paste(panel.crop(cropbox), (width * index, 38))
        draw.text((width * index + 10, 7), title, fill=color, font=font)
    draw.text((10, height + 43), 'Twinspire 3 · cool low-left · identical block counts in both panels',
              fill='#f1f3f3', font=small)
    sheet.save(REVIEW / 'twinspire.jpg', quality=93)


def main():
    paths = sorted((RUN/'cases').glob('*/*.json'))
    rows = [read(p) for p in paths]
    expected = {grade.FIXED_DATASET:104, **COHORTS}
    assert Counter(r['dataset'] for r in rows) == expected
    entry_map = {(r['dataset'],r['id']):r for r in [*fixed.fixed_frames(), *frames(SOURCE,COHORTS)]}
    for row in rows:
        cache = RUN/'audited'/(row['id']+'.json')
        if row['dataset'] == grade.FIXED_DATASET and cache.is_file():
            verified = read(cache)
            assert verified['options'] == row['options']
            assert [(v['selections'], v['result']['pack_sha256']) for v in verified['variants']] == [
                (v['selections'], v['result']['pack_sha256']) for v in row['variants']]
            row.update(verified)
        else:
            audit(row, entry_map[row['dataset'],row['id']])
        if row['dataset'] == grade.FIXED_DATASET:
            replay = read(CONTROL.parent/'fixed-round-controls/cases'/(row['id']+'.json'))
            assert replay['byte_match'] and replay['pack_sha256'] == row['baseline']['pack_sha256']
            row['reference_sha256'] = replay['reference_sha256']
        else:
            folder = CONTROL / row['dataset'] / row['id']
            row['placement_input_sha256'] = digest(folder/'handoffs/placement-input.json.gz')
            assert row['baseline']['pack_sha256'] == digest(folder/'reconstruction.zip')
        assert row['reference_sha256']
    REVIEW.mkdir(parents=True, exist_ok=True)
    save(REVIEW/'cases.json.gz', rows)
    summary = {ds: counts([r for r in rows if r['dataset'] == ds]) for ds in expected}
    for ds in expected:
        dataset = [r for r in rows if r['dataset'] == ds]
        summary[ds]['splits'] = {split: counts([r for r in dataset if r['split'] == split])
                                for split in sorted({r['split'] for r in dataset})}
    save(REVIEW/'summary.json', summary)
    twin_overlay(next(r for r in rows if r['id'] == 'test-twinspire3-cool-low-left'))
    save(REVIEW/'manifest.json', dict(production_revision=fixed.MEASURED_REVISION, minecraft='26.3',
         arm='original versus accepted retry chart independently per surface',
         selection='best actual per-frame facade geometry F1; stable enumeration prefers KEEP on exact baseline tie',
         control_cases_sha256=digest(CONTROL/'scores/cases.json'), control_manifest_sha256=digest(CONTROL/'scores/manifest.json'),
         case_archive_sha256=digest(REVIEW/'cases.json.gz'), summary_sha256=digest(REVIEW/'summary.json'),
         overlay_sha256=digest(REVIEW/'twinspire.jpg'), counts=expected))


if __name__ == '__main__':
    main()
