"""Draw the source, crop and sampling window for fixed matches from the split-only zoom run."""
import argparse
from pathlib import Path
from textwrap import wrap
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.camera_sheet import label
from benchmark.count_metrics import reference_planes
from benchmark.current_misses_sheet import scene_box
from benchmark.ledge_join_sheet import picture
from benchmark.normal_round import groups
from benchmark.recovery_overview import checked
from benchmark.surface_detection_sheet import paint
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from benchmark.surface_reference import ARCHIVE
from engine import masked_count, surface_perimeter
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only


SOURCE = ROOT/'benchmark/reviews/crop-split-01'
# The recorded IDs select illustrations for human review after predictions have been frozen.
TARGETS = dict(shopfront=(8, 2, 13), wing=(2, 7, 3), courtyard=(10, 11, 4), portico=(3, 9, 6),
               workshop=(5, 12, 8), crossing=(14, 7, 9), galleries=(27, 16, 8), registry=(4, 3, 5))
ALTERNATE_VIEWS = {('shopfront',1):'smooth_stone/dev-shopfront-masonry-low-left',
                   ('workshop',2):'light_gray_concrete/holdout-workshop-masonry-high-right'}
STAGES = [
    ('Source', 'White box locates the crop in the saved screenshot.'),
    ('Reference', 'Gold marks the recorded visible surface.'),
    ('Surface Finder', 'Cyan follows the predicted perimeter, including holes.'),
    ('Context crop', 'Original pixels from the saved crop window, with neighbors.'),
    ('Sampling window', 'Violet shows the four corners used for straightening. Cyan is the predicted surface.'),
    ('Straightened crop', 'Actual image passed to the Block Counter. Dark pixels are masked.'),
    ('Block Counter', 'Gold shows the grid where a repeat and its position were resolved.'),
]
LEFT, COL, GAP = 239, 425, 16
TOP, BAND, CELL, PIC = 175, 70, 242, 184
PAPER, INK, MUTED = '#faf9f3', '#1b3035', '#486264'
CYAN, GOLD, VIOLET = '#49ffff', '#f9c347', '#b977ff'


def clipped_box(points, size, pad):
    low = np.maximum(0, np.floor(np.min(points, axis=0)).astype(int)-pad)
    high = np.minimum(size, np.ceil(np.max(points, axis=0)).astype(int)+pad)
    return list(map(int, np.r_[low, high]))


def inside_chart(alpha, bounds, quad):
    """Test the predicted mask's native source-pixel centers against the convex sampling quad."""
    yy, xx = np.nonzero(np.asarray(alpha))
    xx = xx + bounds[0] + .5
    yy = yy + bounds[1] + .5
    signs = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        signs.append((b[0]-a[0])*(yy-a[1]) - (b[1]-a[1])*(xx-a[0]))
    return float(np.mean(np.all(np.asarray(signs)>=-1.e-8, axis=0) | np.all(np.asarray(signs)<=1.e-8, axis=0)))


def panels(photo, part, rid, native, whole_size, iou, scene):
    cropped, tight = surface_perimeter.masked_crop(photo, part)
    context, window, local = surface_perimeter.context_crop(photo, part, mask_box=tight)
    outlined = photo.copy()
    surface_perimeter.draw_outline(outlined, part, CYAN, 2)
    sample = sample_surface(photo, part)
    quad = None if sample is None else sample['quad']
    points = [tight[:2], tight[2:], *(quad or ())]
    points.extend((window[:2], window[2:]))
    ref_bounds = Image.fromarray(native==rid).getbbox()
    detail = clipped_box([*points, ref_bounds[:2], ref_bounds[2:]], photo.size, 6)
    locator = photo.copy()
    ImageDraw.Draw(locator).rectangle((window[0], window[1], window[2]-1, window[3]-1), outline='white', width=3)
    truth = paint(photo, native, {rid:GOLD})
    chart = outlined.copy()
    frac = None
    if quad:
        q = list(map(tuple, quad))
        pen = ImageDraw.Draw(chart)
        pen.line([*q, q[0]], fill='#35163f', width=4)
        pen.line([*q, q[0]], fill=VIOLET, width=2)
        frac = inside_chart(cropped.getchannel('A'), tight, np.asarray(quad))
    panels = [
        (locator.crop(scene), 'Location in building'),
        (truth.crop(detail), f'Recorded R{rid}   ·   {whole_size[0]} × {whole_size[1]} blocks'),
        (outlined.crop(detail), f'Outline overlap with R{rid}   {iou:.0%}'),
        (context, f'{context.width} × {context.height} source pixels'),
        (chart.crop(detail), f'{frac:.0%} of predicted mask inside chart' if quad else 'No usable sampling window'),
    ]
    bounds = dict(tight_xyxy=list(tight), crop_xyxy=list(window), detail_xyxy=detail, scene_xyxy=list(scene),
                  margins_ltrb=[local[0], local[1], context.width-local[2], context.height-local[3]])
    record = dict(reference=rid, reference_size_blocks=whole_size, proposal=part['id'], finder_iou=iou,
                  quad_native=quad, predicted_mask_inside_chart_fraction=frac, **bounds)
    if sample is None:
        panels.extend([(None, 'No straightened crop'), (None, 'No sample to count')])
        record.update(chart_source=None, straightened_wh=None, grid_phase=None, counts=None, blocks=None)
    else:
        result = masked_count.count(sample)
        clean, grid, lines = _straight_photo(sample, result)
        blocks = dimensions(part, quad, photo.size, result['counts'])['blocks']
        guess = ' × '.join('?' if n is None else str(n) for n in blocks)
        axes = sum(bool(axis) for axis in lines)
        caption = ('No grid located', 'One grid axis located', 'Both grid axes located')[axes]
        panels.extend([(clean, f'{clean.width} × {clean.height} chart pixels'),
                       (grid, f'{caption}   ·   {guess} blocks')])
        record.update(chart_source=sample['chart_source'], straightened_wh=list(clean.size), counts=result['counts'],
                      grid_phase=result['phase'], grid_lines=lines, blocks=blocks,
                      straightened_valid_fraction=float(sample['mask'].mean()))
    return panels, record


def page_template(buildings):
    width = LEFT + len(buildings)*(COL+GAP)
    height = TOP + 3*(BAND+len(STAGES)*CELL) + 33
    page = Image.new('RGB', (width,height), PAPER)
    draw = ImageDraw.Draw(page)
    label(draw, (16,13), 'From surface outline to face-on crop', 39, bold=True)
    label(draw, (16,69), '24 selected surfaces   ·   8 buildings   ·   Compare cyan outlines, violet sampling windows and the straightened pixels below.', 23)
    label(draw, (16,107), 'Match percentages grade the finder outline. Dark pixels mark excluded samples. Material matching and placement were outside this review.', 19, fill=MUTED)
    for slot, letter in enumerate('ABC'):
        top = TOP + slot*(BAND + len(STAGES)*CELL)
        draw.rectangle((0,top,width,top+BAND-1),fill='#21484a')
        label(draw, (16, top+21), 'Example '+letter, 25, fill='white', bold=True)
        for col, layout in enumerate(buildings):
            label(draw, (LEFT+col*(COL+GAP),top+20), layout.title()+' '+letter, 28, fill='white', bold=True)
        for i, (name,note) in enumerate(STAGES):
            y = top+BAND+i*CELL
            draw.line((12,y,width-12,y),fill='#b1c2bf',width=2)
            label(draw, (14,y+21), name, 22, bold=True)
            for j,line in enumerate(wrap(note, 23)):
                label(draw, (14,y+64+j*25), line, 17, fill=MUTED)
    return page


def load_view(key, case, row, manifest):
    checked(SOURCE/row['predictions'], manifest['artifacts'][row['predictions']])
    photo, ref, native, grades = picture(SOURCE,row,case,['split'])
    truth = reference_planes(ref,read(case.path('labels'))['faces'])
    return photo, native, grades['split'], truth


@image_only()
def run(out):
    manifest = read(SOURCE/'manifest.json')
    for name in ('cases.json','sheet-views.json'):
        checked(SOURCE/name, manifest['artifacts'][name])
    checked(ARCHIVE,manifest['reference_sha256'])
    rows = {r['key']:r for r in read(SOURCE/'cases.json')}
    roster = read(SOURCE/'sheet-views.json')['views']
    groups_by_split, suite, fixtures = groups()
    if fixtures!=manifest['fixtures'] or [r['layout'] for r in roster]!=list(TARGETS):
        raise ValueError('Review layouts or frozen fixture changed')
    captures = {c.floor+'/'+c.id:c for group in groups_by_split.values() for c in group}
    page = page_template(TARGETS)
    draw = ImageDraw.Draw(page)
    examples = []
    for col, view in enumerate(roster):
        layout, overview = view['layout'], view['overview']
        if rows[overview['frame']]['sha256']!=overview['sha256']:
            raise ValueError('Overview screenshot changed')
        views = {}
        for slot, rid in enumerate(TARGETS[layout]):
            key = ALTERNATE_VIEWS.get((layout, slot), overview['frame'])
            row = rows[key]
            if key not in views:
                views[key] = load_view(key, captures[key], row, manifest)
            photo, native, g, truth = views[key]
            near = g['lookup'][rid]
            if not near['found']:
                raise ValueError(f'Expected a scored match in {key} R{rid}')
            part = g['parts'][near['index']-1]
            alternate = key!=overview['frame']
            building_bounds = Image.fromarray(native>0).getbbox()
            scene = scene_box(native,building_bounds,photo.size) if alternate else overview['scene_xyxy']
            cells, rec = panels(photo,part,rid,native,truth[rid]['size'],near['overlap'],scene)
            view_caption = key.split(layout+'-',1)[1].replace('-', ' ')
            cells[0] = cells[0][0], view_caption + ('  ·  alternate view' if alternate else '')
            code = layout.title()+' '+'ABC'[slot]
            x = LEFT+col*(COL+GAP)
            top = TOP+slot*(BAND+len(STAGES)*CELL)+BAND
            rec.update(id=code, layout=layout, frame=key, sha256=row['sha256'], source_wh=list(photo.size),
                       predictions=row['predictions'], prediction_sha256=manifest['artifacts'][row['predictions']],
                       alternate_view=alternate, stages={})
            for i, ((name,_),(img,caption)) in enumerate(zip(STAGES,cells,strict=True)):
                y = top+i*CELL
                if img is None:
                    draw.rectangle((x,y+9,x+COL-1,y+9+PIC-1),fill='#e1e6e2')
                    label(draw, (x+16,y+84), caption, 20, fill=MUTED)
                else:
                    _picture(page,img,x,y+9,COL,PIC)
                label(draw,(x,y+PIC+19),code+'   ·   '+caption,17,fill=INK)
                rec['stages'][name]=[x,y,x+COL,y+CELL]
            examples.append(rec)
    out.mkdir(parents=True, exist_ok=False)
    page.save(out/'comparison.jpg',quality=94,subsampling=0,optimize=True)
    save(out/'index.json',dict(examples=examples, image_wh=list(page.size), stages=[s[0] for s in STAGES],
         coordinates='Source boxes use half-open native screenshot pixels. Sheet stage boxes are in comparison.jpg.',
         selection='Three matched examples per building. Shopfront B and Workshop C use two additional saved views to show clipping. Recorded IDs select and grade illustrations only.'))
    save(out/'manifest.json',dict(run_id=out.name, code=revision(), version=suite['version'], fixtures=fixtures,
         source_run=SOURCE.relative_to(ROOT).as_posix(), source_manifest_sha256=digest(SOURCE/'manifest.json'),
         reference_sha256=manifest['reference_sha256'], frame_ids=sorted({e['frame'] for e in examples}),
         finder='plane-recovery-crop-splits', prediction_key='split', sample='engine.surface_sample.sample_surface',
         context_margin_px=surface_perimeter.DEFAULT_CONTEXT_PX, counter=masked_count.METHOD,
         stage_status=dict(material='unrun', placement='unrun'), new_inference_s=0,
         reproduce='python -m benchmark.crop_review_sheet --out runs/crop-review-repeat',
         artifacts={f:digest(out/f) for f in ('comparison.jpg','index.json')}))
    print(f'{len(examples)} crops on {page.width} × {page.height} sheet: {out}',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'crop-review-{uuid4().hex[:12]}')
    run(parser.parse_args().out.resolve())
