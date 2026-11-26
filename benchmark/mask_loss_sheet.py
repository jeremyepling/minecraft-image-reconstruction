"""Draw selected, enlarged B failure examples from the numerical diagnostic."""
import gzip
import json

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import read
from benchmark.camera_sheet import font
from benchmark.moge_probe import normal_image
from benchmark.normal_fields import cached_fields
from benchmark.score_saved_planes import ROUND
from benchmark.surface_metrics import observed_support
from benchmark.surface_reference import recorded, review_case

PANEL = (240, 160)
COLUMNS = ('Source crop', 'MoGe direction', 'First B group', 'Mask sent to fitting',
           'Mask B kept', 'Rectangle B emitted', 'Saved reference')
EXAMPLES = (
    ('weak_raw_orientation', 'Too little direction evidence'),
    ('orientation_present_identity_ambiguous', 'Right direction, ambiguous surface'),
    ('lost_in_grouping', 'Surface lost during grouping'),
    ('rejected_by_fit', 'Fit rejected the region'),
    ('filtered_after_fit', 'Region fitted, then discarded'),
    ('distorted_by_quad', 'Rectangle distorted the mask'),
)


def colored(image, mask, color):
    if mask is None:
        return image
    layer = Image.fromarray((mask*110).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    return Image.composite(Image.new('RGB', image.size, color), image, layer)


def crop_bounds(box, size):
    x0,y0,x1,y1 = box
    # Same native pixels in every panel; show adjacent faces on each side.
    width = min(size[0], max(240, x1-x0+96, (y1-y0+72)*1.5))
    height = min(size[1], width/1.5)
    left = np.clip((x0+x1-width)/2,0,size[0]-width)
    top = np.clip((y0+y1-height)/2,0,size[1]-height)
    return tuple(round(v) for v in (left,top,left+width,top+height))


def render(rows, cache, path):
    from benchmark.mask_loss_trace import METHOD, observe_b

    archive = {row['key']:row for row in read(ROUND/'cases.json')}
    examples = []
    for tag, title in EXAMPLES:
        eligible = [(r,s) for r in rows for s in r['surfaces'] if s['loss']==tag]
        if eligible:
            r,s = max(eligible, key=lambda rs:rs[1]['reference_pixels'])
            examples.append((title,r,s))
    header, row_h = 113, 221
    canvas = Image.new('RGB',(30+len(COLUMNS)*(PANEL[0]+8),header+row_h*len(examples)+12),'#f4f2ed')
    draw=ImageDraw.Draw(canvas)
    draw.text((18,9),'Where the baseline lost a visible surface',font=font(27,True),fill='#1d3036')
    draw.text((18,48),'Blue = predicted pixels. Red = rectangle around the same kept mask. Gold = recorded surface, used only to score.',font=font(18),fill='#354b52')
    for i,title in enumerate(COLUMNS):
        draw.text((18+i*(PANEL[0]+8),82),title,font=font(17,True),fill='#354b52')
    for j,(title,trace,surface) in enumerate(examples):
        row=archive[trace['frame']]
        case=review_case(ROUND,row)
        _,ref=recorded(case)
        with Image.open(case.path('image')) as src:
            image=src.convert('RGB')
        fields=cached_fields(row['sha256'],cache)
        stages,_=observe_b(image,fields['normal'])
        with gzip.open(ROUND/row['prediction'],'rt',encoding='utf8') as fp:
            planes=json.load(fp)[METHOD]['model']['planes']
        box=crop_bounds(surface['bbox_xyxy'],image.size)
        target=ref['id_map']==surface['patch']
        def from_runs(key):
            index=surface['stages'][key]['candidate']
            if index is None:
                return None
            support=dict(size=stages['analysis_size'],runs=stages[key][index])
            return observed_support(support,target.shape,ref['image_size'])
        index=surface['emitted_support_index_for_same_quad']
        plane=planes[index] if index is not None else None
        kept=observed_support(plane['support'],target.shape,ref['image_size']) if plane else None
        rectangle=image.copy()
        if plane:
            pen=ImageDraw.Draw(rectangle)
            points=[tuple(p) for p in plane['quad']]
            pen.line(points+[points[0]],fill='#ff355e',width=5,joint='curve')
        panels=(image,normal_image(fields['normal']).resize(image.size,Image.Resampling.BILINEAR),
                colored(image,from_runs('initial_groups'),'#00c9ff'), colored(image,from_runs('fit_inputs'),'#00c9ff'),
                colored(image,kept,'#00c9ff'), rectangle, colored(image,target,'#ffb928'))
        top=header+j*row_h
        draw.text((18,top),title,font=font(18,True),fill='#1d3036')
        draw.text((390,top+1),trace['frame']+'  |  surface '+str(surface['patch']),font=font(16),fill='#475b61')
        for i,panel in enumerate(panels):
            canvas.paste(panel.crop(box).resize(PANEL,Image.Resampling.LANCZOS),(18+i*(PANEL[0]+8),top+27))
        st=surface['stages']
        captions=('', f"{surface['orientation_fraction']:.0%} agree on direction",
                  f"overlap {st['initial_groups']['iou']:.2f}", f"overlap {st['fit_inputs']['iou']:.2f}",
                  f"overlap {st['emitted_support']['iou']:.2f}", f"overlap {surface['same_quad_iou'] or 0:.2f}",
                  'Scoring only')
        for i,caption in enumerate(captions):
            draw.text((18+i*(PANEL[0]+8),top+190),caption,font=font(15),fill='#354b52')
    canvas.save(path,quality=91,subsampling=0)
