"""Replay the two B examples on the frozen crop comparison with their native pixel masks."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.camera_sheet import label
from benchmark.crop_fit_sheet import size_text
from benchmark.ledge_join_sheet import picture
from benchmark.normal_round import groups
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from engine import masked_count, surface_perimeter
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only

EXAMPLES = ('Courtyard B', 'Shopfront B')
TAIL = (652, 388, 654, 480)  # Fixed diagnostic box; never supplied to the refiners.
GOLD, CYAN = (255, 194, 73), (45, 233, 212)
WRONG, RIGHT = (252, 62, 152), (248, 164, 40)


def native_mask(part, size):
    layer = Image.new('L', size)
    if part:
        alpha, bounds = surface_perimeter._native_alpha(part['support'], size)
        layer.paste(alpha, bounds)
    return np.asarray(layer) > 0


def overlay(photo, box, layers):
    crop = np.array(photo.crop(box))
    x, y, ex, ey = box
    for mask, color, opacity in layers:
        cut = mask[y:ey, x:ex]
        crop[cut] = np.rint((1-opacity)*crop[cut] + opacity*np.array(color))
    return Image.fromarray(crop)


@image_only()
def render(out):
    from benchmark.attached_strip import refine
    from benchmark.strip_evidence import gate

    review = ROOT / 'benchmark/reviews'
    source = review / 'sky-gap-01'
    previous = review / 'crop-fit-01'
    base_rows = {r['key']:r for r in read(source/'cases.json')}
    source_views = {r['example']:r for r in read(previous/'sheet-views.json')}
    archive = {r['id']:r for r in read(review/'crop-review-02/index.json')['examples']}
    captures = {c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    stages = [('Source', 'White box locates this detail.'), ('Close-up', 'Original screenshot pixels.'),
              ('Recorded surface', 'Gold marks the actual face.'), ('Current Finder', 'Cyan marks its prediction.'),
              ('Shape cleanup', 'Remaining prediction in cyan.'), ('Shape removals', 'Pink = wrong pixels removed.'),
              ('Evidence cleanup', 'Remaining prediction in cyan.'), ('Evidence removals', 'Amber = correct pixels lost.'),
              ('Current crop', 'Face-on pixels and block grid.'), ('Shape crop', 'Same counter after cleanup.'),
              ('Evidence crop', 'Same counter after cleanup.')]
    left,top,width,height=257,160,465,256
    page = Image.new('RGB',(left+width*len(EXAMPLES)+12,top+height*len(stages)+12),'#f8f7f1')
    draw = ImageDraw.Draw(page)
    label(draw,(15,14),'The thin strip in example B',38,bold=True)
    label(draw,(15,71),'Same two screenshots as the crop sheet. Gold is recorded truth. Cyan is the predicted surface.',21)
    label(draw,(15,101),'Removal rows show only changed pixels. Pink means an error removed. Amber means a real surface pixel lost.',18)
    for i,(title,note) in enumerate(stages):
        y=top+i*height
        draw.line((12,y,page.width-12,y),fill='#b1c2bf',width=2)
        label(draw,(14,y+18),title,22,bold=True)
        label(draw,(14,y+53),note,15)
    records=[]
    for col,name in enumerate(EXAMPLES):
        v=source_views[name]
        key,rid=v['frame'],v['reference']
        photo,_,ids,graded=picture(source,base_rows[key],captures[key],['gap'])
        if base_rows[key]['sha256']!=v['sha256']:
            raise ValueError('Review screenshots changed')
        g=graded['gap']
        parts=g['parts']
        current=parts[g['lookup'][rid]['index']-1]
        masks={}
        pieces={'current':current}
        for label_name,checker in [('shape',None),('evidence',gate)]:
            changed,_=refine(photo,parts,gate=checker)
            pieces[label_name]=next((p for p in changed if p['id']==current['id']),None)
        for mode,part in pieces.items():
            masks[mode]=native_mask(part,photo.size)
        truth=ids==rid
        box=v['detail_xyxy']
        location=photo.copy()
        ImageDraw.Draw(location).rectangle(box,outline='white',width=2)
        actual=size_text(v['actual'])
        cells=[(location.crop(archive[name]['scene_xyxy']), key.split('/')[-1]),
               (photo.crop(box), 'Native close-up enlarged at the same scale below'),
               (overlay(photo,box,[(truth,GOLD,.67)]),f'Actual {actual} blocks   ·   R{rid}')]
        rec=dict(example=name,frame=key,image_sha256=v['sha256'],proposal=current['id'],
                 reference=rid,actual=v['actual'],detail_xyxy=box,methods={})
        valid=masks['current']
        for mode,m in masks.items():
            correct=int(np.count_nonzero(m & truth))
            wrong=int(np.count_nonzero(m & ~truth))
            removed=valid & ~m
            lost=int(np.count_nonzero(removed & truth))
            fixed=int(np.count_nonzero(removed & ~truth))
            cells.append((overlay(photo,box,[(m,CYAN,.65)]),f'{correct:,} correct pixels   ·   {wrong:,} extra'))
            if mode!='current':
                cells.append((overlay(photo,box,[(removed & ~truth,WRONG,1.),(removed & truth,RIGHT,1.)]),
                              f'{fixed:,} extra pixels removed   ·   {lost:,} correct lost'))
            stats=dict(correct=correct,extra=wrong,correct_removed=lost,extra_removed=fixed)
            if name=='Courtyard B':
                x,y,ex,ey=TAIL
                stats.update(lower_strip_remaining=int(m[y:ey,x:ex].sum()),
                             lower_strip_correct=int((truth[y:ey,x:ex]&m[y:ey,x:ex]).sum()))
                rec['lower_strip_xyxy']=TAIL
            rec['methods'][mode]=stats
        for mode,part in pieces.items():
            sample=sample_surface(photo,part) if part else None
            count=masked_count.count(sample)
            result=dimensions(part,sample['quad'] if sample else None,photo.size,count['counts'])['blocks'] if part else [None,None]
            rec['methods'][mode]['blocks']=result
            grid=_straight_photo(sample,count)[1] if sample is not None else None
            cells.append((grid,f'Counter {size_text(result)}   |   Actual {actual}'))
        x=left+col*width
        label(draw,(x,top-30),name,23,bold=True)
        for i,(image,caption) in enumerate(cells):
            y=top+i*height
            if image is not None:
                _picture(page,image,x,y+6,width-8,height-44,nearest=i>0)
            label(draw,(x+3,y+height-31),caption,16)
        records.append(rec)
    out.mkdir(parents=True,exist_ok=True)
    page.save(out/'strip-comparison.png',optimize=True)
    save(out/'strip-views.json',dict(source_run='crop-fit-01',source_sheet_sha256=digest(previous/'comparison.jpg'),
                                      views=records))
    first=records[0]
    feedback=dict(run_id='crop-fit-01',sheet_sha256=digest(previous/'comparison.jpg'),example='B',
                  stage='Surface Finder',note='a thin little strip that comes down that isn\'t part of the surface',
                  preference='Remove stray extensions while preserving genuine thin surfaces and suite scores.',
                  identification='Two examples were labeled B. Courtyard B matches the description; Shopfront B is retained as a narrow-face control.',
                  likely_image=first['frame'],likely_image_sha256=first['image_sha256'],
                  likely_proposal=first['proposal'],diagnostic_region_xyxy=TAIL)
    (out/'feedback.jsonl').write_text(json.dumps(feedback)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    render(parser.parse_args().out)
