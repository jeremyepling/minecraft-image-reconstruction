"""Zoomable comparison using eight fixed source views from the earlier crop review."""
from pathlib import Path


import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, read, save
from benchmark.camera_sheet import label
from benchmark.crop_fit_sheet import EXAMPLES, size_text
from benchmark.normal_round import groups
from benchmark.strip_review_sheet import CYAN, GOLD, native_mask, overlay
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface


def zoom_box(detail, changes, truth, side=60):
    """Pick a close-up without reference labels when either challenger moved an edge."""
    l,t,r,b = detail
    area = changes[t:b,l:r]
    if area.any():
        # Pick the most changed square; aligned candidate windows keep this deterministic.
        y,x = np.nonzero(area)
        bins = np.zeros(((b-t+side-1)//side,(r-l+side-1)//side), int)
        np.add.at(bins,(y//side,x//side),1)
        gy,gx = np.unravel_index(bins.argmax(),bins.shape)
        xmid,ymid = l+gx*side+side//2,t+gy*side+side//2
    else:
        y,x = np.nonzero(truth[t:b,l:r])
        xmid,ymid = (l+x[0], t+y[0]) if x.size else ((l+r)//2,(t+b)//2)
    h,w = truth.shape
    x0,y0 = int(np.clip(xmid-side//2,0,w-side)), int(np.clip(ymid-side//2,0,h-side))
    return (x0,y0,x0+side,y0+side)


def render(out):
    """Rows index archived prediction files; methods maps their keys to review labels."""
    out = Path(out)
    from benchmark.plane_detail import register, source_field

    rows = read(out/'cases.json')
    keys = list(read_gzip(out/rows[0]['predictions']))
    methods = {key: ('Current Finder' if key=='current' else 'Plane refinement' if key.startswith('plane_detail:') else 'Photo refinement') for key in keys}
    archive = read(ROOT/'benchmark/reviews/crop-review-02/index.json')['examples']
    examples = {ex['id']:ex for ex in archive}
    captures = {c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    rows = {r['key']:r for r in rows}
    keys = list(methods)
    stages = ['Source', 'Recorded surface', 'Raw plane close-up', '4-pixel sampling', '2-pixel sampling'] + [
        stage for key in keys for stage in (methods[key],'Edge close-up','Face-on crop','Block Counter')]
    colwidth, height, left, top = 312, 226, 212, 135
    page = Image.new('RGB',(left+colwidth*len(EXAMPLES),top+height*len(stages)+8),'#f8f7f1')
    pen = ImageDraw.Draw(page)
    label(pen,(14,10),'Which settings bring the outline closer?',35,bold=True)
    label(pen,(14,60),'Gold = recorded surface. Cyan = prediction. In close-ups, green = correctly included, pink = extra, orange = missing. White box locates the close-up.',19)
    for idx,name in enumerate(stages):
        y = top+idx*height
        pen.line((8,y,page.width-8,y),fill='#aab9b5',width=2)
        label(pen,(12,y+15),name,19,bold=True)
    review = []
    for col,example in enumerate(EXAMPLES):
        ex = examples[example]
        key,rid = ex['frame'],ex['reference']
        row,case = rows[key],captures[key]
        if ex['sha256'] != row['sha256']:
            raise ValueError('Review source changed')
        with Image.open(checked(case.path('image'),row['sha256'])) as source:
            photo = source.convert('RGB')
        pose,ref = recorded(case)
        ids = _native_reference(pose,ref,read(case.path('labels'))['faces'])
        truth = ids==rid
        predictions = read_gzip(out/row['predictions'])
        parts = [next((p for p in predictions[k] if p['id']==ex['proposal']), None) for k in keys]
        if parts[0] is None:
            raise ValueError('Review proposal missing from baseline '+example)
        masks = [native_mask(p,photo.size) for p in parts]
        different = np.logical_or.reduce([m!=masks[0] for m in masks[1:]])
        detail = ex['detail_xyxy']
        zoom = zoom_box(detail,different,truth)
        location = photo.copy()
        ImageDraw.Draw(location).rectangle(detail,outline='white',width=3)
        cells = [(location.crop(ex['scene_xyxy']),key.split('/')[-1]),
                 (overlay(photo,detail,[(truth,GOLD,.7)]),f"R{rid}   ·   Actual {size_text(ex['reference_size_blocks'])} blocks")]
        # Choose a single segment by overlap with the baseline prediction, before consulting gold.
        field = source_field(row['sha256'])
        raw = register(field,photo.size,1)
        ids_in_parent = np.bincount(raw[masks[0]],minlength=int(raw.max())+1)
        areas = np.bincount(raw.ravel())
        ious = ids_in_parent / np.maximum(1,areas + masks[0].sum() - ids_in_parent)
        ious[0] = 0
        segment = int(ious.argmax()) if ious.max()>0 else None
        for step in (1,4,2):
            registered = raw if step==1 else register(field,photo.size,step)
            mask = registered==segment if segment is not None else np.zeros_like(truth)
            cells.append((overlay(photo,zoom,[(mask&truth,(63,200,120),.48),(mask&~truth,(247,72,145),.95),
                          (truth&~mask,(255,177,42),.95)]), 'Full-view model diagnostic   ·   same segment'))
        scores = {}
        for method,part,mask in zip(keys,parts,masks):
            whole = overlay(photo,detail,[(mask,CYAN,.6)])
            x0,y0,x1,y1 = zoom
            ImageDraw.Draw(whole).rectangle((x0-detail[0],y0-detail[1],x1-detail[0],y1-detail[1]),outline='white',width=1)
            good,extra,missed = (int(v.sum()) for v in (mask&truth,mask&~truth,truth&~mask))
            cells.append((whole,f'{extra:,} extra pixels   ·   {missed:,} missed'))
            cells.append((overlay(photo,zoom,[(mask&truth,(63,200,120),.48),(mask&~truth,(247,72,145),.95),
                        (truth&~mask,(255,177,42),.95)]),'Same 60 × 60 source pixels'))
            sample = sample_surface(photo,part) if part else None
            result = count(sample)
            clean, grid = _straight_photo(sample,result)[:2] if sample else (None,None)
            guess = dimensions(part,sample['quad'] if sample else None,photo.size,result['counts'])['blocks'] if part else None
            cells.extend([(clean,'Mask after straightening' if sample else 'No usable crop'),
                          (grid,f"Counter {size_text(guess)}   |   Actual {size_text(ex['reference_size_blocks'])}")])
            scores[method] = dict(correct=good,extra=extra,missing=missed,blocks=guess)
        x = left+col*colwidth
        label(pen,(x,top-28),example,21,bold=True)
        for i,(img,caption) in enumerate(cells):
            y=top+i*height
            if img is not None:
                _picture(page,img,x+2,y+5,colwidth-6,height-40)
            label(pen,(x+2,y+height-28),caption,14)
        review.append(dict(example=example,frame=key,sha256=row['sha256'],reference=rid,
                           proposal=ex['proposal'],raw_plane_segment=segment,detail=detail,zoom=zoom,methods=scores))
    page.save(out/'comparison.jpg',quality=93,subsampling=0)
    save(out/'sheet-views.json',review)
