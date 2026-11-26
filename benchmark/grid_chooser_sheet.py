"""Two scored visible surfaces per building on the fixed grid-chooser comparison."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import read, save
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.normal_round import groups
from benchmark.surface_pipeline_sheet import _picture, _straight_photo, PAPER, MUTED
from benchmark.surface_reference import recorded
from benchmark.surface_sheet import text
from benchmark.wall_join_test import checked, read_gzip
from engine.strip_neighbor import clean_surfaces
from engine.surface_perimeter import context_crop, draw_outline, with_outline
from engine.surface_sample import sample_surface


def size_text(size):
    return ' × '.join('?' if n is None else str(n) for n in size)


def examples(inputs, targets, choices):
    by_layout = defaultdict(list)
    for frame, labels, selected in zip(inputs, targets, choices, strict=True):
        if any(r['sha256'] != frame['sha256'] for r in (labels, selected)):
            raise ValueError('Changed frame')
        for p,t,s in zip(frame['surfaces'],labels['surfaces'],selected['surfaces'],strict=True):
            if not p['id']==t['id']==s['id']:
                raise ValueError('Changed proposal')
            if t['expected'] is None or p['quad'] is None:
                continue
            hits = [c for c in p['candidates'] if c['blocks']==t['expected']]
            improved = bool(hits) and p['baseline']['blocks']!=t['expected']
            if s['index'] is None:
                chosen = p['baseline']
            elif s['index']==-1:
                chosen = dict(counts=[None,None],blocks=[None,None],phase=[None,None])
            else:
                chosen = p['candidates'][s['index']]
            by_layout[frame['layout']].append((frame['key'],p,t,hits,chosen,improved))
    for layout,cases in by_layout.items():
        first = max(cases,key=lambda c:(c[5],bool(c[3]),c[1]['baseline']['blocks']!=c[4]['blocks']))
        empty_pools = [c for c in cases if c[0]!=first[0] and not c[1]['candidates']]
        second = min(empty_pools,key=lambda c:c[0].endswith('close-front'))
        yield layout,(first,second)


def reference_photo(image, ref, faces, rid):
    mask = Image.fromarray(np.uint8(ref['id_map']==rid)*255).resize(image.size,Image.Resampling.NEAREST)
    marked = image.copy()
    pen = ImageDraw.Draw(marked)
    for f in faces[rid]['faces'].values():
        q = [tuple(x) for x in f['image_corners']]
        pen.line(q+[q[0]],fill='#51ffd4',width=2)
    return Image.composite(marked,image,mask)


def render(run, selections, out):
    source = read(checked(BASE/'manifest.json',BASE_SHA))
    rows = {r['key']:r for r in read(checked(BASE/'cases.json',source['artifacts']['cases.json']))}
    cohorts,_,_ = groups()
    captures = {c.floor+'/'+c.id:c for cc in cohorts.values() for c in cc}
    columns = sorted(examples(read(run/'candidates.json'),read(run/'targets.json'),read(selections)))
    col, left, head, row_h, sub_h = 265, 176, 111, 395, 193
    names = ('Screenshot', 'Detected surface', 'Default counter', 'Forced chooser', 'Perfect chooser', 'Visible answer')
    page = Image.new('RGB',(left+len(columns)*col+10,head+len(names)*row_h),PAPER)
    pen = ImageDraw.Draw(page)
    text(pen,18,10,'Can the Block Counter choose a better grid?',32,True)
    text(pen,18,52,'Each building has two views per row. Gold lines show located seams. Green lines show recorded visible block edges.',18)
    for i,name in enumerate(names):
        y = head+i*row_h
        pen.line((12,y,page.width-12,y),fill='#c7d3cf',width=2)
        text(pen,14,y+11,name,18,True)
    ids = []
    for j,(layout,views) in enumerate(columns):
        x = left+j*col
        text(pen,x,82,layout.title(),22,True)
        for i,(key,p,t,hits,chosen,_) in enumerate(views):
            row = rows[key]
            cap = captures[key]
            with Image.open(checked(cap.path('image'),row['sha256'])) as src:
                image = src.convert('RGB')
            upstream = read_gzip(checked(BASE/row['predictions'],source['artifacts'][row['predictions']]))['gap']
            parts,_ = clean_surfaces(image,upstream)
            part = next(rec for rec in parts if rec['id']==p['id'])
            marked = image.copy()
            draw_outline(marked,with_outline(part),width=3)
            _,box,_ = context_crop(image,part,margin_px=16)
            sample = sample_surface(image,part)
            _,ref = recorded(cap)
            planes = reference_planes(ref,read(cap.path('labels'))['faces'])
            answers = reference_photo(image,ref,planes,t['reference_id'])
            panels = [(image,' '.join(key.split('-')[-3:])),
                      (marked.crop(box),p['id'])]
            for r in [p['baseline'],chosen,hits[0] if hits else None]:
                if r is None:
                    blank = dict(counts=[None,None],phase=[None,None])
                    panels.append((_straight_photo(sample,blank)[0], 'No exact candidate'))
                else:
                    panels.append((_straight_photo(sample,r)[1],size_text(r['blocks'])+'   ·   '+
                                   str(sum(v is not None for v in r['phase']))+'/2 axes located'))
            panels.append((answers.crop(box),size_text(t['expected'])+' visible extent'))
            for stage,(photo,caption) in enumerate(panels):
                y = head+stage*row_h+i*sub_h+8
                _picture(page,photo,x,y,col-10,151)
                text(pen,x,y+154,caption,15,fill=MUTED)
            ids.append(dict(layout=layout,frame=key,proposal=p['id'],reference_id=t['reference_id']))
    out.mkdir(parents=True,exist_ok=True)
    page.save(out/'comparison.jpg',quality=90)
    save(out/'sheet-views.json',ids)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('forced_selections',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    render(args.run,args.forced_selections,args.out)
