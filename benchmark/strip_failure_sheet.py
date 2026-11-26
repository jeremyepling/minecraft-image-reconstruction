"""Show where color-supported cleanup crosses a genuine reference surface."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.attached_strip import refine
from benchmark.camera_round import ROOT, read, save
from benchmark.camera_sheet import label
from benchmark.ledge_join_sheet import picture
from benchmark.normal_round import groups
from benchmark.strip_evidence import gate
from benchmark.strip_review_sheet import GOLD, CYAN, WRONG, RIGHT, native_mask, overlay
from benchmark.surface_pipeline_sheet import _picture
from runtime.mode import image_only


@image_only()
def render(out):
    key='light_gray_concrete/holdout-courtyard-plaster-low-left'
    rid=5
    source=ROOT/'benchmark/reviews/sky-gap-01'
    row=next(r for r in read(source/'cases.json') if r['key']==key)
    capture=next(c for group in groups()[0].values() for c in group if c.floor+'/'+c.id==key)
    photo,_,ids,graded=picture(source,row,capture,['gap'])
    group=graded['gap']
    parts=group['parts']
    part=parts[group['lookup'][rid]['index']-1]
    result,_=refine(photo,parts,gate=gate)
    after_part=next(p for p in result if p['id']==part['id'])
    before=native_mask(part,photo.size)
    after=native_mask(after_part,photo.size)
    truth=ids==rid
    removed=before & ~after
    lost=removed & truth
    fixed=removed & ~truth
    def bounds(mask,pad):
        x,y,ex,ey=Image.fromarray(mask).getbbox()
        return [max(0,x-pad),max(0,y-pad),min(photo.width,ex+pad),min(photo.height,ey+pad)]
    detail=bounds(before | truth,12)
    zoom=bounds(lost,11)
    overview=bounds(ids>0,28)
    locator=photo.copy()
    ImageDraw.Draw(locator).rectangle(zoom,outline='white',width=3)
    def make_cells(box):
        return [photo.crop(box),overlay(photo,box,[(truth,GOLD,.64)]),
                overlay(photo,box,[(before,CYAN,.68)]),overlay(photo,box,[(after,CYAN,.68)]),
                overlay(photo,box,[(lost,RIGHT,1.),(fixed,WRONG,1.)])]
    page=Image.new('RGB',(1050,1776),'#f8f7f1')
    pen=ImageDraw.Draw(page)
    label(pen,(18,17),'Where the strip cleanup goes too far',36,bold=True)
    label(pen,(18,74),'Courtyard, plaster walls, low-left view on concrete. The white box locates the close-up.',20)
    _picture(page,locator.crop(overview),20,111,1010,247)
    label(pen,(18,376),'Whole surface',24,bold=True)
    label(pen,(541,376),'Removed area enlarged',24,bold=True)
    titles=['Original screenshot','Recorded surface in gold','Current Finder in cyan',
            'Evidence cleanup in cyan','Removed pixels: amber = real surface lost, pink = extra removed']
    for i,(a,b,title) in enumerate(zip(make_cells(detail),make_cells(zoom),titles)):
        y=421+i*260
        label(pen,(19,y),title,20,bold=True)
        for x,crop in [(20,a),(535,b)]:
            _picture(page,crop,x,y+34,496,212,nearest=True)
    count_lost=int(lost.sum())
    count_fixed=int(fixed.sum())
    label(pen,(18,1725),f'R{rid}  ·  {count_lost:,} correctly assigned pixels removed  ·  {count_fixed:,} extra pixels removed',22,bold=True)
    out.mkdir(parents=True,exist_ok=True)
    page.save(out/'strip-failure.png',optimize=True)
    save(out/'strip-failure.json',dict(frame=key,image_sha256=row['sha256'],reference=rid,
        proposal=part['id'],detail_xyxy=detail,zoom_xyxy=zoom,correct_removed=count_lost,
        extra_removed=count_fixed,correct_before=int((before & truth).sum()),correct_after=int((after & truth).sum())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    render(parser.parse_args().out)
