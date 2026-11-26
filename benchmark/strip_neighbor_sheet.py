"""Compare fixed review crops for the beam, wall extension, and real thin side face."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.attached_strip import refine as trim_by_photo
from benchmark.camera_round import ROOT, digest, read, save
from benchmark.camera_sheet import label
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_sheet import size_text
from benchmark.ledge_join_sheet import picture
from benchmark.normal_round import groups
from benchmark.strip_evidence import gate
from benchmark.strip_neighbor import refine
from benchmark.strip_review_sheet import CYAN, GOLD, RIGHT, WRONG, native_mask, overlay
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only

GREEN = (83, 234, 110)
STAGES = ['Source image', 'Recorded surface', 'Current Finder', 'Color rule',
          'Color rule cuts', 'Neighbor evidence', 'Neighbor rule', 'Neighbor rule cuts',
          'Current block grid', 'Color rule grid', 'Neighbor rule grid']


@image_only()
def render(out):
    review = ROOT / 'benchmark/reviews'
    source = review / 'sky-gap-01'
    old = review / 'attached-strip-01'
    beam, side = read(old/'strip-views.json')['views']
    wall = read(old/'strip-failure.json')
    rows = {r['key']:r for r in read(source/'cases.json')}
    captures = {c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    examples = [('Courtyard beam',beam), ('Courtyard wall',wall), ('Shopfront side',side)]
    records = []
    left, top, width, height = 230, 165, 425, 252
    page = Image.new('RGB',(left+3*width+12,top+len(STAGES)*height+10),'#f8f7f1')
    pen = ImageDraw.Draw(page)
    label(pen,(15,15),'Does another surface explain the strip?',36,bold=True)
    label(pen,(15,65),'Gold = recorded surface. Cyan = prediction. Green = neighboring mask used to support a cut.',21)
    label(pen,(15,98),'Pink = a wrong pixel removed. Amber = a real pixel lost. Removal counts cover the whole selected surface.',18)
    for i,title in enumerate(STAGES):
        y = top+i*height
        pen.line((12,y,page.width-12,y),fill='#b1c2bf',width=2)
        label(pen,(14,y+18),title,21,bold=True)
    for col,(title,ex) in enumerate(examples):
        key,rid = ex['frame'],ex['reference']
        row = rows[key]
        assert row['sha256']==ex['image_sha256']
        capture=captures[key]
        photo, reference, ids, graded = picture(source,row,capture,['gap'])
        parts = graded['gap']['parts']
        donor = next(p for p in parts if p['id']==ex['proposal'])
        color_parts,_ = trim_by_photo(photo,parts,gate=gate)
        decisions = []
        neighbor_parts,_ = refine(photo,parts,decisions=decisions)
        decisions = [d for d in decisions if d['donor']==donor['id']]
        receiver_ids = {d['receiver'] for d in decisions if d['receiver']}
        pieces = [donor] + [next(p for p in items if p['id']==donor['id']) for items in (color_parts,neighbor_parts)]
        current, color, neighbor = masks = [native_mask(p,photo.size) for p in pieces]
        truth = ids == rid
        box = ex.get('zoom_xyxy',ex['detail_xyxy'])
        locator = photo.copy()
        ImageDraw.Draw(locator).rectangle(box,outline='white',width=3)
        scene = Image.fromarray(ids>0).getbbox()
        actual = reference_planes(reference,read(capture.path('labels'))['faces'])[rid]['size']
        cells = [(locator.crop(scene),'White box locates the close-up'),
                 (overlay(photo,box,[(truth,GOLD,.65)]),f'Actual {size_text(actual)} blocks   ·   R{rid}')]
        methods = {}
        for name,mask in zip(('current','color','neighbor'),masks):
            removed = current & ~mask
            lost = removed & truth
            extra = removed & ~truth
            methods[name] = dict(correct=int((mask & truth).sum()),extra=int((mask & ~truth).sum()),
                                 correct_removed=int(lost.sum()),extra_removed=int(extra.sum()))
            if name=='neighbor':
                layers = [(current,CYAN,.40)]
                layers.extend((native_mask(p,photo.size),GREEN,.72) for p in parts if p['id'] in receiver_ids)
                layers.append((current & ~color,WRONG,.88))
                note = 'Pink = proposed cut; green = supporting neighbor' if receiver_ids else (
                       'No supporting neighbor accepted' if decisions else 'No color-supported cut proposed')
                cells.append((overlay(photo,box,layers),note))
            cells.append((overlay(photo,box,[(mask,CYAN,.66)]),
                          f'{methods[name]["correct"]:,} correct pixels   ·   {methods[name]["extra"]:,} extra'))
            if name!='current':
                cells.append((overlay(photo,box,[(extra,WRONG,1.),(lost,RIGHT,1.)]),
                              f'{int(extra.sum()):,} stray pixels removed   ·   {int(lost.sum()):,} real lost'))
        for name,part in zip(methods,pieces):
            sample = sample_surface(photo,part)
            counter = count(sample)
            grid = _straight_photo(sample,counter)[1] if sample else None
            guess = dimensions(part,sample['quad'] if sample else None,photo.size,counter['counts'])['blocks']
            methods[name]['blocks'] = guess
            cells.append((grid,f'Counter {size_text(guess)}   |   Actual {size_text(actual)}'))
        x = left+col*width
        label(pen,(x,top-31),title,23,bold=True)
        for i,(pic,note) in enumerate(cells):
            y = top+i*height
            if pic is not None:
                _picture(page,pic,x,y+5,width-8,height-43,nearest=i>0)
            label(pen,(x+2,y+height-31),note,15)
        records.append(dict(title=title,frame=key,image_sha256=ex['image_sha256'],reference=rid,
                            proposal=donor['id'],crop=box,actual=actual,methods=methods,decisions=decisions))
    out.mkdir(parents=True,exist_ok=True)
    page.save(out/'comparison.png',optimize=True)
    save(out/'sheet-views.json',dict(views=records,
         source_reports={p:digest(old/p) for p in ('strip-views.json','strip-failure.json')}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    render(parser.parse_args().out)
