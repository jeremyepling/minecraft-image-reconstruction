"""Show disputed pixels from saved surface predictions at their original coordinates."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, read, save, digest
from benchmark.camera_sheet import label
from benchmark.normal_round import groups
from benchmark.sky_round import occupied, reference_masks
from benchmark.wall_join_test import checked, read_gzip


def window(x, y, w, h, width, height):
    x0 = int(np.clip(x - w//2, 0, width-w))
    y0 = int(np.clip(y - h//2, 0, height-h))
    return x0, y0, x0+w, y0+h


def run(out):
    root = ROOT/'benchmark/reviews/sky-01'
    hashes = read(root/'manifest.json')['artifacts']
    rows = {r['key']:r for r in read(checked(root/'cases.json',hashes['cases.json']))}
    captures = {c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    cards = [('Crossing','masonry'), ('Galleries','masonry'), ('Crossing','plaster'),
             ('Galleries','plaster'), ('Wing','masonry'), ('Portico','masonry')]
    page = Image.new('RGB',(1472,2390),'#f7f5ef')
    draw = ImageDraw.Draw(page)
    ink = '#162d36'
    def title(x,y,msg,size,bold=False):
        label(draw,(x,y),msg,size,fill=ink,bold=bold)
    title(28,22,'Where sky meets the build',44,True)
    title(29,85,'First four crops: orange pixels stayed in a surface where the recorded camera sees open sky.',25)
    title(29,123,'Last two: pink pixels were removed where the recorded outline includes a building edge.',25)
    title(29,160,'Each card shows the original view, the untouched detail, then the same detail with disputed pixels marked.',22)
    records = []
    for i,(layout,palette) in enumerate(cards):
        case_kind = 'dev' if layout=='Wing' else 'holdout'
        key=f'light_gray_concrete/{case_kind}-{layout.lower()}-{palette}-low-left'
        row=rows[key]
        case=captures[key]
        with Image.open(checked(case.path('image'),row['sha256'])) as src:
            photo=src.convert('RGB')
        parts=read_gzip(checked(root/row['predictions'],hashes[row['predictions']]))
        old,new = (occupied(parts[m],photo.size) for m in ('current','sky'))
        building,sky,_=reference_masks(case)
        missed=i<4
        disputed = (new & sky) if missed else ((old & ~new) & building)
        ys,xs=np.nonzero(disputed)
        assert len(xs)==row['native_pixels']['upward_background_after' if missed else 'removed_recorded_building']
        # Select the densest close-up around any disputed pixel, retaining source alignment.
        width,height = (112,58) if missed else (36,19)
        integral=np.pad(disputed,((1,0),(1,0))).cumsum(0).cumsum(1)
        boxes=[window(x,y,width,height,*photo.size) for x,y in zip(xs,ys)]
        def area(b):
            l,t,r,d=b
            return integral[d,r]-integral[t,r]-integral[d,l]+integral[t,l]
        box=max(boxes,key=area)
        l,t,r,d=box
        x=28+(i%2)*722
        y=224+(i//2)*717
        letter=chr(65+i)
        title(x,y,f'{letter}   {layout}  /  {palette}',30,True)
        thumb=photo.resize((424,265),Image.Resampling.LANCZOS)
        locator=ImageDraw.Draw(thumb)
        locator.rectangle(tuple(v*(thumb.width/photo.width if j%2==0 else thumb.height/photo.height) for j,v in enumerate(box)),outline='white',width=3)
        page.paste(thumb,(x,y+47))
        title(x+435,y+140,'Low-left view',21)
        title(x+435,y+175,'White box = detail',19)
        crop=photo.crop(box)
        marked=np.array(crop)
        marked[disputed[t:d,l:r]]=(255,156,0) if missed else (255,42,136)
        for j,im in enumerate((crop,Image.fromarray(marked))):
            page.paste(im.resize((336,round(336*height/width)),Image.Resampling.NEAREST),(x+345*j,y+370))
        title(x,y+333,'Original pixels',23,True)
        title(x+345,y+333,'Filter decision',23,True)
        title(x,y+580,f'{int(area(box)):,} disputed pixels in this crop  /  {len(xs):,} in the view',23)
        title(x,y+625,'Kept as surface' if missed else 'Removed as sky  ·  one-pixel boundary disagreement',22,True)
        records.append(dict(id=letter,frame=key,sha256=row['sha256'],box=box,marked=int(area(box)),in_view=len(xs)))
    out.mkdir(parents=True,exist_ok=False)
    page.save(out/'comparison.png',optimize=True)
    save(out/'index.json',dict(examples=records,source_manifest=digest(root/'manifest.json'),image=digest(out/'comparison.png')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
