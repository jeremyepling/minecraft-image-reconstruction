"""Diagnostic: read the same texture with the predicted crop and the true visible face corners."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, read, save
from benchmark.camera_sheet import label
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_sheet import size_text
from benchmark.normal_round import groups
from benchmark.strip_review_sheet import GOLD, overlay
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from benchmark.surface_reference import recorded
from benchmark.texture_count_round import SOURCE
from benchmark.wall_join_test import checked, read_gzip
from engine.atlas_count import load_atlas
from engine.geometry import UNIT, project
from engine.masked_count import count
from engine.plane_consensus import row_runs
from engine.surface_dimensions import dimensions
from engine.surface_model import canonical
from engine.surface_sample import sample_surface


PORTICO='smooth_stone/holdout-portico-plaster-low-left'
EXAMPLES=[('Portico beam',PORTICO,20),('Portico wall',PORTICO,13),('Portico pillar',PORTICO,8),
          ('Courtyard wall','smooth_stone/holdout-courtyard-plaster-high-right',10)]


def measured(sample,part,photo,result):
    return dimensions(part,sample['quad'] if sample else None,photo.size,result['counts'])['blocks']


def render(out):
    from benchmark.portico_repeat import predict as local_count
    from benchmark.portico_atlas import predict as atlas_count

    rows={r['key']:r for r in read(SOURCE/'cases.json')}
    captures={c.floor+'/'+c.id:c for g in groups()[0].values() for c in g}
    atlas=load_atlas()
    width,height,left,top=392,210,230,141
    stages=['Source','Recorded surface','Vanilla tile','Finder crop','Shipped counter','Crop-only trial',
            'Atlas-assisted trial','Recorded crop','Recorded grid','Same counter on\nrecorded crop']
    page=Image.new('RGB',(left+width*len(EXAMPLES),top+height*len(stages)+8),'#f8f7f1')
    pen=ImageDraw.Draw(page)
    label(pen,(16,12),'Can the same texture tell us how many blocks we see?',31,bold=True)
    label(pen,(16,59),'Gold surface and the last three rows use recorded geometry for diagnosis. The three counters above see the same predicted crop.',18)
    for i,title in enumerate(stages):
        y=top+i*height
        pen.line((6,y,page.width-6,y),fill='#aab9b5',width=2)
        for j,line in enumerate(title.splitlines()):
            label(pen,(12,y+14+j*27),line,19,bold=True)
    examples=[]
    for col,(name,key,rid) in enumerate(EXAMPLES):
        row,cap=rows[key],captures[key]
        with Image.open(checked(cap.path('image'),row['sha256'])) as opened:
            photo=opened.convert('RGB')
        pose,ref=recorded(cap)
        faces=read(cap.path('labels'))['faces']
        native=_native_reference(pose,ref,faces)
        target=reference_planes(ref,faces)[rid]
        part_id=next(p['proposal'] for p in row['paired_surfaces'] if p['reference']==rid)
        part=next(p for p in read_gzip(SOURCE/row['predictions'])['current'] if p['id']==part_id)
        truth=native==rid
        l,t,r,b=Image.fromarray(truth).getbbox()
        bx=(max(0,l-8),max(0,t-8),min(photo.width,r+8),min(photo.height,b+8))
        locator=photo.copy()
        ImageDraw.Draw(locator).rectangle(bx,outline='white',width=3)
        ids={f['block_id'] for f in target['faces'].values()}
        if len(ids)!=1:
            raise ValueError('Expected a single-material review control')
        material=ids.pop().split(':')[1]
        actual=target['size']
        coords=np.asarray(list(target['faces']))
        lo=coords.min(0); hi=coords.max(0)+1
        quad=canonical(project(ref['charts'][rid],lo+UNIT*(hi-lo)))
        recorded_part=dict(quad=quad,support=dict(size=list(photo.size),runs=row_runs(truth)))
        correct_sample=sample_surface(photo,recorded_part)
        sample=sample_surface(photo,part)
        cells=[(locator.crop(Image.fromarray(native>0).getbbox()),'Box marks selected surface'),
               (overlay(photo,bx,[(truth,GOLD,.68)]),f'Actual {size_text(actual)} blocks   ·   R{rid}'),
               (Image.fromarray(atlas.tiles[atlas.labels.index(material)].astype(np.uint8)),material.replace('_',' ')+'   ·   16 × 16 texels')]
        base=count(sample)
        clean,_grid,_lines=_straight_photo(sample,base)
        cells.append((clean,f'{clean.width} × {clean.height} crop pixels   ·   {sample["mask"].mean():.0%} usable'))
        guesses={}
        for method,res in [('current',base),('repeat',local_count(sample,base)),('atlas',atlas_count(sample,base))]:
            guess=measured(sample,part,photo,res)
            cells.append((_straight_photo(sample,res)[1],'Counter '+size_text(guess)+'   |   Actual '+size_text(actual)))
            guesses[method]=guess
        diag=count(correct_sample)
        known=measured(correct_sample,recorded_part,photo,diag)
        clean,grid,_=_straight_photo(correct_sample,diag)
        cells.extend([(clean,'True visible mask and four true corners'),
                      (_straight_photo(correct_sample,dict(counts=actual,phase=[0,0]))[1],'Recorded one-block spacing'),
                      (grid,'Counter '+size_text(known)+'   |   Actual '+size_text(actual))])
        examples.append(dict(frame=key,sha256=row['sha256'],reference=rid,proposal=part_id,material=material,
                             actual=actual,estimates=guesses,known_crop_estimate=known,known_crop_evidence=diag,
                             predicted_crop_wh=list(sample['mask'].shape[::-1]),valid_fraction=float(sample['mask'].mean())))
        x=left+col*width
        label(pen,(x,top-30),name,22,bold=True)
        for i,(img,note) in enumerate(cells):
            y=top+i*height
            _picture(page,img,x+2,y+5,width-6,height-42,nearest=i>0)
            label(pen,(x+2,y+height-30),note,16)
    page.save(out/'portico.jpg',quality=93,subsampling=0)
    save(out/'portico.json',examples)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    render(parser.parse_args().out)
