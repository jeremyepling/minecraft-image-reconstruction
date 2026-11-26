"""Review the same-wall patch and its far-edge check on the saved building views."""
import argparse
import io
import json
import textwrap
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.camera_sheet import label
from benchmark.count_transfer_review import indexed
from benchmark.crop_fit_sheet import size_text
from benchmark.detail_round import support
from benchmark.detail_transfer import ARCHIVE, member, shards
from benchmark.normal_round import groups
from benchmark.strip_review_sheet import overlay
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_metrics import truth
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from benchmark.surface_reference import recorded
from benchmark.texture_count_round import FINDERS
from benchmark.wall_join_test import checked, read_gzip
from engine.photo_edges import refine_surfaces
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from engine.texture_guided_count import count_surface
from runtime.mode import image_only


def select(rows):
    choices={}
    for row in rows:
        building=row['layout'].rstrip('123') if row['cohort']=='additional' else row['layout']
        for part in row['surfaces']:
            if 'actual' not in part:
                continue
            patch=part.get('patch') or {}
            priority=(bool(patch.get('accepted')),bool(patch.get('donor_box')),
                      part['guess']['current']!=part['actual'],
                      max(part['actual']), min(part['actual']))
            if building not in choices or priority>choices[building][0]:
                choices[building]=priority,row,part
    return {k:(r,p) for k,(_,r,p) in sorted(choices.items())}


def indexed_shards(dataset):
    data={}
    for _,path,_ in shards(dataset):
        with ZipFile(path) as z:
            for line in z.read('frames.jsonl').splitlines():
                row=json.loads(line)
                data[row['id']]=path,row
    return data


def new_frame(key,index,archives):
    path,row=index[key]
    name='ordinary/'+key+'.json.gz'
    parts=read_gzip(checked(ARCHIVE/name,archives['artifacts'][name]))['planes']
    with ZipFile(path) as z:
        photo=Image.open(io.BytesIO(member(z,row,'rgb.png'))).convert('RGB')
        pose=json.loads(member(z,row,'camera.json'))
        faces=json.loads(member(z,row,'reference-faces.json'))['faces']
    return photo,refine_surfaces(photo,parts)[0],_native_reference(pose,truth(pose,faces),faces),row['image_sha256']


def marked(clean,patch):
    marked=clean.copy()
    pen=ImageDraw.Draw(marked)
    for k,color in [('donor_box','#f066d3'),('far_box','#ffe05c')]:
        x,y,r,b=patch[k]
        pen.rectangle((x,y,r-1,b-1),outline=color,width=max(1,round(min(clean.size)*.035)))
    return marked


@image_only()
def render(rows,out,dataset):
    from benchmark.within_wall_grid import METHOD
    src,method=FINDERS['photo']
    manifest=read(src/'manifest.json')
    records=indexed(read(checked(src/'cases.json',manifest['artifacts']['cases.json'])),'key')
    captures={c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    released=indexed_shards(dataset)
    archive=read(ARCHIVE/'manifest.json')
    choices=select(rows)
    stages=['Screenshot','Finder outline','Nearby pixels','Straightened crop','Default count','Local patch','Far edge','Patch result','Recorded surface','Comparison']
    left,top,w,h=174,110,254,288
    page=Image.new('RGB',(left+w*len(choices),top+h*len(stages)),'#f7f6f1')
    pen=ImageDraw.Draw(page)
    label(pen,(16,12),'Can one textured part of a wall measure the whole surface?',34,bold=True)
    label(pen,(16,62),'Pink = measured patch. Yellow = far-end check. Teal = predicted surface. The gold reference row uses Minecraft geometry. ? = unknown.',20)
    for n,stage in enumerate(stages):
        y=top+n*h
        pen.line((6,y,page.width-6,y),fill='#b7c3bf')
        label(pen,(10,y+10),stage,17,bold=True)
    index={}
    for i,(name,(row,part)) in enumerate(choices.items()):
        key=row['key']
        if key in captures:
            cap=captures[key]
            with Image.open(checked(cap.path('image'),row['sha256'])) as f:
                photo=f.convert('RGB')
            path=records[key]['predictions']
            parts=read_gzip(checked(src/path,manifest['artifacts'][path]))[method]
            pose,ref=recorded(cap)
            native=_native_reference(pose,ref,read(cap.path('labels'))['faces'])
        else:
            photo,parts,native,image_sha=new_frame(key,released,archive)
            if image_sha!=row['sha256']:
                raise ValueError(key)
        p=next(p for p in parts if p['id']==part['proposal'])
        base,chart=count_surface(photo,p,sample_surface(photo,p))
        guessed=dimensions(p,chart['quad'] if chart else None,photo.size,base['counts'])['blocks']
        if guessed!=part['guess']['current']:
            raise ValueError('Production count mismatch in '+key)
        mask=support(p,photo.size)
        real=native==part['reference']
        l,t,r,b=Image.fromarray(mask|real).getbbox()
        box=max(0,l-10),max(0,t-10),min(photo.width,r+10),min(photo.height,b+10)
        loc=photo.copy()
        ImageDraw.Draw(loc).rectangle(box,outline='white',width=4)
        clean,grid,_=_straight_photo(chart,base) if chart else (None,None,None)
        patch=part.get('patch') or {}
        accepted=patch.get('accepted',False)
        candidate=part['guess']['patch']
        evidence=bool(patch.get('donor_box'))
        marked_chart=marked(clean,patch) if evidence else None
        far=clean.crop(tuple(patch['far_box'])) if evidence else None
        trial=grid
        if accepted:
            phase=[None,None]
            phase[patch['axis']]=patch['phase']
            trial=_straight_photo(chart,dict(counts=patch['raw_counts'],phase=phase))[1]
        actual=part['actual']
        rows_here=[(loc,'White box locates the surface'),
                   (overlay(photo,box,[(mask,(22,218,206),.62)]),'Predicted support'),
                   (photo.crop(box),'Source pixels with context'),(clean,'Visible pixels after straightening'),
                   (grid,'Default '+size_text(guessed)),
                   (marked_chart,('Repeat every %.1f pixels'%patch['pitch_px']) if evidence else 'No local pitch proposed'),
                   (far,'Reserved far-end pixels' if evidence else 'No far-end check'),
                   (trial,('Accepted ' if accepted else 'Kept default ')+size_text(candidate)),
                   (overlay(photo,box,[(real,(250,180,55),.65)]),'Recorded '+size_text(actual))]
        x=left+i*w
        label(pen,(x+2,top-30),name.title(),19,bold=True)
        for j,(img,caption) in enumerate(rows_here):
            y=top+j*h
            if img:
                _picture(page,img,x+2,y+5,w-4,h-42,nearest=j>0)
            label(pen,(x+2,y+h-29),caption,15)
        y=top+(len(stages)-1)*h
        for j,(title,guess) in enumerate([('Default',guessed),('Patch',candidate)]):
            verdict='correct' if guess==actual else 'unresolved' if None in guess else 'wrong'
            label(pen,(x+2,y+12+36*j),title+' '+verdict,18,bold=True)
        for j,line in enumerate(textwrap.wrap(patch.get('reason','No qualifying patch'),28)):
            label(pen,(x+2,y+100+25*j),line,15)
        index[name]=dict(key=key,proposal=p['id'],reference=part['reference'],guess=part['guess'],actual=actual,
                         patch=patch,sha256=row['sha256'])
    out.parent.mkdir(parents=True,exist_ok=True)
    page.save(out,quality=93,subsampling=0)
    save(out.with_suffix('.json'),dict(method=METHOD,examples=index))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cases',type=Path,nargs='+',required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--dataset',type=Path,default=ROOT/'runs/visible-builds-v1')
    a=cli.parse_args()
    render([row for path in a.cases for row in read(path)],a.out,a.dataset)
