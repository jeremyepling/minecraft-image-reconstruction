"""The round's review sheet. All columns use identical native crops and scoring thresholds."""
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from benchmark.camera_round import read,save
from benchmark.camera_labels import rasterize
from benchmark.normal_round import METHODS,BASE
from benchmark.surface_metrics import truth,DOWNSAMPLE
from benchmark.surface_topology import opening_mask
from benchmark.surface_sheet import scene_box,overlay_scene,text,panel,box_of,PAPER,DIM


TITLES=('Contrast + shading','Contrast scored by normals','Regions grouped by normals')


def case_data(out,row):
    src=(out/row['source']).resolve()
    root=src.parent.parent
    frame=next(f for f in read(root/'index.json')['frames'] if f['id']==row['id'])
    im=Image.open(src).convert('RGB')
    t=truth(read(root/frame['pose']),read(root/frame['labels'])['faces'])
    pred=json.loads(gzip.decompress((out/row['prediction']).read_bytes()))
    return im,t,pred


def delta(row,method):
    def f1(m):
        s=row['methods'][m]['outlines']
        return 2*s['matched']/max(1,s['proposed']+s['eligible'])
    return f1(method)-f1(BASE)


def render(out,rows,winner):
    scores=read(out/'scores.json')
    canvas=Image.new('RGB',(1580,1570),PAPER)
    pen=ImageDraw.Draw(canvas)
    text(pen,34,22,'Recovering visible planes from one screenshot',40,True)
    text(pen,34,83,'Minecraft Java 26.3   /   clear noon   /   48 suite frames and 8 new blind frames',22,fill=DIM)
    text(pen,34,148,'Correctly outlined planes by structure',28,True)
    text(pen,34,191,'One-to-one overlap ≥ 50%. Each cell shows matches / proposals. Eligible counts include missed planes.',20)
    xs=(34,320,498,775,1125)
    for x,label in zip(xs,('Layout','Eligible','Contrast + shading','Contrast + normals','Normal regions')):
        text(pen,x,245,label,22,True)
    for i,(layout,group) in enumerate(scores['layouts'].items()):
        is_new=any(r['layout']==layout and r['cohort']=='blind' for r in rows)
        y=290+34*i
        if i%2==0:
            pen.rectangle((31,y-3,1545,y+27),fill='#e9ede5')
        text(pen,xs[0],y,layout.title()+('  ·  new' if is_new else ''),20)
        for j,m in enumerate(METHODS):
            o=group['methods'][m]['outlines']
            if j==0:
                text(pen,xs[1],y,str(o['eligible']),20)
            text(pen,xs[2+j],y,f"{o['matched']} / {o['proposed']}     F1 {o['f1']:.2f}",19,bold=(m==winner))
    examples=[]
    prior=[r for r in rows if r['cohort']=='regression']
    examples.append(('Clearest gain on a previously tested structure',max(prior,key=lambda r:delta(r,winner))))
    blind=[r for r in rows if r['cohort']=='blind']
    # Among new views show the one with the most unmatched physical surfaces.
    difficult=max(blind,key=lambda r:(r['methods'][winner]['outlines']['eligible']-r['methods'][winner]['outlines']['matched'],
                                      r['methods'][winner]['gaps']['filled_pixels']))
    examples.append(('Most missed planes on a new structure',difficult))
    regions=[]
    text(pen,34,585,'Gold = recorded boundary     Cyan = matched outline     Pink = unmatched outline     Red fill = covered opening',19)
    for i,(description,row) in enumerate(examples):
        im,t,pred=case_data(out,row)
        crop=scene_box(t,im)
        y=649+443*i
        text(pen,34,y,description+'  /  '+row['id'].removeprefix('holdout-'),25,True)
        for j,m in enumerate(METHODS):
            x=34+j*507
            planes=pred[m]['model']['planes']
            outlines=row['methods'][m]['outlines']
            view=overlay_scene(im,t,planes,outlines)
            gap=opening_mask(t['id_map'])
            spill=np.zeros_like(gap)
            for p in planes:
                q=np.column_stack([np.array(p['quad'])/DOWNSAMPLE,np.ones(4)])
                hit=rasterize(q,gap.shape[1],gap.shape[0])
                if hit:
                    box,mask=hit[:2]
                    spill[box]|=mask&gap[box]
            view.paste('#fc463f',(0,0,*view.size),Image.fromarray(spill).resize(view.size,Image.Resampling.NEAREST))
            text(pen,x,y+45,TITLES[j],20,True)
            panel(canvas,view,crop,x,y+79,492,280)
            text(pen,x,y+377,f"{outlines['matched']} matched   /   {outlines['proposed']} proposed   /   {outlines['eligible']} eligible",18)
        regions.append(dict(image=row['id'],floor=row['floor'],region_xyxy=crop,stage='outlines'))
    canvas.save(out/'comparison.jpg',quality=94,subsampling=0)
    save(out/'review-regions.json',regions)
