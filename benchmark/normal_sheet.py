"""The round's review sheet. All columns use identical native crops and scoring thresholds."""
import gzip
import json

import numpy as np
from PIL import Image,ImageDraw

from benchmark.camera_round import read,save
from benchmark.camera_labels import rasterize
from benchmark.normal_round import METHODS,BASE
from benchmark.surface_metrics import truth,DOWNSAMPLE,observed_support
from benchmark.surface_topology import opening_mask
from benchmark.surface_sheet import scene_box,overlay_scene,text,panel,poly,box_of,PAPER,DIM


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


def quad_coverage(planes,shape):
    covered=np.zeros(shape,dtype=bool)
    for plane in planes:
        q=np.column_stack([np.asarray(plane['quad'])/DOWNSAMPLE,np.ones(4)])
        hit=rasterize(q,shape[1],shape[0])
        if hit:
            box,mask=hit[:2]
            covered[box]|=mask
    return covered


def support_review(canvas,out,row,y):
    im,t,pred=case_data(out,row)
    method='normal-regions'
    planes=pred[method]['model']['planes']
    opening=opening_mask(t['id_map'])
    quads=quad_coverage(planes,opening.shape)
    support=np.zeros_like(opening)
    for plane in planes:
        support|=observed_support(plane['support'],opening.shape,im.size)
    masks=[quads,support]
    stats=row['methods'][method]
    opening_size=int(opening.sum())
    fills=[int((m&opening).sum()) for m in masks]
    for filled,key in zip(fills,['gaps','support_gaps']):
        if (filled,opening_size)!=(stats[key]['filled_pixels'],stats[key]['opening_pixels']):
            raise ValueError(f'Opening overlay differs from the saved {key} score for {row["id"]}')
    ys,xs=np.where(opening)
    crop=scene_box(t,im)
    if len(xs):
        ends=np.array([[xs.min(),ys.min()],[xs.max()+1,ys.max()+1]])*DOWNSAMPLE
        crop=box_of(ends,im.size,32)

    def native(mask):
        return Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST)

    # Mark the opening perimeter on the two diagnostic images. The raw crop
    # has no marks. Red takes precedence even on a single boundary sample.
    padded=np.pad(opening,1)
    interior=(padded[1:-1,:-2]&padded[1:-1,2:]&padded[:-2,1:-1]&padded[2:,1:-1])
    rim=native(opening&~interior)
    cyan=Image.blend(im,Image.new('RGB',im.size,'#18d9d0'),.49)
    views=[im]
    for mask in masks:
        view=Image.composite(cyan,im,native(mask))
        view.paste('#ffd055',(0,0,*im.size),rim)
        view.paste('#fc463f',(0,0,*im.size),native(mask&opening))
        views.append(view)
    pen=ImageDraw.Draw(canvas)
    text(pen,34,y,'How much of the opening does each representation fill?',27,True)
    x0,y0,x1,y1=crop
    text(pen,34,y+44,f"{row['id']}   /   Normal regions   /   Native crop ({x0}, {y0}) to ({x1}, {y1})",19)
    text(pen,34,y+81,'Teal = predicted coverage     Gold edge = recorded opening     Red = coverage inside a recorded opening',18)
    labels=['Source screenshot','Fitted quadrilaterals','Predicted support mask (RLE)']
    captions=[f'{opening_size:,} opening pixels on a {opening.shape[1]} × {opening.shape[0]} scoring grid']+[
        f'{n:,} / {opening_size:,} opening pixels filled' for n in fills]
    for j,(view,label,caption) in enumerate(zip(views,labels,captions)):
        x=34+j*507
        text(pen,x,y+131,label,22,True)
        panel(canvas,view,crop,x,y+170,492,325)
        text(pen,x,y+511,caption,18)
    return dict(image=row['id'],floor=row['floor'],region_xyxy=crop,stage='outlines',
                name='Fitted outlines and saved region support over openings',method=method)


def render(out,rows,winner,*,destination=None):
    destination=out if destination is None else destination
    scores=read(out/'scores.json')
    canvas=Image.new('RGB',(1580,2580),PAPER)
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
    text(pen,34,585,'Gold = recorded boundary     Cyan = matched outline     Pink = unmatched outline     Red fill = opening inside a proposed quad',18)
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
            spill=quad_coverage(planes,gap.shape)&gap
            view.paste('#fc463f',(0,0,*view.size),Image.fromarray(spill).resize(view.size,Image.Resampling.NEAREST))
            text(pen,x,y+45,TITLES[j],20,True)
            panel(canvas,view,crop,x,y+79,492,280)
            text(pen,x,y+377,f"{outlines['matched']} matched   /   {outlines['proposed']} proposed   /   {outlines['eligible']} eligible",18)
        regions.append(dict(image=row['id'],floor=row['floor'],region_xyxy=crop,stage='outlines'))
    regions.append(support_review(canvas,out,difficult,1558))
    text(pen,34,2128,'Which surface is in front?',27,True)
    # Choose a measured automatic relationship; the two methods share the same predicted quads.
    choices=sorted(rows,key=lambda r:(r['cohort']=='blind',r['methods'][winner]['order']['depth-anything']['correct']),reverse=True)
    for row in choices:
        if row['methods'][winner]['order']['depth-anything']['correct']==0:
            continue
        im,t,pred=case_data(out,row)
        data=pred[winner]
        planes=data['model']['planes']
        by_id={p['id']:p for p in planes}
        matches={planes[m['proposal_index']]['id']:m['patch_id'] for m in row['methods'][winner]['outlines']['matches']}
        truth_pairs={(p['near_patch'],p['far_patch']) for p in row['depth_diagnostic']['pairs']}
        pair=next((p for p in data['relations']['depth-anything'] if
            (matches.get(p['near']),matches.get(p['far'])) in truth_pairs and
            (matches.get(p['far']),matches.get(p['near'])) not in truth_pairs),None)
        if not pair:
            continue
        near,far=pair['near'],pair['far']
        crop=box_of([by_id[near]['quad'],by_id[far]['quad']],im.size,20)
        text(pen,34,2170,row['id']+'   /   Cyan = nearer, amber = farther',19)
        old=next((p for p in data['relations']['geometry'] if {p['near'],p['far']}=={near,far}),None)
        for j,(title,relation) in enumerate((('Recorded order on matched planes',pair),('Geometry ordering',old),('Relative-depth ordering',pair))):
            x=34+j*507
            text(pen,x,2225,title,21,True)
            view=im.copy()
            d=ImageDraw.Draw(view)
            for a in (near,far):
                color=('#35fff0' if relation['near']==a else '#ffd055') if relation else '#eeeeee'
                poly(d,by_id[a]['quad'],color,4,True)
            panel(canvas,view,crop,x,2275,492,236)
            if j:
                text(pen,x,2526,'No comparable depths' if relation is None else
                     ('Matches recorded order' if relation['near']==near else 'Order reversed'),18)
        regions.append(dict(image=row['id'],floor=row['floor'],region_xyxy=crop,stage='order'))
        break
    else:
        text(pen,34,2190,'No automatic relationship matched a measurable occlusion boundary.',20)
    destination.mkdir(parents=True,exist_ok=True)
    canvas.save(destination/'comparison.jpg',quality=94,subsampling=0)
    save(destination/'review-regions.json',regions)
