"""Use the same saved camera view for every column and keep recorded labels in their own column."""
import gzip
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark.camera_round import read, save
from benchmark.normal_building_sheet import REFERENCE, selected_views
from benchmark.surface_reference import recorded, review_case
from benchmark.surface_sheet import scene_box, text
from benchmark.mask_metrics import footprint


TITLES={'current-b':'Current B','mask-first':'Regions first','scalelsd':'+ ScaleLSD v2','deeplsd':'+ DeepLSD'}
COLORS=['#66dce5','#f4b67a','#9bc2ff','#df98d4','#a5d484','#e0ca75','#bcadff']
PANEL=(448,263)


def region_view(source,regions,is_mask):
    img=source.copy()
    w,h=source.size
    # Render every prediction alone. No ground-truth outline or match decision is involved.
    shape=(h//4,w//4)
    masks=[footprint(p,shape,source.size,is_mask) for p in regions]
    for idx in sorted(range(len(masks)),key=lambda i:-int(masks[i].sum())):
        mask=Image.fromarray(masks[idx]).resize(source.size,Image.Resampling.NEAREST).convert('L')
        img.paste(COLORS[idx%len(COLORS)],(0,0,w,h),mask.point(lambda x:x*.43))
        inside=mask.filter(ImageFilter.MinFilter(5))
        border=Image.fromarray(np.asarray(mask)-np.asarray(inside))
        img.paste('#083c4c',(0,0,w,h),border)
    return img


def reference_view(source,reference):
    ids=reference['id_map']
    edge=np.zeros(ids.shape,bool)
    edge[:-1]|=ids[:-1]!=ids[1:]
    edge[:,:-1]|=ids[:,:-1]!=ids[:,1:]
    img=Image.blend(source,Image.new('RGB',source.size,'#14232c'),.28)
    img.paste('#ffe476',(0,0,*source.size),Image.fromarray(edge).resize(source.size,Image.Resampling.NEAREST))
    return img


def framed(box,size,detail):
    x0,y0,x1,y1=box
    x,y=(x0+x1)/2,(y0+y1)/2
    w,h=x1-x0,y1-y0
    if detail:
        fx,fy,scale=detail
        x,y,w,h=x0+fx*w,y0+fy*h,w*scale,h*scale
    w=max(w,h*PANEL[0]/PANEL[1])
    h=w*PANEL[1]/PANEL[0]
    x0=max(0,min(x-w/2,size[0]-w))
    y0=max(0,min(y-h/2,size[1]-h))
    return tuple(round(v) for v in (x0,y0,min(size[0],x0+w),min(size[1],y0+h)))


def render(out,rows,methods):
    available={r['key']:r for r in rows}
    chosen=[available[r['key']] for r in selected_views() if r['key'] in available]
    details={'shopfront':('Inside the shop bays',(.53,.59,.40)),
             'wing':('Where the two walls meet',(.67,.53,.38)),
             'galleries':('Between the lower pillars',(.53,.71,.41))}
    views=[(row,None,None) for row in chosen]+[(r,*details[r['layout']]) for r in chosen if r['layout'] in details]
    columns=['Source screenshot']+[TITLES[m] for m in methods]+['MoGe directions','Recorded reference']
    margin,gap,stride=28,12,350
    width=2*margin+len(columns)*(PANEL[0]+gap)-gap
    canvas=Image.new('RGB',(width,171+len(views)*stride),'#f7f6ef')
    pen=ImageDraw.Draw(canvas)
    text(pen,margin,16,'Which surfaces survive?',43,True)
    text(pen,margin,77,'Colored patches show predictions. Gold edges in the last column show the recorded building. Each row uses one screenshot.',25)
    for i,title in enumerate(columns):
        text(pen,margin+i*(PANEL[0]+gap),128,title,26,True)
    crops=[]
    with ZipFile(REFERENCE/'direction-previews.zip') as previews:
        for k,(row,detail_title,detail) in enumerate(views):
            y=171+k*stride
            case=review_case(out,row)
            with Image.open(case.path('image')) as f:
                source=f.convert('RGB')
            _,reference=recorded(case)
            with gzip.open(out/row['predictions'],'rt',encoding='utf8') as f:
                predictions=json.load(f)
            with Image.open(BytesIO(previews.read(row['sha256']+'.png'))) as colors:
                directions=colors.convert('RGB').resize(source.size,Image.Resampling.BILINEAR)
            images=[source]+[region_view(source,predictions[m],m!='current-b') for m in methods]+[directions,reference_view(source,reference)]
            crop=framed(scene_box(reference,source),source.size,detail)
            title=detail_title+'  /  '+row['layout'].title() if detail else row['layout'].title()
            text(pen,margin,y+3,title,26,True)
            for j,im in enumerate(images):
                x=margin+j*(PANEL[0]+gap)
                canvas.paste(im.crop(crop).resize(PANEL,Image.Resampling.LANCZOS),(x,y+44))
                if 0<j<=len(methods):
                    s=row['scores'][methods[j-1]]
                    caption=f"{s['matched']} of {s['eligible']} found   /   {s['proposed']-s['matched']} extra"
                    if detail:
                        caption='Enlarged region'
                    text(pen,x,y+315,caption,19)
            crops.append(dict(key=row['key'],native_xyxy=crop,detail=detail_title))
    canvas.save(out/'comparison.jpg',quality=92,subsampling=0)
    save(out/'review-regions.json',crops)
