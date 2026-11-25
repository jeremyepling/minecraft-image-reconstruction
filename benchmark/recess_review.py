"""Measure blind recess predictions against projected opening positions and flat controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.live_color_gallery import label_font
from calibration.projection import camera_project
from engine.depth_layers import inspect_depth
from engine.geometry import project, transform


def known_openings(building,pose):
    found={}
    for r in building['regions']:
        if r['kind'] not in ('recess','heldout_opening'):
            continue
        a,b=r['bounds']
        quad=[[a[0],b[1],b[2]],[b[0],b[1],b[2]],[b[0],a[1],b[2]],[a[0],a[1],b[2]]]
        depth=r['depth'] if 'depth' in r else next(m['value'] for m in building['measurements'] if m['id']==r['id']+'_depth')
        found[r['id']]=(np.linalg.inv(transform(camera_project(quad,pose))),depth)
    return found


def inspect(path, known):
    with Image.open(path) as im:
        result=inspect_depth(np.asarray(im.convert('RGB')))
    observations=[]
    for r in result['recesses']:
        xy=np.mean(r['rim'],axis=0)
        target=None
        for name,(inv,d) in known.items():
            uv=project(inv,xy)
            if ((uv>=0)&(uv<=1)).all():
                target=name
                break
        observations.append(dict(target=target,**r))
    return dict(ground_calibrated=bool(result['calibration']),openings={k:v[1] for k,v in known.items()},
                candidates=observations,ledges=result['ledges'])


def illustration(source,controls,rows,path):
    views=[next(k for k in rows if k.startswith('test-') and k.endswith('-front')),
           'dark-inlay-front','physical-shade-front']
    images=[]
    for j,view in enumerate(views):
        with Image.open((source if j==0 else controls)/'captures'/(view+'.png')) as im:
            images.append(im.convert('RGB'))
    canvas=Image.new('RGB',(1020,540),'#182129')
    draw=ImageDraw.Draw(canvas)
    measured=[r for r in rows[views[0]]['candidates'] if r['depth'] is not None]
    labels=['Recessed stone wall','Flush dark inlays','Flat wall under shade']
    # All controls use the same camera. Crop the same portion in all three.
    for j,im in enumerate(images):
        w,h=im.size
        crop=im.crop((int(w*.13),int(h*.28),int(w*.86),int(h*.73)))
        crop.thumbnail((330,170))
        canvas.paste(crop,(j*340+5,40))
        draw.text((j*340+6,10),labels[j],font=label_font(20),fill='white')
        n=sum(r['depth'] is not None for r in rows[views[j]]['candidates'])
        draw.text((j*340+6,194),f'{n} depth estimates',font=label_font(16),fill='#c2d2df')
    if measured:
        row=max(measured,key=lambda r:np.linalg.norm(np.diff(r['reveal_segment'],axis=0)))
        rim=np.array(row['rim']);back=np.array(row['back_rim'])
        start,end=map(np.asarray,row['reveal_segment'])
        i=np.linalg.norm(rim-start,axis=1).argmin()
        lo=np.floor(rim.min(0)-15).astype(int);hi=np.ceil(rim.max(0)+15).astype(int)
        crop=images[0].crop((*lo,*hi))
        ratio=min(430/crop.width,303/crop.height)
        crop=crop.resize((round(crop.width*ratio),round(crop.height*ratio)),Image.Resampling.NEAREST)
        canvas.paste(crop,(32,223))
        def line(points,color):
            pts=[tuple((np.array(p)-lo)*ratio+[32,223]) for p in points]
            draw.line(pts,fill=color,width=3)
        line([*rim,rim[0]],'#ffcd63')
        # Clip each inferred back edge to the visible entrance for display.
        inv=np.linalg.inv(transform(rim))
        for neighbor in ((i+1)%4,(i-1)%4):
            pts=np.linspace(back[i],back[neighbor],120)
            uv=project(inv,pts)
            pts=pts[((uv>=0)&(uv<=1)).all(axis=1)]
            if len(pts)>1:
                line(pts,'#57d7dc')
        line([start,end],'#ff866d')
        for j,(txt,color) in enumerate([('Outer rim','#ffcd63'),('Visible part of predicted back rim','#57d7dc'),
                                       ('Connecting edge on the inside face','#ff866d')]):
            draw.text((485,247+43*j),txt,font=label_font(21),fill=color)
        target=row['target'];actual=rows[views[0]]['openings'].get(target)
        draw.text((485,407),f"Estimated {row['depth']} blocks deep",font=label_font(23),fill='white')
        draw.text((485,450),f'World measurement: {actual} blocks',font=label_font(19),fill='#c2d2df')
    canvas.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','controls','out','image'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--positive',type=Path)
    a=parser.parse_args()
    buildings={b['id']:b for b in json.loads((a.source/'buildings.json').read_text())}
    rows={}
    for root in filter(None,(a.source,a.controls,a.positive)):
        if root==a.positive:
            buildings.update({b['id']:b for b in json.loads((root/'buildings.json').read_text())})
        for path in sorted((root/'captures').glob('*.png')):
            view=path.stem
            if root==a.source and '-recess-' not in view:
                continue
            pose=json.loads(path.with_suffix('.json').read_text())
            truth=known_openings(buildings[view.rsplit('-',1)[0]],pose) if root!=a.controls else {}
            rows[view]=inspect(path,truth)
    a.out.write_text(json.dumps(rows,indent=2)+'\n')
    illustration(a.source,a.controls,rows,a.image)


if __name__=='__main__':
    main()
