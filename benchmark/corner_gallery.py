"""Illustrate changed depth decisions on two fresh, independently scored candidate rims."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from benchmark.live_color_gallery import label_font
from engine.geometry import project


def gallery(root,cache,report,out,method=None):
    selected=method or report['selected']
    examples=[]
    for view,rows in report['views'].items():
        for j,row in enumerate(rows):
            truth=row['target']
            if not truth or not truth[1]:continue
            base,chosen=(row['predictions'][k] for k in ('baseline',selected))
            examples.append((view,j,truth[1],base,chosen))
    improved=next((r for r in examples if r[4]==r[2] and r[3]!=r[2]),None)
    difficult=next((r for r in examples if r[4] is not None and r[4]!=r[2]),None)
    if difficult is None:difficult=next((r for r in examples if r[4] is None),None)
    chosen=[r for r in (improved,difficult) if r is not None] or examples[:1]
    canvas=Image.new('RGB',(380*len(chosen),300),'#182129')
    draw=ImageDraw.Draw(canvas)
    for col,(view,index,depth,base,new) in enumerate(chosen):
        file=next(root.glob('*/captures/'+view+'.png'))
        with np.load(cache/(view+'-'+str(index)+'.npz')) as f:
            source_quad=project(f['H'],f['q']/f['pitch'])
            if new:
                q=f['q']; vp=f['vp']
                # Choose the entrance corner furthest from the vanishing point.
                i=int(np.argmax(np.linalg.norm(q-vp,axis=1)))
                far=vp+(q[i]-vp)/(1+new/float(f['distance']))
                segment=project(f['H'],np.array([q[i],far])/f['pitch'])
        with Image.open(file) as im:
            raw=im.convert('RGB')
        pen=ImageDraw.Draw(raw)
        outer=[tuple(x) for x in source_quad]
        pen.line(outer+outer[:1],fill='#f0c775',width=2)
        if new:
            pen.line([tuple(x) for x in segment],fill='#4ce2cd',width=3)
            x,y=segment[-1];pen.ellipse((x-3,y-3,x+3,y+3),fill='#4ce2cd')
        lo=np.floor(source_quad.min(0)-25).astype(int)
        hi=np.ceil(source_quad.max(0)+25).astype(int)
        picture=raw.crop((*lo,*hi))
        picture.thumbnail((360,190))
        x=col*380+10
        canvas.paste(picture,(x,38))
        draw.text((x,7),f'World measurement: {depth} blocks',font=label_font(18),fill='white')
        draw.text((x,240),f"Previous: {base if base is not None else 'unresolved'}",font=label_font(17),fill='#cbd7da')
        draw.text((x,267),f"{selected.capitalize()}: {new if new is not None else 'unresolved'}",font=label_font(17),fill='#dbe4e5')
    canvas.save(out,quality=94)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','cache','report','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--method',choices=('baseline','edges','texture','agreement','corroborated'))
    a=p.parse_args()
    gallery(a.root,a.cache,json.loads(a.report.read_text()),a.out,a.method)
