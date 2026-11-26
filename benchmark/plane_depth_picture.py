"""Draw the selected and original depth grids inside the observed mask."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from benchmark.atlas_plane_round import CACHE,DATA,OUT,read
from engine.plane_offset_features import features
from engine.visible_raster import observed_support


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)


def render(rgb,mask,P,axis,level,color):
    y,x=np.nonzero(mask)
    pixels=np.stack([x*4+2,y*4+2,np.ones(len(y))],axis=1)
    eye=-np.linalg.solve(P[:,:3],P[:,3])
    rays=pixels @ np.linalg.inv(P[:,:3]).T
    depth=(level-eye[axis])/rays[:,axis]
    free=[n for n in range(3) if n!=axis]
    pts=(eye+rays*depth[:,None])[(depth>1.5)&(depth<150)][:,free]
    lo=np.floor(pts.min(0)).astype(int)
    hi=np.ceil(pts.max(0)).astype(int)
    overlay=rgb.copy()
    pen=ImageDraw.Draw(overlay)
    for a in range(2):
        b=1-a
        for k in range(lo[a],hi[a]+1):
            xyz=np.full((2,3),level)
            xyz[:,free[a]]=k
            xyz[:,free[b]]=[lo[b],hi[b]]
            q=xyz @ P[:,:3].T+P[:,3]
            if min(q[:,2])>0:
                coordinates=q[:,:2]/q[:,2:]
                pen.line([tuple(v) for v in coordinates],fill=color,width=2)
    full_mask=Image.fromarray(mask).resize(rgb.size,Image.Resampling.NEAREST)
    mixed=Image.composite(overlay,rgb,full_mask)
    left,top,right,bottom=full_mask.getbbox()
    return mixed.crop((max(0,left-28),max(0,top-28),min(right+28,rgb.width),min(bottom+28,rgb.height)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    key='train-steppedhall1-cool-high-right'
    mask_id='mask-0-junction-0'
    frame=read(CACHE/(key+'.json'))
    rows=read(OUT/'wide-features/train'/(key+'.json'))['rows']
    row=next(r for r in rows if r['id']==mask_id)
    options=row['choices']
    scores=features(options,row['baseline']) @ read(a.model)['weights']
    winner=options[int(scores.argmax())]['level']
    image=Image.open(DATA/'train/frames'/key/'rgb.png').convert('RGB')
    part=next(r for r in frame['parts'] if r['id']==mask_id)
    shape=tuple((v+3)//4 for v in image.size[::-1])
    mask=observed_support(part['support'],shape,image.size)
    canvas=Image.new('RGB',(1020,530),'#18212b')
    draw=ImageDraw.Draw(canvas)
    draw.text((16,10),'Visible texture and grid alignment on Stepped Hall',font=font(23))
    for i,(name,level,color) in enumerate([('Original',row['baseline'],'#ffb14b'),('Ranked',winner,'#18eaff')]):
        crop=render(image,mask,np.asarray(frame['camera']['P']),row['axis'],level,color)
        scale=min(488/crop.width,270/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.BILINEAR)
        left=15+505*i
        draw.text((left,47),f'{name} plane at local offset {level}',font=font(19),fill=color)
        canvas.paste(crop,(left,74))
        opt=next(r for r in options if r['level']==level)
        draw.text((left,355),f"Atlas agreement {opt['atlas']:.2f}   ·   Boundary agreement {opt['boundary']:.2f}",font=font(16))
    draw.text((16,392),'Candidate depth offsets ranked from the same screenshot',font=font(17))
    lo,hi=float(min(scores)),float(max(scores))
    width=980/len(options)
    for i,(opt,score) in enumerate(zip(options,scores)):
        x=17+width*i
        length=6+61*(score-lo)/max(.01,hi-lo)
        color='#18eaff' if opt['level']==winner else '#ffb14b' if opt['level']==row['baseline'] else '#8498a9'
        draw.rectangle((x,488-length,x+width*.76,488),fill=color)
        draw.text((x,495),str(opt['level']),font=font(14),fill=color)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(a.out,quality=91)


if __name__=='__main__':
    main()
