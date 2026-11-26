"""Replay selected Blocker Picker cases with shade evidence and before/after choices."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.blocker_picker_sheet import read, pretty
from engine.geometry import face


def replay(args):
    old = read(args.sheet)
    combined = read(args.results)
    material = {p['key']: p for p in read(Path('benchmark/reviews/count-01/materials.json'))['pictures']}
    lookup = {}
    for row in combined['rows']:
        original = material[row['key']][row['group']][row['index']]
        lookup[row['key'], row['group'], tuple(row['xyz']), original['direction']] = row
    meta = read(args.atlas/'atlas.json')
    catalog = read(args.atlas/'placement-index.json')['choices']
    sprites = {t['id'].split('/')[-1]: t['rect'] for t in meta['textures']}
    atlas = Image.open(args.atlas/meta['atlas']).convert('RGBA')
    width = 1270
    sheet = Image.new('RGB', (width, 135+((len(old)+1)//2)*288), '#e9efee')
    draw = ImageDraw.Draw(sheet)
    def label(x,y,value,size=16,bold=False,color='#173536'):
        font = ImageFont.truetype('C:/Windows/Fonts/segoeui'+('b' if bold else '')+'.ttf',size)
        draw.text((x,y),value,font=font,fill=color)
    label(20,8,'Blocker Picker   /   Texture and shade replay',34,True)
    label(20,64,'Cyan outlines the tested face. Orange shows the estimated lighting correction. Stronger orange means more brightening.')
    label(20,91,'The previous and new picks refer to the same block. All material swatches use vanilla atlas art.')
    for i,case in enumerate(old):
        x,y=12+628*(i%2),132+288*(i//2)
        draw.rounded_rectangle((x,y,x+615,y+277),8,fill='white')
        group='predicted' if case['kind']=='predicted' else 'known'
        row=lookup[case['key'],group,tuple(case['xyz']),case['direction']]
        guess=row['shape']
        correct=guess['block_id']==case['truth']
        label(x+10,y+7,f"{i+1:02}   {case['key'].split('/')[-1]}",18,True)
        label(x+475,y+7,'Corrected' if correct else 'Still missed',15,True,'#087746' if correct else '#963b26')
        label(x+10,y+35,('Predicted grid' if group=='predicted' else 'Recorded corners')+'   /   '+case['direction']+' face')
        image=Image.open(case['image']).convert('RGB')
        q=np.array(row['quad'])
        center=q.mean(0)
        span=max(36,3*np.ptp(q,axis=0).max())
        box=np.rint([*(center-span/2),*(center+span/2)]).astype(int)
        side=106
        context=image.crop(tuple(box)).resize((side,side),Image.Resampling.NEAREST)
        mask=Image.open(args.maps/(case['key'].replace('/','__')+'.png')).crop(tuple(box)).resize((side,side))
        overlay=Image.composite(Image.new('RGB',context.size,'#ff871c'),context,mask)
        path=[tuple(pt) for pt in (q-box[:2])/(box[2:]-box[:2])*side]
        for j,im in enumerate((context,overlay)):
            ImageDraw.Draw(im).line(path+path[:1],fill='#2de8db',width=2)
            sheet.paste(im,(x+10+121*j,y+68))
        crop=Image.fromarray(face(np.asarray(image),q,32)).resize((side,side),Image.Resampling.NEAREST)
        sheet.paste(crop,(x+252,y+68))
        for j,texture in enumerate((guess['texture'],next((name for name,ids in catalog[case['direction']].items() if case['truth'] in ids),case['truth']))):
            if texture in sprites:
                u,v,w,h=sprites[texture]
                tile=atlas.crop((u,v,u+w,v+h)).resize((side,side),Image.Resampling.NEAREST)
                sheet.paste(tile,(x+373+j*121,y+68),tile)
        for j,t in enumerate(('Source','Shade estimate','Sampled pixels','New pick','Actual')):
            label(x+10+j*121,y+179,t,15,True)
        label(x+10,y+207,'Previously  '+pretty(case['block_id'])+'     /     Now  '+pretty(guess['block_id']),16)
        label(x+10,y+237,'Actual  '+pretty(case['truth'])+'     /     Estimated brightening  '+format(row['shadow']['mean_gain']-1,'.0%'),16)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(args.out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('sheet','results','maps','atlas','out'):
        parser.add_argument(name,type=Path)
    replay(parser.parse_args())
