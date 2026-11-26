"""Draw grouped visible faces, marking the face selected for material identification."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_examples(predictions, truth_file, output):
    data = json.loads(gzip.decompress(predictions.read_bytes()))
    labels = {f['key']:{r['id']:r for r in f['faces']} for f in json.loads(truth_file.read_bytes())['frames']}
    fixture = Path('benchmark/reviews/mask-four-01')
    sources = {r['key']:fixture/r['source'] for r in json.loads((fixture/'cases.json').read_bytes())}
    examples=[]
    for frame in data['frames']:
        if frame['kind']!='recorded':
            continue
        truth=labels[frame['key']]
        faces={r['id']:r for r in frame['faces']}
        for g in frame['groups']:
            if len(g['face_ids'])<2 or g['selected_id'] is None:
                continue
            members=[faces[n] for n in g['face_ids']]
            chosen=faces[g['selected_id']]
            same=len({tuple(truth[n]['xyz']) for n in g['face_ids']})==1
            sliver=any(r['effective_width_px']<5 for r in members)
            examples.append((frame['key'],members,chosen,truth[g['selected_id']]['block_id'],same,sliver))
    selected=[]
    for cond in (lambda e:len(e[1])==3,lambda e:e[-1],lambda e:not e[-2],lambda e:e[2]['grouped']['texture']!=e[3],lambda e:True,lambda e:True):
        options=[e for e in examples if cond(e) and all(o[0].split('/')[-1]!=e[0].split('/')[-1] for o in selected)]
        shown={e[0].split('/')[-1].split('-')[1] for e in selected}
        item=max(options,key=lambda e:e[0].split('/')[-1].split('-')[1] not in shown,default=None)
        if item is not None:selected.append(item)
    if not selected:raise RuntimeError('No joined groups to illustrate')
    w,h=470,302
    sheet=Image.new('RGB',(w*2,110+h*((len(selected)+1)//2)),'#f3f5f3')
    pen=ImageDraw.Draw(sheet)
    def label(x,y,s,size,bold=False):
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui'+('b' if bold else '')+'.ttf',size)
        pen.text((x,y),s,font=font,fill='#19363a')
    label(18,8,'One material pick from connected faces',32,True)
    label(18,61,'Recorded face outlines supplied. Green = chosen face. Gold = linked face using the same pick.',17)
    for i,(key,members,chosen,actual,same,sliver) in enumerate(selected):
        x,y=(i%2)*w,110+(i//2)*h
        label(x+12,y+4,f"{i+1:02}   {key.split('/')[-1]}",17,True)
        im=Image.open(sources[key]).convert('RGB')
        coords=np.concatenate([m['quad'] for m in members])
        center=coords.mean(axis=0)
        span=max(25,np.ptp(coords,axis=0).max()*1.8)
        box=np.rint([*(center-span/2),*(center+span/2)]).astype(int)
        crop=im.crop(tuple(box)).resize((190,190),Image.Resampling.NEAREST)
        sheet.paste(crop,(x+12,y+36))
        overlay=crop.copy()
        lines=ImageDraw.Draw(overlay)
        for m in members:
            q=(np.asarray(m['quad'])-box[:2])/(box[2:]-box[:2])*190
            p=[tuple(point) for point in q]
            lines.line(p+p[:1],fill='#28e790' if m['id']==chosen['id'] else '#ffc436',width=3)
        sheet.paste(overlay,(x+210,y+36))
        label(x+12,y+232,f"{len(members)} faces   /   "+('same recorded block' if same else 'incorrect join'),16,True)
        label(x+12,y+259,'Texture pick: '+str(chosen['grouped']['texture'])+'    Actual: '+actual,15)
    output.parent.mkdir(exist_ok=True,parents=True)
    sheet.save(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('predictions','truth','output'):parser.add_argument(name,type=Path)
    a=parser.parse_args()
    draw_examples(a.predictions,a.truth,a.output)
