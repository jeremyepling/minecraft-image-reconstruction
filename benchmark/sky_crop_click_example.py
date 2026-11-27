"""Render one development crop's actual successive accepted SAM edits on native RGB."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import SOURCE, read
from benchmark.sky_crop_corrections import OUTPUT, native_mask, reference_raster

REVIEW = Path(__file__).resolve().parent / 'reviews/sky-crop-corrections-01'


def example():
    trace = read(OUTPUT / 'trace.json')
    cases = {c['id']:c for c in read(SOURCE/'floating-build-grid-01/fixtures.json')['cases']}
    choices = [(row['final_iou']-row['baseline_iou'], frame, ident, row)
               for frame,rows in trace['frames'].items() if cases[frame]['split']=='train'
               for ident,row in rows.items() if sum(s['accepted'] for s in row['steps'])>=3]
    _, frame, ident, row = max(choices, key=lambda e:e[:3])
    case = cases[frame]
    image = np.asarray(Image.open(case['variants']['sky']['image']).convert('RGB'))
    ref = reference_raster(Path(case['frame']))
    target = ref.labels == next(i for i,t in enumerate(ref.targets,1) if t['id']==row['target'])
    accepted = [s for s in row['steps'] if s['accepted']]
    selected = [None, *accepted[:3], *([accepted[-1]] if len(accepted) > 3 else [])]
    masks = [native_mask(read(OUTPUT/f"round-{(s or {}).get('round',0):02}.json")['frames'][frame][ident],
                         image.shape[1::-1]) for s in selected]
    ys,xs = np.nonzero(np.logical_or.reduce([target,*masks]))
    x0,x1 = max(0,xs.min()-8), min(image.shape[1],xs.max()+9)
    y0,y1 = max(0,ys.min()-8), min(image.shape[0],ys.max()+9)
    panel_w,panel_h=292,280
    sheet=Image.new('RGB',(panel_w*len(selected),390),'#101a25')
    draw=ImageDraw.Draw(sheet)
    title=ImageFont.truetype('arial.ttf',20)
    label=ImageFont.truetype('arial.ttf',15)
    draw.text((8,6),f"{case['family']}  /  {row['label']}  /  reference-guided SAM edits",font=title,fill='white')
    draw.text((8,35),'Green = correct     Magenta = extra     Amber = missing     Circle = new corrective click',font=label,fill='white')
    for i,(step,mask) in enumerate(zip(selected,masks)):
        overlay=image.copy()
        for pixels, color in [(mask&target,[35,219,101]),(mask&~target,[243,64,196]),(target&~mask,[255,179,32])]:
            overlay[pixels]=.4*overlay[pixels]+.6*np.array(color)
        tile=Image.fromarray(overlay[y0:y1,x0:x1])
        scale=min((panel_w-12)/tile.width,panel_h/tile.height)
        tile=tile.resize((max(1,round(tile.width*scale)),max(1,round(tile.height*scale))),Image.Resampling.NEAREST)
        px=panel_w*i+(panel_w-tile.width)//2
        py=62+(panel_h-tile.height)//2
        sheet.paste(tile,(px,py))
        if step:
            cx=px+(step['x']-x0+.5)*scale
            cy=py+(step['y']-y0+.5)*scale
            draw.ellipse((cx-7,cy-7,cx+7,cy+7),outline='black',width=4)
            draw.ellipse((cx-7,cy-7,cx+7,cy+7),outline='white',width=2)
        heading = f"Click {step['round']}  |  IoU {step['after_iou']:.3f}" if step else f"Original  |  IoU {row['baseline_iou']:.3f}"
        draw.text((i*panel_w+7,346),heading,font=label,fill='white')
        if step:
            draw.text((i*panel_w+7,369),('Include' if step['label'] else 'Exclude')+f"  ({step['x']}, {step['y']})",font=label,fill='#bdcedb')
    REVIEW.mkdir(parents=True,exist_ok=True)
    sheet.save(REVIEW/'example.png')
    (REVIEW/'example.json').write_text(json.dumps(dict(frame=frame,id=ident,**row),indent=2))
    print(frame,ident,row['label'],row['baseline_iou'],row['final_iou'])


if __name__ == '__main__':
    example()
