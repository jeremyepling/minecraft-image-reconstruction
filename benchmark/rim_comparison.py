"""Draw proposed depths on the fresh flat control and recessed wall at the same angle."""
import argparse
import json
from pathlib import Path

from PIL import Image,ImageDraw
from benchmark.live_color_gallery import label_font


def draw_comparison(root,records,out):
    canvas=Image.new('RGB',(990,255),'#182129')
    draw=ImageDraw.Draw(canvas)
    items=[('controls','dark-inlay-oblique-left','prior_depth','Flat inlays / previous grid'),
           ('controls','dark-inlay-oblique-left','depth','Flat inlays / local spacing'),
           ('original','test-recess-28-oblique-left','depth','Real recesses / local spacing')]
    for col,(folder,view,key,title) in enumerate(items):
        data=records[view]['candidates']
        with Image.open(root/folder/'captures'/(view+'.png')) as image:
            panel=image.convert('RGB')
        pen=ImageDraw.Draw(panel)
        for candidate in data:
            d=candidate[key]
            if d is None:
                continue
            pts=[tuple(p) for p in candidate['rim']]
            actual=candidate['target'][1] if candidate['target'] else 0
            pen.line(pts+pts[:1],width=7,fill='#51dfba' if actual==d else '#ff725e')
        x=330*col
        panel=panel.crop((220,184,1220,705)).resize((324,169),Image.Resampling.LANCZOS)
        canvas.paste(panel,(x+3,39))
        draw.text((x+7,9),title,fill='white',font=label_font(18))
        count=sum(c[key] is not None for c in data)
        wrong=sum(c[key] is not None and (not c['target'] or c['target'][1]!=c[key]) for c in data)
        draw.text((x+7,218),f'{count} measured, {wrong} incorrect',fill='#dae2e5',font=label_font(16))
    canvas.save(out,quality=93)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','results','out'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    draw_comparison(args.root,json.loads(args.results.read_text()),args.out)
