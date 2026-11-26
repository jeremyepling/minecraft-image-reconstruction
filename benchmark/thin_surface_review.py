"""Draw actual surface proposals in bins of narrowest enclosing strip width."""
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'reviews/thin-surfaces-01'
BANDS = [(0,5),(5,8),(8,10),(10,12),(12,16),(16,24),(24,40)]


def hull(points):
    def half(seq):
        chain=[]
        for x,y in seq:
            while len(chain)>=2:
                a,b=chain[-2:]
                if (b[0]-a[0])*(y-b[1])>(b[1]-a[1])*(x-b[0]):
                    break
                chain.pop()
            chain.append((x,y))
        return chain[:-1]
    seq=sorted(set(points))
    return np.array(half(seq)+half(reversed(seq)))


def measure(support, size):
    sx,sy=np.array(size)/support['size']
    bounds={}
    for y,x,end in support['runs']:
        left,right=bounds.get(y,(x,end))
        bounds[y]=(min(x,left),max(end,right))
    points=[(x,v) for y,(a,b) in bounds.items() for x in (a,b) for v in (y,y+1)]
    polygon=hull(points)*[sx,sy]
    sides=np.roll(polygon,-1,axis=0)-polygon
    normals=sides[:,::-1]*[-1,1]
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    span=np.ptp(polygon@normals.T,axis=0).min()
    box=[*polygon.min(0),*polygon.max(0)]
    return float(span),list(map(int,box))


def paint(row, x, y, canvas, draw):
    image=Image.open(row['source']).convert('RGB')
    x0,y0,x1,y1=row['box']
    region=Image.new('L',row['support']['size'])
    pen=ImageDraw.Draw(region)
    for v,a,b in row['support']['runs']:
        pen.line((a,v,b-1,v),fill=255)
    mask=region.resize(image.size,Image.Resampling.NEAREST)
    overlay=Image.composite(Image.blend(image,Image.new('RGB',image.size,'#12f7cc'),.65),image,mask)
    # Keep the context aspect ratio and include the entire proposal.
    bounds=(x0-20,y0-20,x1+20,y1+20)
    context=ImageOps.contain(overlay.crop(bounds),(272,124))
    canvas.paste(context,(x+(274-context.width)//2,y+34+(124-context.height)//2))
    # Select a genuinely occupied pixel near the center and magnify at exactly 4x.
    yy,xx=np.where(np.asarray(mask.crop((x0,y0,x1,y1)))>0)
    k=np.argmin((xx-np.median(xx))**2+(yy-np.median(yy))**2)
    px,py=x0+int(xx[k]),y0+int(yy[k])
    crop=(px-33,py-10,px+34,py+11)
    canvas.paste(image.crop(crop).resize((268,84),Image.Resampling.NEAREST),(x+3,y+165))
    draw.text((x+4,y+4),f"{row['case']:02}   {row['width']:.1f} px thin",font=FONT_B,fill='#073c39')
    draw.text((x+4,y+253),f"{x1-x0} × {y1-y0} px bounds",font=FONT,fill='#263c41')
    draw.text((x+4,y+276),row['key'].split('/')[-1].split(row['layout']+'-')[-1],font=FONT,fill='#263c41')


if __name__=='__main__':
    FONT=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    FONT_B=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',20)
    fixture=ROOT/'reviews/mask-four-01'
    images={r['key']:fixture/r['source'] for r in json.loads((fixture/'cases.json').read_text())}
    archive=ROOT/'reviews/crop-split-01'
    rows=[]
    for case in json.loads((archive/'cases.json').read_text()):
        with Image.open(images[case['key']]) as im:
            size=im.size
        for surface in json.loads(gzip.decompress((archive/case['predictions']).read_bytes()))['split']:
            width,box=measure(surface['support'],size)
            rows.append(dict(key=case['key'],layout=case['layout'],id=surface['id'],box=box,width=width,
                source=str(images[case['key']]),support=surface['support']))
    layouts=['shopfront','wing','galleries','workshop']
    used=set()
    canvas=Image.new('RGB',(1250,164+len(BANDS)*312),'#edf2f0')
    d=ImageDraw.Draw(canvas)
    d.text((18,7),'Blocker Picker  /  How thin is the visible surface?',font=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',36),fill='#153b38')
    d.text((20,66),'Mint = whole predicted surface, fitted to panel. Below = uncolored detail at 4×. Sizes refer to original 1440 × 900 screenshots.',font=FONT,fill='#263c41')
    selected=[]
    stats={}
    for j,layout in enumerate(layouts):
        d.text((125+j*279,111),layout.title(),font=FONT_B,fill='#153b38')
    for i,(lo,hi) in enumerate(BANDS):
        band=[r for r in rows if lo<=r['width']<hi]
        stats[f'{lo}-{hi}']=len(band)
        y=154+i*312
        d.text((9,y+7),f'{lo}–<{hi} px',font=FONT,fill='#153b38')
        d.text((9,y+37),f'{len(band)} masks',font=FONT,fill='#153b38')
        for j,layout in enumerate(layouts):
            options=[r for r in band if r['layout']==layout]
            x=122+j*279
            d.rectangle((x,y,x+274,y+301),fill='white')
            if not options:
                d.text((x+9,y+110),'No proposal in this bin',font=FONT,fill='#536565')
                continue
            def priority(r):
                a,b,c,e=r['box']
                landscape=c-a>e-b
                return (r['key'] not in used,landscape==bool((i+j)%2),-abs(r['width']-(lo+hi)/2))
            r=max(options,key=priority)
            used.add(r['key'])
            r['case']=len(selected)+1
            selected.append({k:v for k,v in r.items() if k not in ('support','source')})
            paint(r,x,y,canvas,d)
    OUT.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT/'surfaces.png')
    (OUT/'cases.json').write_text(json.dumps(dict(method='crop-split-01 / split',frames=len(images),total=len(rows),
        thin_span='minimum distance between parallel lines enclosing the whole predicted mask in source pixels',
        bands=stats,at_least_40=sum(r['width']>=40 for r in rows),examples=selected),indent=2))
    print(stats,len(rows),len(selected))
