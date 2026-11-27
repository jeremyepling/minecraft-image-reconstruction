"""Compare the source and reference masks on the views with the largest placement changes."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import ROOT, SOURCE, read
from engine.surface_perimeter import _native_alpha


def delta(row):
    return row['native']['score']['geometry']['strict']['f1']-row['source']['score']['geometry']['strict']['f1']


def annotated(image, file):
    output=np.array(image)
    rng=np.random.default_rng(47)
    for p in read(file)['planes']:
        area=_native_alpha(p['support'],image.size)
        if not area:
            continue
        alpha,(x,y,r,b)=area
        selected=np.asarray(alpha)!=0
        region=output[y:b,x:r]
        color=rng.integers(30,245,size=3)
        region[selected]=.50*region[selected]+.50*color
    return Image.fromarray(output)


def make_sheet(receipts):
    base=ROOT/'runs/native-crop-ceiling'
    rows=sorted(receipts,key=delta)
    selected=[rows[-1],rows[0]]
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    page=Image.new('RGB',(1230,625),'#ffffff')
    draw=ImageDraw.Draw(page)
    for j,title in enumerate(('Source image','Surface Finder masks','Exact visible masks')):
        draw.text((12+j*410,5),title,fill='black',font=font)
    for i,row in enumerate(selected):
        y=42+i*288
        source=Image.open(SOURCE/row['dataset']/'data/test/frames'/row['id']/'rgb.png').convert('RGB')
        images=[source]+[annotated(source,base/arm/'sky'/row['id']/'source-handoff/surfaces.json') for arm in ('source','native')]
        for j,image in enumerate(images):
            image.thumbnail((400,250))
            page.paste(image,(12+j*410,y))
        left,right=[row[a]['score']['geometry']['strict']['f1']*100 for a in ('source','native')]
        draw.text((12,y+251),f"{row['id']}    Placement F1  {left:.2f}% to {right:.2f}%",fill='black',font=font)
    page.save(ROOT/'benchmark/reviews/expanded-crop-ceiling-01/crop-comparison.jpg',quality=91)


if __name__=='__main__':
    make_sheet(read(ROOT/'benchmark/reviews/expanded-crop-ceiling-01/scored-exports.json.gz')['cases'])
