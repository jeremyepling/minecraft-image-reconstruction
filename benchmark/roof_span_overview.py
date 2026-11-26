"""Render the ten pooled layout ranks and their weakest full source pictures."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmark.texture_camera_eval import read
from benchmark.texture_camera_sheet import font


def draw(ranking,out):
    rows=read(ranking)['layouts'][:10]
    assert len({r['shape_sha256'] for r in rows})==10
    page=Image.new('RGB',(1296,1048),'#122530')
    pen=ImageDraw.Draw(page)
    def label(x,y,s,size,color='#eff4f4'):
        pen.text((x,y),s,font=font(size),fill=color)
    label(18,13,'Ten lowest-scoring structures',31)
    label(18,58,'Placement on the source photo    Green = correct    Red = extra    Blue = missed visible block',19)
    for r in rows:
        i=r['rank']-1
        x=16+(i%3)*426
        y=107+(i//3)*232
        with Image.open(ranking.parent/'overlays'/f"{r['rank']:02}-errors.jpg") as photo:
            thumb=ImageOps.contain(photo,(400,157),Image.Resampling.LANCZOS)
        page.paste(thumb,(x+(400-thumb.width)//2,y))
        name=r['name']
        size=23 if len(name)<24 else 18
        label(x+5,y+163,f"{r['rank']}. {name}",size)
        label(x+5,y+194,f"Layout {r['f1']:.1%}  |  Shown view {r['representative']['f1']:.1%}",17,'#c4d3da')
    page.save(out,quality=93)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args()
    draw(a.ranking,a.out)
