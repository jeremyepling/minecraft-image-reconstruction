"""Show measured decision regions on the chapel and the two observatory wall images."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from benchmark.floating_patch_report import REVIEW, read
from engine.geometry import project

def render():
    rows=[]
    for cohort,fragment in [('sky','chapel'),('supplemental','observatory')]:
        rows.extend(r for r in read(REVIEW/(cohort+'.json.gz'))['frames'] if fragment in r['id'])
    sheet=Image.new('RGB',(1080, len(rows)*250),'white')
    font=ImageFont.truetype('arial.ttf',17)
    for i,row in enumerate(rows):
        source=Image.open(row['source']).convert('RGB')
        marked=source.convert('RGBA')
        layer=Image.new('RGBA',marked.size)
        draw=ImageDraw.Draw(layer)
        packet=read(Path(row['inputs']['whole_chart']))
        cs={c['id']:c for c in packet['charts']}
        for c in read(Path(row['evidence']))['charts']:
            h=np.asarray(cs[c['id']]['H'])
            for region in c['regions']:
                # Draw only supported member cells so boundaries cannot cover sky.
                index=tuple(region['region'])
                for cell in cs[c['id']]['cells']:
                    u,v=cell['u'],cell['v']
                    if (u//4,v//4)!=index:continue
                    corners=project(h,[np.array([u,v])+p for p in [(0,0),(1,0),(1,1),(0,1)]])
                    draw.polygon(list(map(tuple,corners)), fill=(240,58,24,98) if region['rejected'] else (42,191,103,72))
        choices=[source,Image.alpha_composite(marked,layer).convert('RGB'),Image.open(row['packs']['rechart']['folder']+'/source-overlay.jpg').convert('RGB')]
        y=i*250
        for j,pic in enumerate(choices):
            pic.thumbnail((356,208))
            sheet.paste(pic,(360*j+(360-pic.width)//2,y+36))
        a,b=[row['scores'][k]['geometry']['strict'] for k in ('whole','rechart')]
        label=row['id']+'   |   '+str(a['correct'])+' → '+str(b['correct'])+' correct, '+str(a['extras'])+' → '+str(b['extras'])+' extra'
        ImageDraw.Draw(sheet).text((7,y+3),label,fill='black',font=font)
    sheet.save(REVIEW/'patches.jpg',quality=90)

if __name__=='__main__':
    render()

