"""Display the common distant scoring region and candidate grids over native source crops."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from benchmark.floating_distant_seams import ROOT,REVIEW,data,locations,OUT,read,common_domain,project,_native_alpha

EXAMPLES = [('sky','validation-hillside-chapel-1-warm-low-left'),('holdout3','train-sawtooth2-cool-high-right'),('supplemental','train-watchpost2-cool-low-left')]


def main():
    out = Image.new('RGB',(1080,750),'#132332')
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype('arial.ttf',17)
    for row,(g,n) in enumerate(EXAMPLES):
        src,_,whole,cases = data(g)
        rel = Path('sky')/n/'charts.json'
        choices = read(OUT/g/'audit'/(n+'.json'))['charts']
        rec = max(choices,key=lambda r:(r['accepted'],r['new']['segments']))
        cid = rec['id']
        old,new = [next(c for c in read(f/rel)['charts'] if c['id']==cid) for f in (whole,locations(g)/'charts')]
        case = next(c for c in cases if c['id']==n)
        image = Image.open(case['variants']['sky']['image']).convert('RGB')
        part = next(p for p in read(src/'sky'/n/'source-handoff/surfaces.json')['planes'] if p['id']==cid)
        alpha,(x,y,x1,y1) = _native_alpha(part['support'],image.size)
        mask = np.zeros(image.size[::-1],bool)
        mask[y:y1,x:x1] = np.asarray(alpha)>0
        domain = common_domain(mask,old,new)
        draw.text((6,row*250+4),f"{n}   {cid}   |   "+('switched to proposed ruler' if rec['accepted'] else 'kept previous ruler'),font=font)
        for i,chart in enumerate((None,old,new)):
            pic = image.copy()
            if chart:
                layer = pic.copy()
                pen = ImageDraw.Draw(layer)
                h = np.asarray(chart['H'])
                for cell in chart['cells']:
                    corners = project(h,np.asarray([[0,0],[1,0],[1,1],[0,1],[0,0]])+[cell['u'],cell['v']])
                    if np.isfinite(corners).all() and np.max(abs(corners))<100000:
                        pen.line([tuple(p) for p in corners],fill='#08fff1',width=2)
                pic.paste(layer,mask=Image.fromarray(domain))
            pic = pic.crop((x,y,x1,y1))
            pic.thumbnail((354,183),Image.Resampling.LANCZOS)
            out.paste(pic,(i*360+3,row*250+32))
            title = ('Source chart crop' if i==0 else
                     ('Previous' if i==1 else 'Proposed')+f" grid    seam advantage {rec['old' if i==1 else 'new']['score']:+.3f}")
            draw.text((i*360+4,row*250+220),title,font=font)
    out.save(REVIEW/'distant-grids.jpg',quality=92)


if __name__=='__main__':
    main()
