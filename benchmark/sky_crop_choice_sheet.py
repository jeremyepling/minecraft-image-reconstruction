"""Show the largest crop IoU gain and loss per held-out building at matched scale."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import read
from benchmark.sky_crop_decisions_report import ARCHIVE
from benchmark.sky_crop_options import ACTIVE, OUTPUT, SKY, native


def main():
    scores = read(ARCHIVE/'scores.json')
    learned = read(ARCHIVE/'learned.json.gz')['frames']
    truth = read(OUTPUT/'decisions/exact-target-masks.json')['frames']
    cases = [c for c in scores['cases'] if not c['development']]
    canvas = Image.new('RGB',(4*283,1005),'#18232b')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',15)
    title = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',17)
    draw.text((8,6),'Surface crops on the four held-out buildings',font=title,fill='white')
    draw.text((8,33),'Largest IoU gain and loss per building. Every row uses the same native crop bounds within its example.',font=font,fill='#d1dae2')
    for col, case in enumerate(cases):
        ident = case['id']
        image = Image.open(SKY/'images'/ident/'sky.png').convert('RGB')
        parts = {p['id']:p for p in read(ACTIVE/ident/'source-handoff/surfaces.json')['planes']}
        ranked = sorted(case['crops'],key=lambda r:r['learned']-r['current'])
        x = col * 283 + 8
        draw.text((x,61),ident.split('-01')[0].split('-1-warm')[0].partition('-')[2].title(),
                  font=title,fill='white')
        for group, record in enumerate([ranked[-1],ranked[0]]):
            off = 90 + group*455
            pid = record['id']
            current = parts[pid]['support']
            expected = truth[ident].get(pid, dict(size=image.size,runs=[]))
            masks = [native(s,image.size) for s in (current,learned[ident].get(pid,current),expected)]
            yy,xx = np.nonzero(np.logical_or.reduce(masks))
            box = (max(0,int(xx.min())-4),max(0,int(yy.min())-4),
                   min(image.width,int(xx.max())+5),min(image.height,int(yy.max())+5))
            draw.text((x,off),f"{['Gain','Loss'][group]}   {record['label']}   {record['learned_choice']}",
                      font=font,fill='#ffdb94')
            for row, (name,mask) in enumerate(zip(
                    [f"Current  {record['current']:.1%}",f"Chooser  {record['learned']:.1%}",'Recorded visible target'],masks)):
                y = off+26+row*139
                draw.text((x,y),name,font=font,fill='#dce4eb')
                photo = image.crop(box)
                alpha = Image.fromarray(mask).crop(box)
                cut = Image.composite(photo,Image.new('RGB',photo.size,'#344550'),alpha)
                scale = min(262/cut.width,108/cut.height)
                cut = cut.resize((max(1,round(cut.width*scale)),max(1,round(cut.height*scale))),
                                 Image.Resampling.NEAREST if scale>1 else Image.Resampling.LANCZOS)
                canvas.paste(cut,(x+(262-cut.width)//2,y+22))
    canvas.save(ARCHIVE/'held-out-crops.jpg',quality=91)


if __name__ == '__main__':
    main()
