"""Draw one recovered extra and one damaged real position on their full source images."""
from pathlib import Path

from benchmark import material_feedback_geometry as exp
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_examples():
    keys=['legacy-light_gray_concrete-holdout-portico-plaster-low-left',
          'legacy-light_gray_concrete-dev-shopfront-plaster-high-right']
    rows={r['id']:r for r in exp.read(exp.OUT/'cases.json')}
    canvas=Image.new('RGB',(1100,680),'#132127')
    pen=ImageDraw.Draw(canvas)
    def font(size):
        try:
            return ImageFont.truetype('segoeui.ttf',size)
        except OSError:
            return ImageFont.load_default(size=size)
    pen.text((15,10),'When does the front cube hide a better-fitting rear surface?',font=font(25),fill='white')
    pen.text((15,49),'Magenta = cube proposed for removal. Cyan = currently visible pixels of cubes just behind it.',font=font(17),fill='#ccded6')
    for i,key in enumerate(keys):
        inp,cells,_=exp.source_inputs(key,exp.CACHE)
        image=Image.open(inp['rgb']).convert('RGB')
        s=exp.Scene(image,[],inp['camera'],cells)
        choice=rows[key]['choice']
        n=s.cells.index(tuple(choice['cell']))
        owned=s.owned[n]
        rear=np.unique(s.back[owned][s.back[owned]>=0])
        rgb=np.array(image)
        for indexes,color in [(np.flatnonzero(np.isin(s.owner,rear)),[35,225,228]),(owned,[255,50,160])]:
            small=np.zeros(s.shape,np.uint8)
            small.ravel()[indexes]=1
            pixels=np.array(Image.fromarray(small).resize(image.size,Image.Resampling.NEAREST),bool)
            rgb[pixels]=.44*rgb[pixels]+.56*np.array(color)
        overlay=Image.fromarray(rgb)
        y,x=np.divmod(owned,s.shape[1])
        cx,cy=round(x.mean()*s.step),round(y.mean()*s.step)
        side=max(160,int(max(np.ptp(x),np.ptp(y))*s.step*4))
        left=min(max(cx-side//2,0),image.width-side)
        top=min(max(cy-side//2,0),image.height-side)
        box=(left,top,left+side,top+side)
        original=image.copy()
        ImageDraw.Draw(original).rectangle(box,outline='#ff329f',width=4)
        col=15+i*545
        title=('Portico: removed a real extra' if i==0 else 'Shopfront: removed a correct block')
        pen.text((col,85),title.replace(':',' ·'),font=font(20),fill='#f8ce8d')
        canvas.paste(original.resize((520,325),Image.Resampling.LANCZOS),(col,122))
        canvas.paste(overlay.crop(box).resize((195,195),Image.Resampling.NEAREST),(col,463))
        pen.multiline_text((col+207,463),f"Candidate local XYZ\n{tuple(choice['cell'])}\n\nFront RGB error {choice['front']:.1f}\nRear RGB error {choice['rear']:.1f}\n{choice['vote']:.0%} of samples favor rear",font=font(16),fill='white',spacing=4)
    canvas.save(exp.OUT/'portico-shopfront.jpg',quality=90,optimize=True)


if __name__=='__main__':
    draw_examples()
