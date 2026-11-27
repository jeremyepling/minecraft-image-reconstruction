"""Zoom the two original Shopfront eaves and their surface evidence without changing resolution."""
import numpy as np
from PIL import Image, ImageDraw

from benchmark.eave_patch_audit import REVIEW, suite
from benchmark.texture_camera_sheet import font
from engine.eave_pinholes import refine
from engine.pillar_seams import native_mask


def marked(image, selected, tint):
    rgb=np.asarray(image).copy()
    rgb[selected] = .30*rgb[selected]+.70*np.asarray(tint)
    return Image.fromarray(rgb)


def render():
    jobs=[j for j in suite.jobs(suite.framed.EVALUATION) if j[1]['id'].endswith('dev-shopfront-plaster-low-left')]
    board=Image.new('RGB',(1110,1010),'#f5f2e9')
    pen=ImageDraw.Draw(board)
    pen.text((18,12),'The small rings inside Shopfront’s eave',font=font(33),fill='#203635')
    pen.text((18,62),'Both floor variants · Low-left source image · Blocks reconstructed from observed faces',font=font(20),fill='#364e4c')
    roi=(685,356,1045,420)
    for col,job in enumerate(jobs):
        x=18+col*548
        image,parts,_,_=suite.ordinary_inputs(*job)
        updated,events=refine(image,parts)
        host=events[0]['host']
        i=next(i for i,p in enumerate(parts) if p['id']==host)
        old,new=[native_mask(p,image.size) for p in (parts[i],updated[i])]
        truth=suite.visible_map(job[0],job[1])
        owner=np.bincount(truth[old]).argmax()
        source=image.copy()
        ImageDraw.Draw(source).rectangle(roi,outline='#ff9c23',width=5)
        source.thumbnail((525,330),Image.Resampling.LANCZOS)
        name='Light gray concrete' if 'light_gray' in job[1]['id'] else 'Smooth stone'
        pen.text((x,112),name,font=font(25),fill='#203635')
        board.paste(source,(x,148))
        rows=[('Source image crop',image),
              ('Previous underside mask in cyan',marked(image,old,(24,196,201))),
              ('Newly supported visible pixels in orange',marked(image,new&~old,(255,133,14))),
              ('Recorded underside in violet',marked(image,truth==owner,(162,111,222)))]
        for i,(label,panel) in enumerate(rows):
            y=490+i*127
            pen.text((x,y),label,font=font(18),fill='#203635')
            board.paste(panel.crop(roi).resize((522,93),Image.Resampling.NEAREST),(x,y+26))
    board.save(REVIEW/'eave-closeups.png',optimize=True)


if __name__=='__main__':
    render()
