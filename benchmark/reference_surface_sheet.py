"""Draw the ordinary and intervened projections on the original Portico source."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark import reference_surface_feedback as oracle
from benchmark.roof_span_worst_sheet import face_map
from benchmark.surface_sheet import text
from benchmark.texture_camera_eval import checked, read
from engine.pillar_seams import native_mask


def draw(folder,out):
    key='legacy-light_gray_concrete-holdout-portico-masonry-low-left'
    job=next(j for j in oracle.SUITE.jobs(['regression']) if j[1]['id']==key)
    split,frame=job[:2]
    src=oracle.typed.source_path(split,frame)
    pose=read(checked(src.with_name('camera.json'),frame['files']['camera.json']))
    labels,ids=oracle.visible_faces(split,frame)
    photo=Image.open(checked(src,frame['files']['rgb.png'])).convert('RGB')
    truth={tuple(b['xyz_world']) for b in labels['blocks']}
    _,parts,_,_=oracle.viewer.structure_stages(job,oracle.releases()[1][key])
    surface=np.zeros(len(labels['faces'])+1,np.int32)
    for f in labels['faces']:
        surface[f['id']]=f['surface_id']
    native=surface[ids]
    outlines=[]
    for name,masks in [('Predicted surface outlines',[native_mask(p,photo.size) for p in parts]),
                       ('Reference visible surfaces',[native==sid for sid in np.unique(native) if sid])]:
        pixels=np.array(photo)
        for mask in masks:
            a=Image.fromarray(mask.astype(np.uint8)*255)
            edge=np.asarray(a.filter(ImageFilter.MaxFilter(5))) != np.asarray(a.filter(ImageFilter.MinFilter(5)))
            pixels[edge]=(255,160,42) if name.startswith('Predicted') else (74,240,210)
        outlines.append((name,Image.fromarray(pixels),'From the same source image'))
    panels=outlines[:]
    for name,title in [('baseline','Ordinary replay'),('mask','Reference visible masks'),('chart','Reference charts + measured counts'),('grids','Reference charts + reference counts')]:
        row=read(folder/(key+'--'+name+'.json'))
        positions={tuple(p) for p, *_ in row['points']}
        image,_=face_map(photo,pose,truth|positions,lambda p:(45,192,127) if p in truth and p in positions else (55,130,255) if p in truth else (242,66,82))
        s=row['geometry']['strict']
        panels.append((title,image,f"{s['correct']} correct, {s['missed']} missed, {s['extras']} extra · {s['f1']:.2%} F1"))
    w,h=550,348
    sheet=Image.new('RGB',(w*3,2*(h+66)+92),(24,36,39))
    pen=ImageDraw.Draw(sheet)
    text(pen,15,12,'Portico • how one surface correction travels through the build',28,True,fill='white')
    text(pen,15,54,'Green = correct position     Blue = missing position     Red = extra position',20,fill='white')
    for i,(title,img,caption) in enumerate(panels):
        x,y=i%3*w, i//3*(h+66)+87
        text(pen,x+8,y,title,21,True,fill='white')
        sheet.paste(img.resize((w-8,h),Image.Resampling.LANCZOS),(x+4,y+28))
        text(pen,x+8,y+h+33,caption,18,fill='white')
    out.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(out,quality=90,optimize=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in',dest='folder',required=True,type=Path)
    p.add_argument('--out',type=Path,default=oracle.ROOT/'portico.jpg')
    a=p.parse_args()
    draw(a.folder,a.out)

