"""Show the measured roof planes and source-angle geometry of the measured builds."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.roof_span_worst_sheet import face_map
from benchmark.texture_camera_eval import read,checked
from benchmark.texture_camera_sheet import font
from engine.surface_perimeter import draw_outline,with_outline
import numpy as np


def draw(runs,chosen,out):
    keys=['shape-benchmark-26.3--train-canopy-06-left','shape-benchmark-26.3--train-canopy-04-left',
          'shape-benchmark-26.3--dev-canopy-19-right']
    frames={f['id']:(s,f,h) for s,f,h in framed.cases(('old','fresh','extra')) if f['id'] in keys}
    data={name:{r['id']:r for r in read(checked(path/'cases.json',read(path/'summary.json')['cases_sha256']))}
          for name,path in runs}
    overlays=[name for name,_ in runs if name!='control']
    width=1356
    page=Image.new('RGB',(width,410+len(overlays)*230+2*255),'#11262f')
    pen=ImageDraw.Draw(page)
    def text(x,y,s,n=18): pen.text((x,y),s,fill='#eef6f6',font=font(n))
    def put(image,x,y,h=157):
        thumb=ImageOps.contain(image,(422,h),Image.Resampling.LANCZOS)
        page.paste(thumb,(x+(422-thumb.width)//2,y))
    text(22,17,'Can a single screenshot locate the roof height?',35)
    text(22,73,'Cyan outlines the selected surface. Heights count from the lowest building layer (0).',20)
    text(22,110,'In the geometry panels, green is correct, red is extra and blue is a missed visible block.',20)
    for j,key in enumerate(keys):
        x=18+448*j
        split,f,prior=frames[key]
        image,parts,_,camera=framed.photo(split,f,prior)
        path=framed.transfer.rgb_path(f)
        pose=read(checked(path.with_suffix('.json'),f['files']['camera.json']))
        T=registration(np.asarray(camera['P']),pose)
        truth,_,_,_=framed.references('old',f,framed.NATIVE_INPUTS[split],framed.TRAINING_DATA,framed.DEPTH,
                                      dict(old=framed.transfer.BUILDS))
        _,box=face_map(image,pose,truth,lambda p:(150,160,220))
        margin=10
        x0,y0,x1,y1=box
        crop=(max(0,x0-margin),max(0,y0-margin),min(image.width,x1+margin),min(image.height,y1+margin))
        course=int(framed.roof_specs()[key]['roof_y']-T[1,3])
        text(x,157,f['source_id'].split('canopy-')[-1].replace('-','  ').title()+'  |  Recorded course '+str(course),24)
        put(image,x,199,200)
        y=410
        for name in overlays:
            status=data[name][key]['status']
            primary=status.get('primary_height')
            shown=image.copy()
            if primary:
                part=next(p for p in parts if p['id']==primary['part'])
                draw_outline(shown,with_outline(part),'#42ffff',4)
            text(x,y,name,23)
            decision='Abstained' if primary is None else 'Selected course '+str(primary['block_y'])
            text(x,y+33,decision,18)
            put(shown.crop(crop),x,y+62)
            y+=230
        for name,caption in [('control','Previous build'),(chosen,'Selected build')]:
            path=dict(runs)[name]
            row=data[name][key]
            cells=registered(decoded_pack(checked(path/split/key/'reconstruction.zip',row['pack_sha256'])),T)
            aligned=framed.spacing.shift(cells,row['translation'])
            overlay,_=face_map(image,pose,truth|aligned,lambda p:
                               (58,204,143) if p in truth and p in aligned else (240,92,94) if p in aligned else (92,137,241))
            m=row['roof']['spacing']
            text(x,y,caption+f"   {row['f1']:.1%} F1",23)
            text(x,y+34,f"Roof against both posts: {m['anchored_roof_cells']} / {m['visible_roof_cells']}",18)
            put(overlay.crop(crop),x,y+68,174)
            y+=255
    out.parent.mkdir(parents=True,exist_ok=True)
    page.save(out,quality=92)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',nargs=2,action='append',required=True)
    p.add_argument('--chosen',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    draw([(n,Path(d)) for n,d in a.run],a.chosen,a.out)
