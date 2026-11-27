"""Draw the predicted wall mask and matched visible reference on two source images."""
from PIL import Image, ImageDraw

from benchmark import material_feedback_grid as trial
from benchmark.texture_camera_sheet import font
from engine.pillar_seams import native_mask
import numpy as np


EXAMPLES = (
    ('legacy-light_gray_concrete-dev-shopfront-masonry-high-right', 'mask-0-mobile-0-single-238-0'),
    ('legacy-light_gray_concrete-dev-shopfront-plaster-low-left', 'mask-5'),
)


def main():
    cases={r['id']:r for r in trial.read(trial.OUT/'confirmation-cases.json')}
    jobs={j[1]['id']:j for j in trial.typed.SUITE.jobs(['regression'])}
    canvas=Image.new('RGB',(1120,568),'#19272b')
    pen=ImageDraw.Draw(canvas)
    for n,(key,name) in enumerate(EXAMPLES):
        job=jobs[key]
        image,parts,_,_=trial.surfaces.viewer.structure_stages(job,trial.surfaces.baseline()[key])
        labels,rays=trial.surfaces.visible_faces(job[0],job[1])
        part=next(p for p in parts if p['id']==name)
        mask=native_mask(part,image.size)
        target=trial.pair(parts,image.size,labels,rays)[name]
        ref=np.isin(rays,[f['id'] for f in target])
        data=np.asarray(image).copy()
        for region,color in ((ref,(70,252,222)),(mask,(255,160,55))):
            # Mark native pixels just inside a boundary.
            inner=region.copy()
            for dx,dy in ((0,2),(0,-2),(2,0),(-2,0)):
                inner &= np.roll(region,(dy,dx),axis=(0,1))
            data[region & ~inner]=color
        marked=Image.fromarray(data)
        x=n*560
        pen.text((x+8,6),'Shopfront · '+('masonry, high right' if n==0 else 'plaster, low left'),
                 fill='white',font=font(16))
        canvas.paste(marked.resize((552,345)),(x+4,32))
        row=next(r for r in cases[key]['rows'] if r['surface']==name)
        def dim(vals):
            return ' x '.join('?' if v is None else str(v) for v in vals)
        ys,xs=np.nonzero(ref|mask); margin=12
        crop=marked.crop((max(0,xs.min()-margin),max(0,ys.min()-margin),
                          min(marked.width,xs.max()+margin),min(marked.height,ys.max()+margin)))
        crop.thumbnail((182,172))
        canvas.paste(crop,(x+8,386))
        lines=[f"Counter    {dim(row['current'])}",f"Feedback   {dim(row['proposal']['dims'])}",
               f"Reference  {dim(row['actual'])}",'Orange  ·  detected wall','Teal  ·  visible target']
        for i,line in enumerate(lines):
            pen.text((x+201,389+i*30),line,fill='white',font=font(15))
    canvas.save(trial.OUT/'shopfront-counts.jpg',quality=90)


if __name__=='__main__':
    main()

