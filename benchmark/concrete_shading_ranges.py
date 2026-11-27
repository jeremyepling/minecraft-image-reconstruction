"""Measure broad light-to-dark variation inside a single recorded concrete block face."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from benchmark import concrete_shadow_boundaries as trial
from benchmark.pavement_opening_eval import visible_faces
from benchmark.enclosed_shadow_fragment import adjacent_concrete
from benchmark.texture_camera_eval import save


def measure(job):
    split,src,_,_=job
    image,_,_,_=trial.suite.ordinary_inputs(*job)
    labels,rays=visible_faces(split,src)
    _,conc=trial.label_maps(labels,rays)
    sampled=rays[1::3,1::3]
    keep=conc[sampled].copy()
    # Keep only a face's interior sample points, two native pixels away from
    # a change of face. This protects the range against mislabeled AA edges.
    h,w=rays.shape
    yy=np.arange(1,h,3)
    xx=np.arange(1,w,3)
    for oy,ox in ((2,0),(-2,0),(0,2),(0,-2)):
        keep &= sampled==rays[np.ix_(np.clip(yy+oy,0,h-1),np.clip(xx+ox,0,w-1))]
    rgb=np.asarray(image)[1::3,1::3].astype(np.float32)
    lum=rgb.mean(axis=2)
    examples=[]
    counts=Counter()
    for k in np.unique(sampled[keep]):
        pixels=lum[keep & (sampled==k)]
        if len(pixels)<24:
            continue
        q=np.percentile(pixels,[10,90])
        span=int(round(q[1]-q[0]))
        counts['eligible_faces']+=1
        counts['span_ge_12']+=int(span>=12)
        counts['span_ge_24']+=int(span>=24)
        if span>=12:
            examples.append((span,int(k),round(float(q[0]),1),round(float(q[1]),1)))
    ordered=sorted(examples,reverse=True)
    key=src['id']
    if 'legacy-light_gray_concrete-dev-shopfront-plaster-low-left'==key and ordered:
        span,k,lo,hi=ordered[0]
        ys,xs=np.nonzero(rays==k)
        box=(max(0,int(xs.min())-70),max(0,int(ys.min())-65),
             min(w,int(xs.max())+71),min(h,int(ys.max())+66))
        before=image.crop(box)
        tinted=np.asarray(image).copy()
        support=rays==k
        edge=support.copy()
        edge[1:-1,1:-1]&=~(support[1:-1,:-2]&support[1:-1,2:]&support[:-2,1:-1]&support[2:,1:-1])
        tinted[edge]=[40,255,220]
        after=Image.fromarray(tinted).crop(box)
        bw,bh=before.size
        sheet=Image.new('RGB',(max(bw*2,410),bh+50),'#192730')
        sheet.paste(before,(0,25))
        sheet.paste(after,(bw,25))
        pen=ImageDraw.Draw(sheet)
        pen.text((6,5),'Shopfront source image',fill='white')
        pen.text((bw+5,5),'One recorded concrete face',fill='white')
        pen.text((6,bh+32),f'10th to 90th percentile brightness inside this face: {lo} to {hi} on a 0-255 scale',fill='white')
        sheet.save(trial.OUT/'shopfront-concrete-light-range.png')
    return dict(id=key,split=split,counts=dict(counts),widest=ordered[:1],
                adjacent_same_material=adjacent_concrete(image,labels,rays))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    jobs=list(trial.suite.jobs([*trial.suite.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(measure,jobs))
    save(trial.OUT/'concrete-range.json',rows)


if __name__=='__main__':
    main()
