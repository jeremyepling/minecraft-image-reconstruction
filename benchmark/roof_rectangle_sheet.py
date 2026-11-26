"""Plot measured vertical edges and individual additions over three source photographs."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as framed
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.texture_camera_eval import checked,read
from benchmark.texture_camera_sheet import font
from engine import visible_roof_rectangle as rectangle
from engine.thin_atlas_faces import StripEvidence


VIEWS = [('shape-benchmark-open-26.3--train-canopy-00-right','Open Canopy 00'),
         ('validation-cloister3-cool-high-right','Cloister'),
         ('inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right','Colonnade 01')]


def project(P, vertices):
    q=np.asarray(vertices)@P[:,:3].T+P[:,3]
    return [tuple(v) for v in q[:,:2]/q[:,2,None]]


def score_lookup(path):
    return {r['id']:r for r in read(checked(path/'cases.json',read(path/'summary.json')['cases_sha256']))}


def draw(args):
    control=score_lookup(args.control)
    base=score_lookup(args.layout)
    winner=score_lookup(args.guarded)
    frames={f['id']:(s,f,h) for s,f,h in framed.cases(('validation','fresh'))}
    page=Image.new('RGB',(1470,1250),'#11232b')
    pen=ImageDraw.Draw(page)
    def text(x,y,t,size):
        pen.text((x,y),t,fill='#f2f5f7',font=font(size))
    def paste(im,x,y,max_h=240):
        thumb=ImageOps.contain(im,(454,max_h),Image.Resampling.LANCZOS)
        page.paste(thumb,(x+(454-thumb.width)//2,y+(max_h-thumb.height)//2))
    text(18,15,'Roof placement at a photographed corner',36)
    text(18,72,'Cyan and yellow mark perpendicular rims. Green marks a correct addition and red an extra.',20)
    for i,(key,title) in enumerate(VIEWS):
        s,f,h=frames[key]
        source,parts,_,camera=framed.photo(s,f,h)
        prior=set(decoded_pack(checked(args.control/s/key/'reconstruction.zip',control[key]['pack_sha256'])))
        ctx=StripEvidence(source,parts,camera,prior,rectangle._atlas('26.3'))
        upper={c for c in prior if 0<c[1]<ctx.eye[1]-1 and (c[0],c[1]+1,c[2]) not in prior}
        edges=rectangle.observed_edges(ctx,upper)
        rects,nrect=rectangle.rects(edges,ctx.eye,prior)
        union=Image.fromarray(ctx.union).getbbox()
        crop=(max(0,union[0]-18),max(0,union[1]-20),min(source.width,union[2]+18),min(source.height,union[3]+20))
        x=16+486*i
        text(x,120,title,25)
        paste(source,x,158,215)
        image=source.copy()
        for axis,color in [(0,'#20f3ff'),(2,'#ffbf3f')]:
            edge_pen=ImageDraw.Draw(image)
            for cx,cy,cz in edges[axis]:
                corner=[cx,cy+1,cz]
                corner[axis]+=int(ctx.eye[axis]>corner[axis]+1)
                other=corner.copy()
                other[2-axis]+=1
                edge_pen.line(project(ctx.P,[corner,other]),fill=color,width=3)
        text(x,386,'Measured roof edges',22)
        text(x,419,f'{nrect} corner(s), {len(rects)} missing candidate cells',17)
        paste(image.crop(crop),x,448)
        truth,_,T,_=framed.references(s,f,framed.NATIVE_INPUTS.get(s),framed.TRAINING_DATA,framed.DEPTH,
                                     dict(fresh=framed.transfer.BUILDS))
        for r,folder,y,label in [(base,args.layout,716,'Top pixels'),(winner,args.guarded,977,'Same top on both rims')]:
            row=r[key]
            cells=set(decoded_pack(checked(folder/s/key/'reconstruction.zip',row['pack_sha256'])))
            im=source.copy()
            draw=ImageDraw.Draw(im)
            for cx,cy,cz in cells-prior:
                world=next(iter(registered({(cx,cy,cz)},T)))
                aligned=tuple(a+b for a,b in zip(world,row['translation']))
                color='#00e591' if aligned in truth else '#ff334a'
                points=project(ctx.P,[(cx+a,cy+1,cz+b) for a,b in ((0,0),(1,0),(1,1),(0,1),(0,0))])
                draw.line(points,fill=color,width=3)
            m=row['relative_geometry']
            text(x,y,f"{label}   +{m['gained']} correct, +{m['extra_added']} extra",18)
            paste(im.crop(crop),x,y+30,214)
    page.save(args.out,quality=91)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('control','layout','guarded','out'):
        p.add_argument('--'+key,required=True,type=Path)
    draw(p.parse_args())
