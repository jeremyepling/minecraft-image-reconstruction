"""Render decoded coordinates in the original perspective for four fixed examples."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_metrics import truth_camera
from benchmark.texture_camera_eval import DATA,read,STABLE
from benchmark.texture_camera_sheet import font
from engine.geometry import project


EXAMPLES=[
    ('regression','legacy-light_gray_concrete-holdout-galleries-plaster-high-right','Galleries'),
    ('regression','legacy-light_gray_concrete-holdout-galleries-plaster-low-left','Galleries · low angle'),
    ('validation','validation-cloister1-cool-high-right','Cloister'),
    ('test','test-twinspire1-cool-high-right','Twin Spires')]
CORRECT=(69,200,163); EXTRA=(246,166,64); HIDDEN=(207,92,193); TARGET=(142,177,212)


def render(cells,P,eye,bounds,truth,occupied,reference=False):
    x0,y0,x1,y1=bounds
    scale=min(365/(x1-x0),295/(y1-y0))
    w,h=round(scale*(x1-x0)),round(scale*(y1-y0))
    out=Image.new('RGB',(w,h),'#17232d')
    pen=ImageDraw.Draw(out)
    # A cube's exposed axis faces have disjoint interiors at a common distance.
    for cell in sorted(cells,key=lambda a:-np.dot(P[2,:3],np.asarray(a)+.5)):
        color=TARGET if reference else CORRECT if cell in truth else HIDDEN if cell in occupied else EXTRA
        for axis in range(3):
            if cell[axis]<eye[axis]<cell[axis]+1:continue
            side=int(eye[axis]>=cell[axis]+1)
            neighbor=list(cell); neighbor[axis]+=1 if side else -1
            if tuple(neighbor) in cells:continue
            xyz=np.tile(cell,(4,1))
            xyz[:,axis]+=side
            xyz[:,[d for d in range(3) if d!=axis]]+=[[0,0],[1,0],[1,1],[0,1]]
            if np.any(xyz@P[2,:3]+P[2,3]<=0):continue
            xy=(project(P,xyz)-[x0,y0])*scale
            lit=[.82,1.,.70][axis]
            pen.polygon([tuple(p) for p in xy],fill=tuple(round(c*lit) for c in color),
                        outline=tuple(round(c*lit*.80) for c in color))
    return out


def build(out,baseline,candidate):
    width,height=1672,1780
    sheet=Image.new('RGB',(width,height),'#101921');pen=ImageDraw.Draw(sheet)
    pen.text((22,16),'Visible block placement at estimated surface depths',font=font(31))
    pen.text((22,62),'Teal = correct visible coordinate     Amber = extra     Magenta = extra at a hidden structure coordinate',font=font(20))
    labels=['Source','Surface masks','Visible target','Previous build','New default']
    for i,label in enumerate(labels):
        pen.text((16,156+319*i),label,font=font(16))
    for col,(split,key,title) in enumerate(EXAMPLES):
        frame=DATA/'data'/split/'frames'/key
        pose=read(frame/'camera.json');P=truth_camera(pose);eye=pose['eye_world']
        faces=read(frame/'reference-faces.json')['faces']
        target={tuple(b['xyz_world']) for b in read(frame/'labels.json')['blocks']}
        ground=read(frame/'frame.json')['ground_y']
        world={(b['x'],b['y'],b['z']) for b in read(frame/'world-readback.json')
               if b['y']>ground and b['block_id'].split(':')[-1]!='air'}
        old,new=[set(map(tuple,read(root/split/key/'cells.json')['world'])) for root in (baseline,candidate)]
        points=np.concatenate([f['image_corners'] for f in faces])
        for cells in (old,new):
            if cells:
                xyz=np.asarray(sorted(cells))+.5
                valid=xyz@P[2,:3]+P[2,3]>0
                points=np.vstack((points,project(P,xyz[valid])))
        w,h=pose['image_size']
        bounds=(*np.maximum([0,0],np.floor(points.min(0)-15)).astype(int),
                *np.minimum([w,h],np.ceil(points.max(0)+15)).astype(int))
        with Image.open(frame/'rgb.png') as opened: image=opened.convert('RGB')
        surfaces=image.copy()
        masks=read(STABLE/'runs/visible-depth-rounds/inputs'/(key+'.json'))['parts']
        for i,part in enumerate(masks):
            if not part.get('support'):continue
            s=part['support']; mask=Image.new('L',tuple(s['size'])); ink=ImageDraw.Draw(mask)
            for yy,x0,x1 in s['runs']:
                ink.line((x0,yy,x1-1,yy),fill=125)
            color=[(144,97,221),(36,200,207),(243,160,93),(100,163,239)][i%4]
            surfaces.paste(color,(0,0,image.width,image.height),mask.resize(image.size,Image.Resampling.NEAREST))
        tiles=[image.crop(bounds),surfaces.crop(bounds),
               render(target,P,eye,bounds,target,world,True),render(old,P,eye,bounds,target,world),
               render(new,P,eye,bounds,target,world)]
        left=127+col*385
        pen.text((left,107),title,font=font(22))
        for i,tile in enumerate(tiles):
            tile.thumbnail((365,295))
            sheet.paste(tile,(left+(365-tile.width)//2,150+i*319+(295-tile.height)//2))
            if i in (3,4):
                cells=old if i==3 else new
                pen.text((left,448+i*319),f'{len(cells&target)}/{len(target)} visible   {len(cells-target)} extra',font=font(17))
    out.parent.mkdir(parents=True,exist_ok=True);sheet.save(out,quality=91)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.out,a.baseline,a.candidate)
