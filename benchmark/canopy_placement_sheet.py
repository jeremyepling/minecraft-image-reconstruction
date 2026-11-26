"""Draw three preselected views of the visible roof placement comparison."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_metrics import truth_camera
from benchmark.texture_camera_eval import DATA, STABLE, checked, read
from benchmark.texture_camera_sheet import font
from benchmark.visible_depth_canopy_sheet import overlay
from benchmark.visible_depth_sheet import render
from benchmark.visible_depth_transfer import OUT, rgb_path, structure
from engine.geometry import project
from training.visible_labels import visible_labels


BASE = STABLE/'runs/visible-depth-rounds/shipped'
FRESH = STABLE/'runs/canopy-placement/confirmation'
GALLERIES = 'legacy-light_gray_concrete-holdout-galleries-plaster-low-left'


def native(inputs, key, original, current):
    frame = next(f for f in read(inputs/'screen.json')['selected'] if f['id']==key)
    rgb = checked(rgb_path(frame),frame['files']['rgb.png'])
    pose = read(checked(rgb.with_suffix('.json'),frame['files']['camera.json']))
    blocks = read(checked(rgb.with_name(frame['source_id']+'-blocks.json'),frame['files']['world-readback.json']))
    states = {tuple(b[k] for k in 'xyz'):b for b in blocks}
    labels, _, _ = visible_labels(pose,states,(0,0,0))
    return (rgb, pose, read(inputs/'upstream'/key/'surfaces.json')['planes'], labels,
            set(structure(blocks,64)), original/key/'cells.json', current/key/'cells.json')


def galleries(current, previous=BASE):
    root = DATA/'data/regression/frames'/GALLERIES
    return (root/'rgb.png', read(root/'camera.json'),
            read(STABLE/'runs/visible-depth-rounds/inputs'/(GALLERIES+'.json'))['parts'],
            read(root/'labels.json'), set(structure(read(root/'world-readback.json'),read(root/'frame.json')['ground_y'])),
            previous/'regression'/GALLERIES/'cells.json',current/'regression'/GALLERIES/'cells.json')


def build(out, regression, old, fresh, previous=None):
    samples = [
        ('Canopy 09 · right',native(OUT,'shape-benchmark-26.3--train-canopy-09-right',previous/'old' if previous else OUT/'results/production',old)),
        ('Canopy 00 · right',native(FRESH,'shape-benchmark-open-26.3--train-canopy-00-right',previous/'fresh' if previous else FRESH/'results/baseline',fresh)),
        ('Galleries · low left',galleries(regression,previous or BASE))]
    sheet = Image.new('RGB',(1265,1650),'#101921')
    pen = ImageDraw.Draw(sheet)
    pen.text((22,14),'Where did the visible roof blocks land?',font=font(32))
    pen.text((22,61),'Teal = correct position     Amber = extra     Magenta = extra at a hidden structure position',font=font(20))
    for idx,label in enumerate(['Photograph','Detected masks','Visible target','Previous build','New default']):
        pen.text((13,151+idx*288),label,font=font(16))
    for i,(name,(src,pose,parts,labels,occupied,old_file,new_file)) in enumerate(samples):
        before,after=[set(map(tuple,read(p)['world'])) for p in (old_file,new_file)]
        target={tuple(b['xyz_world']) for b in labels['blocks']}
        P=truth_camera(pose)
        corners=np.asarray(list(product((0,1),repeat=3)))
        xyz=(np.array(sorted(before | after | target))[:,None,:]+corners).reshape(-1,3)
        xy=project(P,xyz[xyz@P[2,:3]+P[2,3]>0])
        with Image.open(src) as im:
            image=im.convert('RGB')
        bounds=(*np.maximum(0,np.floor(xy.min(0)-10)).astype(int),
                *np.minimum(image.size,np.ceil(xy.max(0)+10)).astype(int))
        tiles=[image.crop(bounds),overlay(image,parts).crop(bounds)]
        for pts,reference in ((target,True),(before,False),(after,False)):
            tiles.append(render(pts,P,pose['eye_world'],bounds,target,occupied,reference))
        x=135+i*375
        pen.text((x,109),name,font=font(23))
        for j,tile in enumerate(tiles):
            tile.thumbnail((358,255),Image.Resampling.LANCZOS)
            y=150+j*288
            sheet.paste(tile,(x+(358-tile.width)//2,y+(255-tile.height)//2))
            if j>=3:
                pts=before if j==3 else after
                pen.text((x,y+255),f'{len(pts&target)} / {len(target)} visible    {len(pts-target)} extra',font=font(16))
    out.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(out,quality=91)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('out','regression','old','fresh'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--previous',type=Path,help='Root of one baseline for all three comparisons')
    a=p.parse_args()
    build(a.out,a.regression,a.old,a.fresh,a.previous)
