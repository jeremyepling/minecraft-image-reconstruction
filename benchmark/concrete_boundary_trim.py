"""Paired visible-only test of pixel trims along the exterior of segmented planes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from benchmark import concrete_shadow_boundaries as trial
from benchmark.pavement_opening_eval import visible_faces
from benchmark.texture_camera_eval import save
from engine.exterior_foreground import exterior
from engine.pillar_seams import native_mask


def trim_mask(rgb, mask, background, tight):
    # Only a shallow band between confirmed exterior and distinct inside color.
    # Opposite probes are on the same scanline/column. This guards visible holes
    # and avoids carving disconnected islands out of a face interior.
    h,w = mask.shape
    trimmed = np.zeros_like(mask)
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        def bounds(distance):
            oy,ox = distance*dy,distance*dx
            return np.s_[6+oy:h-6+oy,6+ox:w-6+ox]
        center,outer,inner = bounds(0),bounds(-3),bounds(5)
        valid = mask[center] & background[center] & ~mask[outer] & background[outer]
        valid &= mask[inner] & ~background[inner]
        if not valid.any():
            continue
        yy,xx = np.nonzero(valid)
        y,x = yy+6,xx+6
        color = rgb[y,x]
        out_delta = abs(color-rgb[y-3*dy,x-3*dx]).max(axis=1)
        in_delta = abs(color-rgb[y+5*dy,x+5*dx]).max(axis=1)
        good = (out_delta <= (2 if tight else 6)) & (in_delta >= (24 if tight else 10))
        trimmed[y[good],x[good]] = True
    return trimmed


def measure(job):
    split,src,_,_ = job
    key = src['id']
    image,parts,_,_ = trial.viewer.structure_stages(job,trial.baseline()[key])
    # Compute trim proposals using RGB and current supports before reading labels.
    rgb = np.asarray(image).astype(np.int16)
    background = exterior(image)
    proposals = {True:np.zeros(background.shape,bool),False:np.zeros(background.shape,bool)}
    for p in parts:
        mask = native_mask(p,image.size)
        if not (mask & background).any():
            continue
        ys,xs = np.nonzero(mask)
        box = np.s_[max(0,ys.min()-6):ys.max()+7,max(0,xs.min()-6):xs.max()+7]
        for tight,test in proposals.items():
            test[box] |= trim_mask(rgb[box],mask[box],background[box],tight)
    labels,rays = visible_faces(split,src)
    n = len(labels['faces'])+1
    directions = np.full(n,'',dtype='U5')
    for f in labels['faces']:
        directions[f['id']] = f['direction']
    row = dict(id=key,split=split)
    for tight,test in proposals.items():
        ids = rays[test]
        bg = ids==0
        samples = rgb[test]
        sky = (samples[:,2]-samples[:,0]>20) & (samples[:,2]>=samples[:,1])
        row['tight' if tight else 'loose'] = dict(
            background=int(bg.sum()),blue_background=int((bg & sky).sum()),visible=int((~bg).sum()),
            tops=int((directions[ids]=='up').sum()),undersides=int((directions[ids]=='down').sum()))
    if 'legacy-light_gray_concrete-' in key and any(token in key for token in trial.FOCUS):
        truth,_ = trial.label_maps(labels,rays)
        ys,xs = np.nonzero(truth)
        box = (max(0,int(xs.min())-14),max(0,int(ys.min())-14),
               min(image.width,int(xs.max())+15),min(image.height,int(ys.max())+15))
        mask = proposals[True]
        rgb8 = np.asarray(image).copy()
        rgb8[mask & (truth==0)] = [0,254,182]
        rgb8[mask & (truth>0)] = [255,32,99]
        color = Image.fromarray(rgb8).crop(box)
        panel = Image.new('RGB',(color.width, color.height+25),'#192730')
        panel.paste(color,(0,25))
        ImageDraw.Draw(panel).text((6,5),'Tight trim: green removes background; pink removes visible structure',fill='white')
        panel.save(trial.OUT/(key.replace('legacy-light_gray_concrete-','')+'-trim.png'))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    jobs = list(trial.suite.jobs([*trial.suite.framed.SIZES,'shelter']))
    trial.OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(measure,jobs))
    groups = {}
    for group,selected in (('fixed',True),('other',False)):
        rr = [r for r in rows if (r['split'] in trial.suite.framed.EVALUATION)==selected]
        groups[group] = {}
        for mode in ('loose','tight'):
            c = Counter()
            for r in rr:
                c.update(r[mode])
                c['views_with_visible_loss'] += int(r[mode]['visible']>0)
            groups[group][mode] = dict(c)
    save(trial.OUT/'trim-cases.json',rows)
    save(trial.OUT/'trim-summary.json',groups)


if __name__ == '__main__':
    main()
