"""Find a persistent shade edge using smooth local samples on either side of a line."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import concrete_shadow_boundaries as trial
from benchmark.concrete_shade_split import two_witnesses
from benchmark.normal_fields import cached_fields
from benchmark.pavement_opening_eval import visible_faces
from benchmark.texture_camera_eval import save
from benchmark.wall_texture_evaluate import metrics
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


def local_step(rgb, mask):
    ys,xs = np.nonzero(mask)
    if len(ys)<70:
        return
    y0,y1 = int(ys.min()),int(ys.max())+1
    x0,x1 = int(xs.min()),int(xs.max())+1
    own = mask[y0:y1,x0:x1]
    pixels = rgb[y0:y1,x0:x1]
    lum = pixels.mean(axis=2)
    if np.ptp(lum[own])<12:
        return
    best = None
    for vertical in (True,False):
        m,bright,colors = (own,lum,pixels) if vertical else (own.T,lum.T,pixels.transpose(1,0,2))
        if min(m.shape)<4:
            continue
        available = m[:,1:-2] & m[:,2:-1] & m[:,:-3] & m[:,3:]
        change = bright[:,2:-1] - bright[:,1:-2]
        left,right = colors[:,1:-2],colors[:,2:-1]
        stable = np.abs(left-right*((left.mean(2)+1)/(right.mean(2)+1))[:,:,None]).max(2)<=5
        stable &= (np.abs(bright[:,1:-2]-bright[:,:-3])<=6) & (np.abs(bright[:,2:-1]-bright[:,3:])<=6)
        ey,ex = np.nonzero(available & stable & (abs(change)>=12))
        if len(ey)<8:
            continue
        signs = np.sign(change[ey,ex])
        ex=ex+1
        for slope in (0.,-.125,.125,-.25,.25,-.5,.5,-1.,1.):
            intercept=np.rint(ex-ey*slope).astype(int)
            for sign in (-1,1):
                bins,hits=np.unique(intercept[signs==sign],return_counts=True)
                for b in bins[hits>=8]:
                    ry=np.arange(m.shape[0])
                    rx=np.rint(b+ry*slope).astype(int)
                    valid=(rx>=1)&(rx<m.shape[1]-2)
                    ry,rx=ry[valid],rx[valid]
                    inside=available[ry,rx-1]
                    if inside.sum()<8:
                        continue
                    score=inside & stable[ry,rx-1] & (sign*change[ry,rx-1]>=12)
                    if score.sum()<max(8,.75*inside.sum()):
                        continue
                    if np.ptp(ry[score])>1.6*score.sum():
                        continue
                    half=(np.arange(m.shape[1])[None,:] > b+np.arange(m.shape[0])[:,None]*slope+.5)
                    small=min(np.count_nonzero(m & half),np.count_nonzero(m & ~half))
                    if small<max(9,.045*m.sum()):
                        continue
                    rank=(int(score.sum()),float(np.median(abs(change[ry[score],rx[score]-1]))))
                    if best is None or rank>best[0]:
                        best=(rank,vertical,b+.5,slope)
    if best:
        _,vertical,b,slope=best
        def split(y,x):
            return x-x0>b+(y-y0)*slope if vertical else y-y0>b+(x-x0)*slope
        y,x=np.ogrid[:mask.shape[0],:mask.shape[1]]
        return split,mask & split(y,x)


def measure(job):
    split,src,_,_=job
    image,parts,_,_=trial.viewer.structure_stages(job,trial.baseline()[src['id']])
    native=[native_mask(p,image.size) for p in parts]
    masks=[m[1::3,1::3] for m in native]
    rgb=np.asarray(image)[1::3,1::3].astype(np.float32)
    field=cached_fields(src['files']['rgb.png'],Path('../normal-round/runs/normal-fields'),('normal',))
    normals=None
    if field:
        hm,wm=field['normal'].shape[:2]
        yy=np.minimum(hm-1,np.arange(1,image.height,3)*hm//image.height)
        xx=np.minimum(wm-1,np.arange(1,image.width,3)*wm//image.width)
        normals=field['normal'][np.ix_(yy,xx)]
    outputs={k:[] for k in ('local','witness')}
    events={k:[] for k in outputs}
    for i,p in enumerate(parts):
        cut=local_step(rgb,masks[i])
        witness=None
        new=[]
        if cut is not None:
            f,side=cut
            witness=two_witnesses(normals,masks,parts,i,(side,masks[i]&~side))
            y,x=np.ogrid[:image.height,:image.width]
            half=f((y-1)/3,(x-1)/3)
            for j,mask in enumerate((native[i]&half,native[i]&~half)):
                new.append(dict(id='split-'+p['id']+'-'+str(j),support=dict(size=list(image.size),runs=row_runs(mask))))
        for mode in outputs:
            if new and (mode=='local' or witness):
                outputs[mode].extend(new)
                events[mode].append(dict(id=p['id'],witness=witness))
            else:
                outputs[mode].append(p)
    labels,rays=visible_faces(split,src)
    truth,_=trial.label_maps(labels,rays)
    base=metrics(parts,truth,image.size)
    row=dict(id=src['id'],split=split,baseline=trial.surface_score(base))
    for mode,result in outputs.items():
        score=metrics(result,truth,image.size) if events[mode] else base
        row[mode]=dict(trial.surface_score(score),events=events[mode],
                       gained=sorted(set(score['matched'])-set(base['matched'])),
                       lost=sorted(set(base['matched'])-set(score['matched'])))
    if src['id'] in ('train-arcade2-cool-low-left','train-arcbridge3-earth-low-left'):
        trial.picture(image,parts,outputs['local'],truth,'local-step-'+src['id'],'Local shade split (cyan)')
    if events['witness'] and ('light_gray_concrete' in src['id'] or split=='train') and (row['witness']['gained'] or row['witness']['lost']):
        trial.picture(image,parts,outputs['witness'],truth,'local-step-'+src['id'],'Local shade and normal split (cyan)')
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('cohort',choices=('development','confirmation'))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    jobs=[j for j in trial.suite.jobs([*trial.suite.framed.SIZES,'shelter']) if (j[0]=='train')==(args.cohort=='development')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(measure,jobs))
    save(trial.OUT/('local-step-'+args.cohort+'.json'),rows)


if __name__=='__main__':
    main()
