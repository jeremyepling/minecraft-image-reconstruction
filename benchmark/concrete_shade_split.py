"""Split a smooth surface at a long straight shade step, testing independent normal evidence."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from benchmark import concrete_shadow_boundaries as trial
from benchmark.normal_fields import cached_fields
from benchmark.pavement_opening_eval import visible_faces
from benchmark.texture_camera_eval import save
from benchmark.wall_texture_evaluate import metrics
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


def axis(v):
    v = np.asarray(v,dtype=float)
    return v / max(1.e-8,np.linalg.norm(v))


def shade_cut(rgb, mask):
    # Fit two constant-intensity patches. Speckled textures, three-way junctions,
    # and checkerboards cannot vote for a split because the boundary must follow
    # one long line and have mostly uniform material colors on both sides.
    pixels = rgb[mask]
    if len(pixels)<80:
        return
    lum = np.rint(pixels.mean(axis=1)).astype(int)
    hist = np.bincount(lum,minlength=256)
    peak = int(hist.argmax())
    hist[max(0,peak-11):peak+12] = 0
    second = int(hist.argmax())
    lo,hi = sorted((peak,second))
    if hi-lo < 12 or hi-lo>96 or hist[second]<max(10,.08*len(pixels)):
        return
    if (np.minimum(abs(lum-lo),abs(lum-hi))<=4).mean()<.90:
        return
    bright = lum>(lo+hi)*.5
    if min(bright.mean(),1-bright.mean())<.08:
        return
    colors = [np.median(pixels[bright==side],axis=0) for side in [False,True]]
    if abs(colors[1]-colors[0]*((colors[1].mean()+1)/(colors[0].mean()+1))).max()>5:
        return
    side = np.zeros_like(mask)
    side[mask] = bright
    edges = []
    for a,b in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        y,x = np.nonzero(mask[a] & mask[b] & (side[a]!=side[b]))
        edges.extend(zip(x,y))
    if len(edges)<8:
        return
    xy = np.asarray(edges)
    val,vec = np.linalg.eigh(np.cov(xy.T))
    if val[1]<10 or val[0]>.035*val[1] or val[0]>2.0:
        return
    projection=(xy-xy.mean(axis=0))@vec[:,1]
    if len(np.unique(projection.round())) < .75*max(np.ptp(projection),1):
        return
    return (lo+hi)*.5, side


def two_witnesses(normals, masks, parts, i, sides):
    if normals is not None:
        vectors = [normals[s] for s in sides]
        med = [axis(np.median(v,axis=0)) for v in vectors]
        if all(np.mean(v@m>.93)>=.65 for v,m in zip(vectors,med)) and med[0]@med[1]<.82:
            return 'dense'
    witnesses=[]
    for side in sides:
        rim = np.asarray(Image.fromarray(side).filter(ImageFilter.MaxFilter(5))) & ~masks[i]
        votes = []
        for j,p in enumerate(parts):
            if j==i or not p.get('normal_camera'):
                continue
            area = int((masks[j]&rim).sum())
            if area>=max(6,.12*rim.sum()):
                votes.append((area,j))
        if not votes:
            return
        votes.sort(reverse=True)
        if len(votes)>1 and votes[1][0]>=.7*votes[0][0]:
            return
        witnesses.append(votes[0][1])
    if witnesses[0]!=witnesses[1] and axis(parts[witnesses[0]]['normal_camera'])@axis(parts[witnesses[1]]['normal_camera'])<.82:
        return 'adjacent'


def measure(job):
    split,src,_,_=job
    image,parts,_,_=trial.viewer.structure_stages(job,trial.baseline()[src['id']])
    native=[native_mask(p,image.size) for p in parts]
    masks=[m[1::3,1::3] for m in native]
    full=np.asarray(image)
    rgb=full[1::3,1::3].astype(float)
    field=cached_fields(src['files']['rgb.png'],Path('../normal-round/runs/normal-fields'),('normal',))
    normals=None
    if field:
        hm,wm=field['normal'].shape[:2]
        yy=np.minimum(hm-1,np.arange(1,image.height,3)*hm//image.height)
        xx=np.minimum(wm-1,np.arange(1,image.width,3)*wm//image.width)
        normals=field['normal'][np.ix_(yy,xx)]
    output={k:[] for k in ('shade','orientation')}
    logs={k:[] for k in output}
    for i,p in enumerate(parts):
        cut=shade_cut(rgb,masks[i])
        witness=None
        new=[]
        if cut is not None:
            t,side=cut
            witness=two_witnesses(normals,masks,parts,i,(side,masks[i]&~side))
            native_side=full.mean(axis=2)>t
            for n,mask in enumerate((native[i]&native_side,native[i]&~native_side)):
                new.append(dict(id='union-'+p['id']+'-side-'+str(n),support=dict(size=list(image.size),runs=row_runs(mask))))
        for mode in output:
            if cut is not None and (mode=='shade' or witness):
                output[mode].extend(new)
                logs[mode].append(dict(id=p['id'],normal_witness=witness))
            else:
                output[mode].append(p)
    labels,rays=visible_faces(split,src)
    truth,_=trial.label_maps(labels,rays)
    base=metrics(parts,truth,image.size)
    row=dict(id=src['id'],split=split,baseline=trial.surface_score(base),cached_normal=normals is not None)
    for mode,result in output.items():
        score=metrics(result,truth,image.size) if logs[mode] else base
        row[mode]=dict(trial.surface_score(score),events=logs[mode],
                       gained=sorted(set(score['matched'])-set(base['matched'])),
                       lost=sorted(set(base['matched'])-set(score['matched'])))
    if logs['orientation'] and (row['orientation']['gained'] or row['orientation']['lost']):
        trial.picture(image,parts,output['orientation'],truth,'shade-split-'+src['id'],'Shade and normal split (cyan)')
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('cohort',choices=('development','confirmation'))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    jobs=[j for j in trial.suite.jobs([*trial.suite.framed.SIZES,'shelter']) if (j[0]=='train')==(args.cohort=='development')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(measure,jobs))
    save(trial.OUT/('shade-split-'+args.cohort+'.json'),rows)


if __name__=='__main__':
    main()
