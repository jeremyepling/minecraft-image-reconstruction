"""Test a small, enclosed fragment with evidence on three sides of a larger host."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageFilter

from benchmark import concrete_shadow_boundaries as trial
from benchmark.pavement_opening_eval import visible_faces
from benchmark.texture_camera_eval import save
from benchmark.wall_texture_evaluate import metrics
from engine.pillar_seams import native_mask


def enclosed(host, child):
    h,w = child.shape
    area = child.sum()
    if area < 4 or area > .25*host.sum() or (child & host).sum() > .08*area:
        return False
    edge = child & ~np.asarray(Image.fromarray(child).filter(ImageFilter.MinFilter(3)))
    near = np.asarray(Image.fromarray(host).filter(ImageFilter.MaxFilter(5)))
    if (edge & near).sum() < .6*edge.sum():
        return False
    # Require independent evidence on three sides of the child, so a short
    # strip adjoining only the lip of a shelf cannot qualify.
    padded = np.pad(host,2)
    votes = 0
    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
        def offset(mask,k):
            y,x = 2+k*dy,2+k*dx
            return mask[y:y+h,x:x+w]
        side = child & ~offset(np.pad(child,2),1)
        contact = offset(padded,1) | offset(padded,2)
        votes += int((side & contact).sum() >= max(2,.4*side.sum()))
    return votes>=3


def adjacent_concrete(image, labels, rays):
    n = len(labels['faces'])+1
    names = sorted({b['block_id'] for b in labels['blocks']})
    material = {b['id']:names.index(b['block_id'])+1 for b in labels['blocks']}
    dirs = np.zeros(n,int)
    dirs_by_name = {d:i+1 for i,d in enumerate(('west','east','north','south','up','down'))}
    mats = np.zeros(n,int)
    surf,conc = trial.label_maps(labels,rays)
    for f in labels['faces']:
        mats[f['id']] = material[f['block']]
        dirs[f['id']] = dirs_by_name[f['direction']]
    step = 2
    rgb = np.asarray(image)[::step,::step].astype(np.float32)
    ids = rays[::step,::step]
    surf = surf[::step,::step]
    counts = {}
    for name in ('same_plane','different_planes','different_direction','parallel_separate'):
        counts[name] = Counter()
    for a,b in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        common = (ids[a]!=ids[b]) & conc[ids[a]] & (mats[ids[a]]==mats[ids[b]])
        for name,keep in [('same_plane',surf[a]==surf[b]),('different_planes',surf[a]!=surf[b]),
                          ('different_direction',dirs[ids[a]]!=dirs[ids[b]]),
                          ('parallel_separate',(surf[a]!=surf[b]) & (dirs[ids[a]]==dirs[ids[b]]))]:
            valid = common & keep
            ca,cb = rgb[a][valid],rgb[b][valid]
            lum = np.abs(ca.mean(1)-cb.mean(1))
            res = np.abs(cb - ca*((cb.mean(1)+1)/(ca.mean(1)+1))[:,None]).max(1)
            c = counts[name]
            c['pairs'] += int(valid.sum())
            c['step_ge_12'] += int((lum>=12).sum())
            c['stable_steps'] += int(((lum>=12) & (res<=4)).sum())
    return counts


def measure(job):
    split,src,_,_ = job
    key = src['id']
    image,parts,_,_ = trial.viewer.structure_stages(job,trial.baseline()[key])
    native = [native_mask(p,image.size) for p in parts]
    masks = [m[1::3,1::3] for m in native]
    rgb = np.asarray(image)[1::3,1::3].astype(np.float32)
    props = []
    for rank,i,j,g in trial.seam_features(rgb,masks,parts):
        if not g:
            continue
        order = (i,j) if masks[i].sum()>=masks[j].sum() else (j,i)
        if enclosed(masks[order[0]],masks[order[1]]):
            props.append((rank,i,j,g))
    revised,joins = trial.apply_pairs(parts,props,image.size,'guarded')
    labels,rays = visible_faces(split,src)
    truth,_ = trial.label_maps(labels,rays)
    current = metrics(parts,truth,image.size)
    after = metrics(revised,truth,image.size) if joins else current
    clean = Counter()
    for i,p in enumerate(current['parts']):
        sid = trial.dominance(p)
        if not sid:
            continue
        for j,q in enumerate(current['parts'][:i]):
            if sid != trial.dominance(q):
                continue
            mask = truth==sid
            a,b = (native[k] & mask for k in (i,j))
            frac = (a & b).sum()/min(a.sum(),b.sum())
            clean['same_dominant_pairs']+=1
            clean['overlap_le_10pct']+=int(frac<=.1)
    if joins:
        name = 'enclosed-'+key
        trial.picture(image,parts,revised,truth,name)
    audit = adjacent_concrete(image,labels,rays)
    return dict(id=key,split=split,before=trial.surface_score(current),after=trial.surface_score(after),
                joins=[[parts[i]['id'],parts[j]['id']] for i,j in joins],
                fragmentation=dict(clean),adjacent_concrete={key:audit[key] for key in ('same_plane','different_planes')})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cohort',choices=('development','confirmation'))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    jobs = [j for j in trial.suite.jobs([*trial.suite.framed.SIZES,'shelter']) if (j[0]=='train')==(args.cohort=='development')]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(measure,jobs))
    save(trial.OUT/('enclosed-'+args.cohort+'.json'),rows)


if __name__=='__main__':
    main()
