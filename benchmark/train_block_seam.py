"""Fit and calibrate a block-seam classifier using only Visible Builds v1 training frames."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
from PIL import Image

from benchmark.camera_round import digest, read
from benchmark.learned_block_seam import (FEATURES, MODEL, best_new, best_phase, probabilities, profiles)
from engine.geometry import UNIT, bilinear, project, transform


TRAIN_INDEX = 'a45ba730e3af6448ccd402dd71e6d74aafab58b57efff42edec9f2747e8dc95c'
CALIBRATION_FAMILY = 'watchpost'


def charts(image, face_ids, faces, rng):
    """Training-only rectifications. Four RGB taps must be supported by that visible plane."""
    by_surface = defaultdict(list)
    id_surface = np.zeros(int(face_ids.max())+1,dtype=int)
    for face in faces:
        by_surface[face['surface_id']].append(face)
        id_surface[face['id']] = face['surface_id']
    mask_ids = id_surface[face_ids]
    for surface, group in by_surface.items():
        first = group[0]
        unit = np.array(first['corners_world'])
        basis = (unit[[1,3]]-unit[0]).T
        world = np.array([f['corners_world'] for f in group]).reshape(-1,3)
        coords = (world-unit[0])@basis
        # Fractional starts prevent either axis from learning a crop-border origin.
        lo = coords.min(0)+rng.uniform(0,.33,2)
        hi = coords.max(0)-rng.uniform(0,.33,2)
        span = hi-lo
        if max(span)<1.6:
            continue
        plane = transform(first['corners_px'])
        quad = project(plane,lo+UNIT*span)
        edges = np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1)
        dims = (edges[:2]+edges[2:])/2
        w,h = np.maximum(2,np.rint(dims*min(1.,640/max(dims))).astype(int))
        if min(w,h)<6:
            continue
        uv = np.stack(np.meshgrid((np.arange(w)+.5)/w,(np.arange(h)+.5)/h),axis=-1)
        xy = project(plane,lo+uv*span)
        x,y = np.floor(xy-.5).astype(int).transpose(2,0,1)
        valid = (x>=0)&(y>=0)&(x+1<image.shape[1])&(y+1<image.shape[0])
        x,y = x.clip(0,image.shape[1]-2),y.clip(0,image.shape[0]-2)
        for dy,dx in ((0,0),(1,0),(0,1),(1,1)):
            valid &= mask_ids[y+dy,x+dx]==surface
        rgb = np.round(bilinear(image,xy)).astype(np.uint8)
        for axis in (0,1):
            pitch = (w,h)[axis]/span[axis]
            if span[axis]<1.6 or pitch<6:
                continue
            profile = profiles(rgb if axis==0 else rgb.transpose(1,0,2),valid if axis==0 else valid.T)
            if profile is not None:
                yield profile,float(span[axis]),float(-lo[axis]%1.)


def frame_examples(job):
    directory,row = job
    path = directory/row['directory']
    for name in ('rgb.png','labels.json','labels.npz'):
        if digest(path/name)!=row['files'][name]:
            raise ValueError(f'Changed {path/name}')
    with Image.open(path/'rgb.png') as src:
        image = np.array(src.convert('RGB'))
    with np.load(path/'labels.npz',allow_pickle=False) as data:
        ids = data['face_id']
    faces = read(path/'labels.json')['faces']
    rng = np.random.default_rng(int(row['image_sha256'][:16],16))
    train_x, train_y, calibration = [],[],[]
    for profile,n,phase in charts(image,ids,faces,rng):
        features,valid = profile
        if row['family']==CALIBRATION_FAMILY:
            calibration.append((profile,n,phase))
            continue
        u = (np.arange(len(valid))+.5)/len(valid)
        distance = abs((u*n-phase+.5)%1.-.5)
        positives = np.flatnonzero(valid & (distance<=max(.045,.65*n/len(valid))))
        negatives = np.flatnonzero(valid & (distance>=.17))
        if not len(positives) or not len(negatives):
            continue
        positives = rng.choice(positives,min(45,len(positives)),replace=False)
        # Include strong internal decorative edges among the negative examples.
        hard = negatives[np.argsort(features[negatives,0])[-min(len(negatives),len(positives)):]]
        random = rng.choice(negatives,min(3*len(positives),len(negatives)),replace=False)
        neg = np.unique(np.r_[hard,random])
        train_x.append(features[np.r_[positives,neg]])
        train_y.append(np.r_[np.ones(len(positives)),np.zeros(len(neg))])
    return train_x,train_y,calibration


def fit(x,y):
    mean,scale = x.mean(axis=0),np.maximum(x.std(axis=0),.025)
    features = np.column_stack((np.ones(len(x)),(x-mean)/scale))
    weights = np.zeros(features.shape[1])
    penalty = np.diag([0.]+[.003]*(len(weights)-1))
    for _ in range(32):
        prob = 1/(1+np.exp(-np.clip(features@weights,-30,30)))
        curvature = (features.T*(prob*(1-prob)))@features/len(y) + penalty
        step = np.linalg.solve(curvature,features.T@(prob-y)/len(y)+penalty@weights)
        weights -= step
        if abs(step).max()<1e-7:
            break
    return dict(mean=mean.tolist(),scale=scale.tolist(),coef=weights[1:].tolist(),intercept=float(weights[0]))


def threshold(trials):
    # Frozen calibration utility: a wrong axis costs four accurate axes. Prefer fewer guesses on ties.
    options = []
    for level in np.arange(.02,.52,.02):
        accepted = [ok for score,ok in trials if score>=level]
        correct = sum(accepted)
        options.append((correct-4*(len(accepted)-correct),-len(accepted),-float(level),correct))
    value,negative_count,negative_level,correct = max(options+[(0,0,-1.,0)])
    return -negative_level,dict(axes=len(trials),accepted=-negative_count,correct=int(correct),utility=int(value))


def train(dataset,out,workers):
    index = dataset/'frames.jsonl'
    if digest(index)!=TRAIN_INDEX:
        raise ValueError('Expected the immutable 72-frame training split index')
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if len(rows)!=72 or {r['split'] for r in rows}!={'train'}:
        raise ValueError('Unexpected training split')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        samples = list(pool.map(frame_examples,[(dataset,r) for r in rows]))
    x = np.concatenate([x for xx,_,_ in samples for x in xx])
    y = np.concatenate([y for _,yy,_ in samples for y in yy])
    weights = fit(x,y)
    fixed_trials, free_trials = [],[]
    for _,_,calibration in samples:
        for profile,n,truth_phase in calibration:
            prob,valid = probabilities(profile,weights)
            peak = best_phase(prob,valid,n)
            if peak:
                score,phase = peak
                delta = abs((phase-truth_phase+.5)%1.-.5)
                fixed_trials.append((score,delta<.15))
            proposal = best_new(prob,valid,None)
            if proposal:
                score,estimate,phase = proposal
                # A single integer grid shift is arbitrary. Drift over the full chart is measurable.
                left = (phase-truth_phase+.5)%1.-.5
                right = left + n-estimate
                free_trials.append((score,max(abs(left),abs(right))<.20))
    weights['phase_threshold'], fixed = threshold(fixed_trials)
    weights['pitch_threshold'], free = threshold(free_trials)
    weights.update(features=FEATURES,version='26.3',source_index_sha256=TRAIN_INDEX,
                   examples=int(len(y)),positive_columns=int(y.sum()),calibration_family=CALIBRATION_FAMILY,
                   calibration=dict(known_pitch=fixed,new_pitch=free),
                   frames=[dict(id=r['id'],family=r['family'],files={k:r['files'][k] for k in
                         ('rgb.png','labels.json','labels.npz')}) for r in rows])
    out.write_text(json.dumps(weights,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(f'Training columns {len(y)} with {int(y.sum())} seam positives')
    print(f'Known pitch threshold {weights["phase_threshold"]} {fixed}')
    print(f'New pitch threshold {weights["pitch_threshold"]} {free}')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_directory',type=Path)
    parser.add_argument('--out',type=Path,default=MODEL)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    train(args.train_directory,args.out,args.workers)
