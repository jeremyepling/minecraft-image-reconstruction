"""Freeze per-aperture RGB-only candidates, then grade each possible recessed plane."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

import benchmark
from benchmark import framed_depth_eval as framed
from benchmark.texture_camera_eval import ROOT, digest, save
from engine.facade_inset_phase import host_grid, refine
from engine.pillar_seams import native_mask

try:
    from benchmark import facade_inset_round as common
except ImportError:
    benchmark.__path__.append(str(ROOT.parent/'facade-inset-round/benchmark'))
    from benchmark import facade_inset_round as common
    common.ORDINARY=Path(common.__file__).parent/'reviews/facade-inset-round-01/baseline-stages.zip'


REVIEW=Path('benchmark/reviews/facade-inset-phase-01')


def candidate_masks(image,parts,camera,observed):
    by_id={p['id']:p for p in parts}
    rows=[]
    hosts={}
    for obj in observed:
        parent=obj['parent']
        if parent not in hosts:
            hosts[parent]=host_grid(image,by_id[parent],camera)
        data=hosts[parent]
        lo,hi=np.asarray(obj['opening'])
        uv=data['uv']
        clip=((uv>=lo)&(uv<hi)).all(axis=1)
        x,y=data['x'],data['y']
        sign=np.sign(data['plane']-data['eye'][data['axis']])
        masks={}
        for dep in (1,2,3):
            back=uv+dep*sign*data['ratio']
            photo_pixels=clip&((back>=lo)&(back<hi)).all(axis=1)
            masks[str(dep)]=(y[photo_pixels],x[photo_pixels])
        rows.append((obj,masks))
    return rows


def check(job):
    split,frame,prior,control=job
    image,parts,_,camera=common.ordinary_inputs(split,frame,prior,control)
    observed=[]
    start=perf_counter()
    after,events=refine(image,parts,camera,candidate_log=observed)
    seconds=perf_counter()-start
    # All candidate decisions and all depth masks are fixed before opening source truth.
    options=candidate_masks(image,parts,camera,observed)
    record=dict(id=frame['id'],split=split,rgb_sha256=frame['files']['rgb.png'],events=events,
                inference_seconds=seconds,candidates=[])
    if not options:
        return record
    truth=common.visible_map(split,frame)
    areas=np.bincount(truth.ravel())
    before=common.surface_stats(parts,truth,image.size)
    by_id={p['id']:p for p in parts}
    for obj,masks in options:
        host_mask=native_mask(by_id[obj['parent']],image.size)
        counts=np.bincount(truth[host_mask],minlength=len(areas))
        host_reference=int(np.argmax(counts[1:])+1)
        alternatives={}
        for k,(y,x) in masks.items():
            hits=np.bincount(truth[y,x],minlength=len(areas))
            overlap=hits/(areas+len(x)-hits).clip(1)
            sid=int(np.argmax(overlap[1:])+1)
            alternatives[k]=dict(surface=sid,matched_before=sid in before['matched'],iou=float(overlap[sid]),
                pixels=len(x),purity=float(hits[sid]/max(1,len(x))),distinct=sid!=host_reference)
        record['candidates'].append(dict(image_only=obj,grading=dict(host=host_reference,depths=alternatives)))
    if events:
        current=common.surface_stats(after,truth,image.size)
        record['surface_change']=dict(gained=sorted(set(current['matched'])-set(before['matched'])),
            lost=sorted(set(before['matched'])-set(current['matched'])),
            unmatched_delta=current['unmatched_predictions']-before['unmatched_predictions'],
            paired_pixel_delta=current['paired_pixels']-before['paired_pixels'])
    return record


def run(args):
    REVIEW.mkdir(parents=True,exist_ok=True)
    work=list(common.jobs([*framed.SIZES,'shelter']))
    with ProcessPoolExecutor(args.workers) as pool:
        records=list(pool.map(check,work))
    save(REVIEW/'apertures.json',dict(baseline='c589f1a',candidate='9542984',frames=records))
    print([(r['id'],len(r['candidates']),len(r['events'])) for r in records if r['candidates']])
    print('frames',len(records),'sha256',digest(REVIEW/'apertures.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
