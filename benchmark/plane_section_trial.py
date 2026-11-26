"""Fit section confidence rules on complete building photos, then replay the frozen rule."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import time

import numpy as np
from benchmark import nearby_face_trial as source
from benchmark.mask_blocks import _write_read, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import ROOT, STABLE, read, save, digest
from engine import section_relocation as model

BASE=ROOT.parent/'nearby-thin-geometry/runs/release'
RUN=ROOT/'runs/plane-section'
ATLAS=STABLE/'runs/26.3/atlas'
RULES={
    'strict':(.87,.20,5,.75),
    'consensus':(.78,.12,4,.60),
    'coverage':(.67,.08,3,.46),
    'no_removal':(.78,.12,4,2),
}


def jobs(split):
    base=BASE/split
    rows={r['id']:r for r in read(base/'cases.json')}
    return [(s,f,src,base,rows[f['id']]) for s,f,src,_,old in source.cases(split)
            if not f['id'].endswith('-close-front')]


def replay(job, mode):
    split,frame,src,base,old=job
    image,parts,camera,prior=source.load_prediction(job)
    key=frame['id']
    start=time.perf_counter()
    cache=RUN/'evidence'/split/(key+'.npz')
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            e=(z['points'],z['owner'],z['P'],z['eye'],z['rows'].tolist()) if 'points' in z else None
    else:
        e=model.sections(image,parts,camera,prior,atlas_path=ATLAS)
        data=dict(zip(('points','owner','P','eye','rows'),e)) if e is not None else {}
        np.savez_compressed(cache,**data)
    T,truth,occupied=source.labels(job,camera['P'])
    before=registered(prior,T)
    rows={}
    for name,rule in RULES.items():
        if mode!='all' and name!=mode:continue
        result=model.decide(image,prior,e,*rule)
        folder=RUN/name/split/key
        folder.mkdir(exist_ok=True,parents=True)
        pack=folder/'reconstruction.zip'
        local=_write_read(pack,result)
        world=registered(local,T)
        score=placement_scores(world,before,truth,occupied)
        rows[name]=dict(**score,pack_sha256=digest(pack),**{k:old[k] for k in old if k.endswith('_sha256') and k!='pack_sha256'})
        save(folder/'cells.json',dict(local=sorted(local),world=sorted(world)))
    return dict(id=key,split=split,family=frame.get('family'),seconds=round(time.perf_counter()-start,3),rules=rows)


def postprocess(image,parts,counts,camera,prior):
    return model.recover(image,parts,camera,prior,atlas_path=ATLAS)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--splits',nargs='+',default=['train'])
    ap.add_argument('--mode',default='all',choices=['all',*RULES])
    ap.add_argument('--workers',default=4,type=int)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--out',default='scores.json')
    args=ap.parse_args()
    work=[job for split in args.splits for job in jobs(split)[:args.limit]]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(partial(replay,mode=args.mode),work):
            rows.append(row)
            if len(rows)%10==0 or len(rows)==len(work):
                print(len(rows),row['id'],row['seconds'],flush=True)
                save(RUN/args.out,rows)
    for name in rows[0]['rules']:
        total=Counter()
        for row in rows:
            total.update(row['rules'][name]['relative_geometry'])
        print(name,dict(total))
