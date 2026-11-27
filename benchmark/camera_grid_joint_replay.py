"""Coupled and ablated placement replays on a bounded, source-selected case set."""
import os
os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from unittest.mock import patch

import numpy as np
from benchmark.camera_grid_joint import BASE,MOGE,RUN,entries,read,save,signature,mask


def matches(source, current):
    if signature(source)==signature(current):
        return True
    a,b = (mask(p['support'],(448,280)) for p in (source,current))
    union = (a|b).sum()
    return bool(union and (a&b).sum()/union>=.9)


def raw_grid_levels(raw_camera, joint_camera, overrides):
    raw = np.asarray(raw_camera['P'])
    joint = np.asarray(joint_camera['P'])
    if not np.allclose(raw[:, :3], joint[:, :3]):
        raise ValueError('Grid-only diagnostic requires common camera axes and intrinsics')
    def eye(P):
        return -np.linalg.solve(P[:, :3], P[:, 3])
    shift = np.rint(eye(raw)-eye(joint)).astype(int)
    shift[1] = 0  # Y=0 is the observed ground datum in both solves.
    return {k:[ax,level+int(shift[ax])] for k,(ax,level) in overrides.items()}, shift.tolist()


def build(task):
    from benchmark.camera_horizon_replay import replay
    from engine import atlas_plane_depth
    e,arm = task
    key=e['id']
    folder=RUN/key
    proposal=read(folder/'choices.json')['best'][int(arm=='joint2')]
    raw=read(MOGE/e['dataset']/key/'proposal.json')['camera']
    camera=raw if arm=='grid' else proposal['camera']
    overrides={} if arm=='camera' else proposal['overrides']
    shift=[0,0,0]
    if arm=='grid':
        overrides,shift=raw_grid_levels(raw,proposal['camera'],overrides)
    source_folder=BASE/e['dataset']/key
    source={p['id']:p for p in read(source_folder/'handoffs/placement-input.json.gz')['parts']}
    calls=[]
    original=atlas_plane_depth._place
    def place(image,version,parts,counts,passed_camera,atlas_path,**kwargs):
        linear=np.array(passed_camera['P'])[:,:3]
        from engine.surface_model import intrinsic
        focal=np.linalg.norm(linear[0]-image.width/2*linear[2])
        R=np.linalg.solve(intrinsic(image.size,focal),linear)
        selected={}
        for p in parts:
            k=p['id']
            if k not in overrides or not p.get('support') or p.get('normal_camera') is None:
                continue
            ax,level=overrides[k]
            if ax==np.argmax(abs(R.T @ p['normal_camera'])) and matches(source[k],p):
                selected[k]=level
        kwargs['height_overrides']=(kwargs.get('height_overrides') or {})|selected
        cells,status=original(image,version,parts,counts,passed_camera,atlas_path,**kwargs)
        actual={d['id']:[d['axis'],d['level']] for d in status.get('plane_offsets',[])}
        consumed={k:v for k,v in selected.items() if actual.get(k)==overrides[k]}
        if consumed!=selected:
            raise ValueError((consumed,selected,actual))
        calls.append(dict(horizontal_gauge_shift=shift,applied=consumed,unmatched=sorted(overrides.keys()-selected.keys())))
        return cells,status
    with patch.object(atlas_plane_depth,'_place',side_effect=place):
        case=replay(e,camera,source_folder,folder/arm)
    save(folder/(arm+'.json'),dict(case=case,calls=calls))
    return key,arm,calls,case['geometry']['strict']


def tasks():
    all_entries=list(entries())
    # One strongest source-ranked development and held-out case with a raw camera for ablations.
    selected=[]
    for development in (True,False):
        pool=[]
        for e in all_entries:
            if (e['split']=='train')!=development: continue
            p=read(RUN/e['id']/'choices.json')
            if p['best'] and read(MOGE/e['dataset']/e['id']/'proposal.json')['camera']:
                pool.append((p['best'][0]['rank'],e['id'],e))
        if pool: selected.append(min(pool)[2])
    for e in all_entries:
        if e in selected or read(RUN/e['id']/'choices.json')['accepted']:
            choices=read(RUN/e['id']/'choices.json')['best']
            for arm in ['joint1','camera']+(['joint2'] if len(choices)>1 else [])+(['grid'] if read(MOGE/e['dataset']/e['id']/'proposal.json')['camera'] else []):
                yield e,arm


if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['grid','camera','joint1','joint2'])
    selected=parser.parse_args().arm
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(build,[t for t in tasks() if selected is None or t[1]==selected]):
            print(row,flush=True)
