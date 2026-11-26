"""Recheck photographed horizontal faces after removing an incorrectly placed occluder."""
from concurrent.futures import ProcessPoolExecutor
import argparse

import numpy as np
from benchmark import grounded_depth_round as ger
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import save
from engine import thin_atlas_faces, visible_horizontal_faces
from engine.camera_atlas_floor import _atlas


from engine.grounded_exposed_faces import newly_exposed


def additions(image, parts, camera, original, pruned, atlas, mode):
    if original==pruned:
        return set()
    if mode=='horizontal':
        post=visible_horizontal_faces.recover(image,parts,camera,pruned)
    elif mode=='atlas':
        post=thin_atlas_faces.recover(image,parts,camera,pruned,atlas,mode='rank75')
    else:
        raise ValueError(mode)
    return post-original


def trial(job):
    _,f,*_=job
    image,parts,camera,orig=ger.trial.load_prediction(job)
    pack=ger.ROOT/'runs/apply/train'/f['id']/'reconstruction.zip'
    pruned=set(decoded_pack(pack))
    atlas=_atlas('26.3',ger.ATLAS)
    result=dict(id=f['id'],family=f['family'],removed=len(orig-pruned))
    T,truth,occupied=ger.trial.labels(job,camera['P'])
    def counts(added):
        pts=np.array(sorted(added)).reshape(-1,3)+.5
        world=np.floor(pts@T[:3,:3].T+T[:3,3]).astype(int)+job[-1]['translation']
        cells=set(map(tuple,world))
        return [len(cells & truth.keys()),len(cells-truth.keys()),len((cells & occupied)-truth.keys())]
    if pruned==orig:
        return result
    a=thin_atlas_faces.recover(image,parts,camera,orig,atlas,mode='rank75')-orig
    result['third_atlas']=counts(a)
    for mode in ('horizontal','atlas'):
        proposed=additions(image,parts,camera,orig,pruned,atlas,mode)
        exposed=newly_exposed(image,parts,camera,orig,pruned,proposed)
        result[mode+'_all']=counts(proposed)
        result[mode+'_exposed']=counts(exposed)
        result[mode+'_exposed_novel']=counts(exposed-a)
    return result


def apply(image,parts,counts,camera,prior):
    from engine.grounded_exposed_faces import recover
    pruned,status=ger.apply(image,parts,counts,camera,prior)
    result=recover(image,parts,camera,prior,pruned,atlas_path=ger.ATLAS)
    return result,dict(status,restored=len(result-pruned))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(trial,ger.jobs('train')))
    groups={}
    for r in rows:
        for k,v in r.items():
            if not isinstance(v,list):continue
            groups.setdefault(k,np.zeros(3,int))
            groups[k]+=v
    print({k:v.tolist() for k,v in groups.items()})
    save(ger.REPORT/'exposed-train.json',dict(cases=rows,groups={k:v.tolist() for k,v in groups.items()}))
