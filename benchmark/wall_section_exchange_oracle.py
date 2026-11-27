"""Assess matched depth replacements on occupied visible wall rays from the frozen control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked,read,save
from benchmark.wall_section_oracle import CONTROL,ATLAS
from engine import nearby_face_decision as nearby, wall_section_exchange as exchange


def score(job):
    split,f,source,row=job
    image,parts,counts,camera=framed.photo(split,f,source)
    pack=checked(CONTROL/split/f['id']/'reconstruction.zip',row['pack_sha256'])
    prior=set(decoded_pack(pack))
    data=nearby.observations(image,parts,camera,prior)
    after,status=exchange.choose(image,parts,camera,prior,data,atlas_path=ATLAS)
    ref_split='old' if split in ('new','extra') else split
    builds=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
    truth,occupied,T,_=framed.references(ref_split,f,framed.NATIVE_INPUTS.get(split),framed.TRAINING_DATA,
                                         framed.DEPTH,dict(native=framed.native.BUILDS,old=builds,fresh=builds))
    def world(c):
        return tuple(np.floor((np.asarray(c)+.5)@T[:3,:3].T+T[:3,3]).astype(int)+row['translation'])
    axes=0
    pairs=[]
    improved=prior
    if data is not None:
        valid=np.array([world(c) in truth for c in data['points']])
        s,t,a=data['edges'].T
        mask=(a!=1)&~valid[s]&valid[t]
        axes=int(mask.sum())
        pairs=exchange.matching(data['edges'][mask],data['features'][mask,2])
        improved=exchange.finalize(image,parts,camera,prior,data,pairs)
    def diff(result):
        added={world(c) for c in result-prior}
        removed={world(c) for c in prior-result}
        return dict(gained=len(added & truth),lost=len(removed & truth),extra_added=len(added-truth),
                    extra_removed=len(removed-truth),hidden_added=len((added-truth)&occupied),
                    hidden_removed=len((removed-truth)&occupied))
    oracle=diff(improved)
    assert oracle['gained']==oracle['extra_removed'] and oracle['lost']==oracle['extra_added']==0
    return dict(id=f['id'],split=split,hash=row['pack_sha256'],oracle=oracle,method=diff(after),
                edges=axes,pairs=len(pairs),status=status)


def run(args):
    rows=read(CONTROL/'cases.json')
    index={(r['split'],r['id']):r for r in rows}
    jobs=[(s,f,h,index[s,f['id']]) for s,f,h in framed.cases(args.splits)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result=list(pool.map(score,jobs))
    save(args.out,dict(control='028819a',rows=result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',nargs='+',default=list(framed.SIZES))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
