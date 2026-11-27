"""Measure the skyline strip opportunity using the fixed control translation after prediction."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed, wall_section_eval
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked,read,save
from engine.visible_roof_skyline import proposals


CONTROL=Path('../wall-section-round/runs/control')


def evaluate(job):
    split,f,source,row=job
    image,parts,_,camera=framed.photo(split,f,source)
    pack=checked(CONTROL/split/f['id']/'reconstruction.zip',row['pack_sha256'])
    prior=set(decoded_pack(pack))
    ctx,pool,selected=proposals(image,parts,camera,prior)
    runs=ctx.rim_options() if ctx is not None else []
    # Opening geometry and recorded camera registration only after predictions.
    ref_split='old' if split in ('new','extra') else split
    builds=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
    truth,_,T,_=framed.references(ref_split,f,framed.NATIVE_INPUTS.get(split),framed.TRAINING_DATA,
                                  framed.DEPTH,dict(native=framed.native.BUILDS,old=builds,fresh=builds))
    positive={c for c in pool if tuple(np.floor((np.asarray(c)+.5)@T[:3,:3].T+T[:3,3]).astype(int)
                                         +row['translation']) in truth}
    clean=set()
    for opt in runs:
        supported=opt['cells'] & pool
        if supported<=positive:
            clean.update(supported)
    whole=ctx.finalize(clean,False) if clean else set()
    tiles=ctx.finalize(positive,False) if positive else set()
    return dict(id=f['id'],split=split,source_sha256=row['rgb_sha256'],control_pack_sha256=row['pack_sha256'],
                proposed=len(pool),correct=len(positive),clean_strips=len(whole),per_cell=len(tiles),
                roof_levels=len(ctx.levels) if ctx is not None else 0,sections=len(runs),selected=len(selected))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--splits',nargs='+',default=list(framed.SIZES))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    control=read(CONTROL/'summary.json')
    index={(r['split'],r['id']):r for r in read(checked(CONTROL/'cases.json',control['cases_sha256']))}
    jobs=[(s,f,h,index[s,f['id']]) for s,f,h in wall_section_eval.cases(args.splits)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows=list(executor.map(evaluate,jobs))
    save(args.out,dict(control='bec7c94',control_cases_sha256=control['cases_sha256'],cases=rows))
