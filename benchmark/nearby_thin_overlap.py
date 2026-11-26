"""Join independent candidate pools in the frozen local coordinate frame."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse

import numpy as np

from benchmark.nearby_face_trial import SPLITS, RUN, PUBLIC, cases, frozen, labels, load_prediction
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, read, save

THIN=RUN.parents[2]/'thin-face-atlas'


def paths(split,key):
    cohort='train-final' if split=='train' else 'extra' if split=='extra' else 'suite'
    return THIN/'runs'/cohort/split/key


def count(case):
    split,f,_,_,base=case
    key=f['id']
    data=frozen(case)
    _,_,camera,prior=load_prediction(case)
    T,truth,_=labels(case,camera['P'])
    depth={tuple(x) for x in data['points'][int(data['base_count']):]}
    path=paths(split,key)
    atlas_rows=read(path/'trace.json')
    atlas={tuple(r['cell']) for r in atlas_rows}
    assert not prior&(depth|atlas)
    new=depth|atlas
    if new:
        pts=np.array(sorted(new))
        xyz=np.floor((pts+.5)@T[:3,:3].T+T[:3,3]).astype(int)
        target={tuple(p) for p,q in zip(pts,xyz) if tuple(q) in truth}
    else:
        target=set()
    assert atlas&target=={tuple(r['cell']) for r in atlas_rows if r['target']}
    depth_pack=set(decoded_pack(RUN/'predictions'/split/key/'reconstruction.zip'))
    thin_pack=set(decoded_pack(path/'rank85.zip'))
    both=depth&atlas
    unselected=both-depth_pack-thin_pack
    return dict(id=key,split=split,rgb_sha256=base['rgb_sha256'],baseline_pack=base['pack_sha256'],
                union=len(new),depth=len(depth),atlas=len(atlas),common=len(both),
                correct_union=len(target),correct_depth=len(depth&target),correct_atlas=len(atlas&target),
                correct_common=len(both&target),common_not_selected=len(unselected),
                correct_common_not_selected=len(unselected&target),baseline_correct=base['relative_geometry']['correct'],
                baseline_extra=base['relative_geometry']['extras'],target=base['relative_geometry']['target'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    jobs=[j for s in ('train',*SPLITS) for j in cases(s)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(count,jobs))
    summary={}
    for s in ['fixed','extra','train']:
        rr=[r for r in rows if r['split'] in (SPLITS[:-1] if s=='fixed' else (s,))]
        m=Counter()
        for r in rr:
            m.update({k:v for k,v in r.items() if isinstance(v,int)})
        total=m['baseline_correct']+m['correct_union']
        m['add_only_f1']=2*total/(total+m['baseline_extra']+m['target'])
        m['also_remove_all_extras_f1']=2*total/(total+m['target'])
        summary[s]=dict(m)
        print(s,dict(m),flush=True)
    save(PUBLIC/'joint-candidate-headroom.json',dict(summary=summary,cases=rows))


if __name__=='__main__':
    main()

