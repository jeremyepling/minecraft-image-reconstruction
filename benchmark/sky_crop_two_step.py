"""Reference-only test of delayed rewards for trimming or extending the current crop."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from engine.surface_perimeter import _native_alpha


REFERENCE = WORKTREES/'sky-crop-reference/runs/sky-crop-reference-01'
FROZEN = WORKTREES/'floating-visible-builds/runs/floating-cloud-gate-01/sky'
OPTIONS = {
    **{name:WORKTREES/'sky-crop-chooser/runs/sky-crop-chooser'/f'{name}.json'
       for name in ('sky-trim','colour-edge','depth-normal','dominant-piece','planarity')},
    **{name:WORKTREES/'sky-crop-models/runs/sky-crop-models'/f'{name}.json'
       for name in ('mobile-box','mobile-points','sam-b-box','sam-b-points','sam-b-zoom')}}


def native(support, size):
    mask=np.zeros(size[::-1],np.uint8)
    section=_native_alpha(support,size)
    if section:
        alpha,(x,y,x1,y1)=section
        mask[y:y1,x:x1]=np.asarray(alpha)>0
    return mask


def scores(current, target, options):
    actions=[('keep',current)]
    for name,other in options:
        actions.extend([('trim '+name,current&other),('extend '+name,current|other)])
    return [(name,mask,(mask&target).bit_count()/max(1,(mask|target).bit_count())) for name,mask in actions]


def case_scores(job):
    record, exact=job
    name=record['id']
    variants={k:read(p)['frames'][name] for k,p in OPTIONS.items()}
    path=FROZEN/name/'source-handoff/surfaces.json'
    assert digest(path)==record['surfaces_sha256']
    surface=read(path)
    size=tuple(surface['image_size'])
    data=[]
    for p in surface['planes']:
        sid=p['id']
        if sid not in exact:
            continue
        masks=[native(p['support'],size),native(exact[sid],size)]
        masks.extend(native(v.get(sid,p['support']),size) for v in variants.values())
        y,x=np.nonzero(np.logical_or.reduce(masks))
        bounds=np.s_[y.min():y.max()+1,x.min():x.max()+1]
        bits=[int.from_bytes(np.packbits(m[bounds]).tobytes(),'little') for m in masks]
        initial,target,*others=bits
        options=list(zip(variants,others))
        first=scores(initial,target,options)
        greedy=max(first,key=lambda item:item[2])
        best_next=max(scores(greedy[1],target,options),key=lambda item:item[2])
        best2=best_next[2]
        best_increasing=best2
        best_path=[greedy[0],best_next[0]]
        for action,mask,iou in first:
            after=max(scores(mask,target,options),key=lambda item:item[2])
            if after[2]>best2:
                best2=after[2]
                best_path=[action,after[0]]
            if iou>=first[0][2]:
                best_increasing=max(best_increasing,after[2])
        data.append(dict(surface=sid,initial=first[0][2],greedy_first=greedy[2],greedy_two=best_next[2],
                         optimal_two=best2,nondecreasing_first=best_increasing,optimal_actions=best_path))
    return dict(id=name,split=record['split'],surfaces=data)


if __name__=='__main__':
    report=read(REFERENCE/'report.json')
    targets=read(REFERENCE/'exact-target-masks.json')['frames']
    jobs=[(case,targets[case['id']]) for case in report['cases']]
    with ProcessPoolExecutor(3) as pool:
        frames=list(pool.map(case_scores,jobs))
    save(ROOT/'benchmark/reviews/sky-crop-downstream-01/two-edit-oracle.json.gz',
         dict(option_sha256={name:digest(p) for name,p in OPTIONS.items()},frames=frames))
    for split in ('all','held-out'):
        rows=[r for case in frames if split=='all' or case['split']!='train' for r in case['surfaces']]
        print(split,len(rows),{k:round(100*np.mean([r[k] for r in rows]),3) for k in
                             ('initial','greedy_first','greedy_two','optimal_two','nondecreasing_first')},
              'strict first-step loss required:',sum(r['optimal_two']>r['nondecreasing_first']+1e-9 for r in rows))
