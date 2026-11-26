"""Verify local decision packs and measure the exact new-coordinate proposal ceiling."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from benchmark.nearby_face_trial import RUN, PUBLIC, SPLITS, cases, frozen, labels, load_prediction
from benchmark.texture_camera_eval import read, save, checked
from benchmark.mask_blocks import decoded_pack
from benchmark.visible_recovery_report import summarize


def check(job):
    case,folder=job
    split,f,_,_,prior_case=case
    key=f['id']
    base=frozen(case)
    _,_,camera,prior=load_prediction(case)
    T,truth,_=labels(case,camera['P'])
    p=base['points']
    if len(p):
        world=np.floor((p+.5)@T[:3,:3].T+T[:3,3]).astype(int)
        correct=np.array([tuple(x) in truth for x in world],bool)
    else:
        correct=np.zeros(0,bool)
    root=folder/split
    row=next(r for r in read(root/'cases.json') if r['id']==key)
    assert row['rgb_sha256']==prior_case['rgb_sha256']
    pack=decoded_pack(checked(root/key/'reconstruction.zip',row['pack_sha256']))
    saved=set(map(tuple,read(root/key/'cells.json')['local']))
    assert set(pack)==saved
    result=Counter()
    first=int(base['base_count'])
    result['candidate_count']=len(p)-first
    result['correct_candidates']=int(correct[first:].sum())
    unclaimed={tuple(pos):i for i,pos in enumerate(p)}
    edges=base['edges']
    switched=base['features'][:,4]>.5
    result['direction_change_candidates_correct']=int(sum(correct[i] for i in np.unique(edges[switched,1])))
    result['true_additions_with_direction_change_evidence']=sum(
        int(correct[unclaimed[pos]]) for pos in saved-prior if unclaimed[pos] in set(edges[switched,1]))
    result['changed_depth_candidates_correct']=int(sum(correct[i] for i in np.unique(edges[
        p[edges[:,1],edges[:,2]]!=p[edges[:,0],edges[:,2]],1])))
    return dict(result,id=key,split=split)


def report(folder,out):
    packs=RUN/folder
    rows={split:read(packs/split/'cases.json') for split in ('train',*SPLITS)}
    jobs=[(c,packs) for split in rows for c in cases(split)]
    with ProcessPoolExecutor(max_workers=5) as pool:
        oracle=list(pool.map(check,jobs))
    fixed=[r for s in SPLITS[:-1] for r in rows[s]]
    metrics=dict(fixed=summarize(fixed),cohorts={s:summarize(v) for s,v in rows.items()})
    for s,expected in [('fixed',112),('extra',20),('train',72)]:
        pool=[r for r in oracle if (r['split'] in SPLITS[:-1] if s=='fixed' else r['split']==s)]
        assert len(pool)==expected
        counts=Counter()
        for r in pool:
            counts.update({k:v for k,v in r.items() if k not in ('id','split')})
        metrics.setdefault('proposal_counts',{})[s]=dict(counts)
    save(PUBLIC/(out+'.json'),metrics)
    with ZipFile(PUBLIC/(out+'.zip'),'w',ZIP_DEFLATED) as z:
        for p in packs.rglob('*'):
            if p.is_file():
                z.write(p,p.relative_to(packs).as_posix())
    print(metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context',action='store_true')
    args=p.parse_args()
    report('context-predictions' if args.context else 'predictions','context-scores' if args.context else 'individual-scores')

