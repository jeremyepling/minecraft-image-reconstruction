"""Offline donor-grid comparison on saved Java 26.3 photo-refined surfaces."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT,digest,read,revision,save
from benchmark.coplanar_count import SETTINGS,predict_frame
from benchmark.count_metrics import reference_planes
from benchmark.normal_round import groups
from benchmark.surface_reference import recorded
from benchmark.texture_count_round import FINDERS,summarize
from benchmark.wall_join_test import checked,read_gzip
from engine.geometry import UNIT,project
from engine.surface_dimensions import dimensions
from runtime.mode import image_only,mark_image_only

BASE=ROOT/'benchmark/reviews/texture-photo-count-01'
OUTLINES,METHOD=FINDERS['photo']
VARIANTS={'strict':SETTINGS,'loose':dict(SETTINGS,depth_log_tolerance=.25,recipient_grid_residual=.42,donor_grid_residual=.44,minimum_repeat=.55)}


def eval_frame(job):
    cap,row,hashes,variant=job
    with Image.open(checked(cap.path('image'),row['sha256'])) as img:
        photo=img.convert('RGB')
    pred=read_gzip(checked(BASE/row['predictions'],hashes[0]))
    parts=read_gzip(checked(OUTLINES/row['predictions'],hashes[1]))[METHOD]
    if [p['id'] for p in parts]!=[b['proposal'] for b in pred]:
        raise ValueError('Changed proposal order')
    # The same ordinary counts and image-derived charts can be replayed without resampling RGB.
    samples=[dict(quad=b['chart']) if b['chart'] is not None else None for b in pred]
    current=[b['methods']['current'] for b in pred]
    audit=[]
    start=perf_counter()
    results=predict_frame(photo,parts,samples,current,VARIANTS[variant],audit)
    elapsed=perf_counter()-start
    # Open scorer-only data after inference has finished.
    _,ref=recorded(cap)
    targets=reference_planes(ref,read(cap.path('labels'))['faces'])
    refs={r['proposal']:r['reference'] for r in row['surfaces'] if 'reference' in r}
    recs=[]
    for p,b,new,oldrec in zip(parts,pred,results,row['surfaces'],strict=True):
        old=dimensions(p,b['chart'],photo.size,b['methods']['current']['counts'])['blocks']
        if old!=oldrec['guess']['current']:
            raise ValueError('Baseline replay differs')
        guess=dimensions(p,new.get('quad',b['chart']),photo.size,new['counts'])['blocks']
        r=dict(proposal=p['id'],guess=dict(current=old,coplanar=guess))
        if p['id'] in refs:
            r.update(reference=refs[p['id']],actual=targets[refs[p['id']]]['size'])
        t=new['evidence'].get('transfer')
        if t is not None:
            r.update(transfer=t,quad=new['quad'],phase=new['phase'])
            rid,did=refs.get(p['id']),refs.get(t['donor'])
            if rid is not None and did is not None:
                a,bplane=(ref['patches'][i] for i in (rid,did))
                parallel=a['normal']==bplane['normal']
                coplanar=parallel and a['offset']==bplane['offset']
                mats=[sorted({f['block_id'] for f in targets[i]['faces'].values()}) for i in (did,rid)]
                r['transfer_truth']=dict(donor_reference=did,parallel=parallel,coplanar=coplanar,materials=mats)
                if coplanar:
                    # Pick the orientation of world axes corresponding to the predictor's donor axes.
                    grid=t['donor_grid_to_image']
                    basis=project(np.linalg.inv(ref['charts'][did]),project(np.asarray(grid),UNIT))
                    vecs=basis[[1,3]]-basis[0]
                    axes=np.argmax(abs(vecs),axis=1)
                    if axes[0]!=axes[1]:
                        origins=[]
                        for i in (did,rid):
                            cells=np.asarray(list(targets[i]['faces']))
                            origins.append(np.array([min(cells[:,k]) if v[k]>0 else -max(cells[:,k])-1 for v,k in zip(vecs,axes)]))
                        actual=(origins[1]-origins[0]).tolist()
                        r['transfer_truth'].update(actual_offset=actual,offset_correct=t['recipient_offset']==actual)
        recs.append(r)
    votes=Counter()
    for pair in audit:
        rid,did=(refs.get(pair[k]) for k in ('target','donor'))
        if rid is None or did is None:
            label='unpaired'
        else:
            a,b=(ref['patches'][i] for i in (rid,did))
            label='different_direction' if a['normal']!=b['normal'] else 'same_plane' if a['offset']==b['offset'] else 'parallel_other_depth'
        votes[label+('_pass' if pair['depth_pass'] else '_reject')]+=1
    return dict(key=row['key'],sha256=row['sha256'],cohort=row['cohort'],layout=row['layout'],
                eligible=row['eligible'],surfaces=recs,depth_gate=dict(votes),processing_s=elapsed)


@image_only()
def run(out,variant,split,workers):
    out.mkdir(exist_ok=False,parents=True)
    b,o=[read(folder/'manifest.json') for folder in (BASE,OUTLINES)]
    captures={c.floor+'/'+c.id:c for g in groups()[0].values() for c in g}
    cases=read(checked(BASE/'cases.json',b['artifacts']['cases.json']))
    rows=[r for r in cases if split=='all' or r['cohort']==split]
    jobs=[(captures[r['key']],r,[m['artifacts'][r['predictions']] for m in (b,o)],variant) for r in rows]
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        records=list(pool.map(eval_frame,jobs))
    scores={label:summarize([r for r in records if label in ('all',r['layout'],r['cohort'])],['current','coplanar'])
            for label in dict.fromkeys(['all',*[r['cohort'] for r in records],*[r['layout'] for r in records]])}
    save(out/'scores.json',scores)
    save(out/'cases.json',records)
    save(out/'manifest.json',dict(run_id=out.name,code=revision(),predictor_sha256=digest(ROOT/'benchmark/coplanar_count.py'),
        suite=b['fixtures'],reference_sha256=b['reference_sha256'],frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in rows],
        baseline_manifest_sha256=digest(BASE/'manifest.json'),outline_manifest_sha256=digest(OUTLINES/'manifest.json'),
        split=split,variant=variant,settings=VARIANTS[variant],version=b['version'],models='Inherited from the photo-refined Surface Finder',
        timing='Cached grid transfer per frame. No inference repeated.',artifacts={k:digest(out/k) for k in ('scores.json','cases.json')},
        reproduce=f'python -m benchmark.coplanar_count_round --variant {variant} --split {split} --out runs/my-coplanar'))
    print(scores['all'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--split',choices=['development','all'],default='all')
    parser.add_argument('--variant',choices=VARIANTS,default='strict')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    run(args.out,args.variant,args.split,args.workers)
