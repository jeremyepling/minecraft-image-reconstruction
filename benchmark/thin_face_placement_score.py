"""Score frozen source-only gates, a competing depth exchange, and the proposal-pool oracle."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from benchmark.thin_face_placement_run import (suite,framed,current_inputs,ref_inputs,read,save,
    COHORTS,HERE,REPORT,registered,digest)
from benchmark.relative_geometry import placement_scores
from benchmark.roof_span_worst_stages import read_view
from benchmark.strip_visibility_audit import horizontal_slivers
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.pillar_seams import orientation
import numpy as np

METHODS=('all_pairs','balanced','texture_checked','exchange_checked','oracle')
FIELDS=('correct','missed','extras','hidden_extras','gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')


def gates(r):
    if r['existing']:
        return ()
    modes=['all_pairs']
    side=r.get('side_partial') or {}
    host=r.get('host_partial') or {}
    counted=r['count_axes']==2 or r['host_atlas']
    if not counted or r['shared'] < .50 or r['host_fraction']<.76 or r['side_visible']<.24:
        return modes
    if side.get('split',0)>=.65 and side.get('phase',0)>=.08 and host.get('split',0)>=.25:
        modes.append('balanced')
        if (side['split']>=.80 and host.get('phase',0)>=.08 and side['texels']>=8
                and host['color']<8 and side['color']<8):
            modes.extend(('texture_checked','exchange_checked'))
    return modes


def case(task):
    job,path=task
    split,f,prior,ctl=job
    sample=read(path)
    assert sample['rgb']==f['files']['rgb.png']
    rows=sample['rows']
    proposals={m:set() for m in METHODS}
    removals=set()
    pool={}
    for r in rows:
        c=tuple(r['cell'])
        for m in gates(r):
            proposals[m].add(c)
            conflict=r['conflict']
            if m=='exchange_checked' and conflict and conflict['rays']>=4 and conflict['fraction']>=.85 and conflict['wrong_axis']>=.5:
                removals.add(tuple(conflict['cell']))
        if not r['existing'] and r['reference']:
            proposals['oracle'].add(c)
        if not r['existing'] and r['conflict']:
            pool[(c,tuple(r['conflict']['cell']))]=(r['reference'],r['conflict']['reference'])
    groups={m:dict(sample['baseline'],**{f:0 for f in FIELDS[4:]},required_gain=0,required_loss=0) for m in METHODS}
    item=dict(id=f['id'],split=split,rgb=sample['rgb'],pack=sample['pack'],base=sample['baseline'],
        existing_pairs=sum(r['existing'] for r in rows),existing_blocks=len({tuple(r['cell']) for r in rows if r['existing']}),
        candidate_blocks=len(proposals['all_pairs']),host_complete_candidates=len({tuple(r['cell']) for r in rows if not r['existing'] and r['count_axes']==2}),
        exchange_pairs=len(pool),beneficial_true_swaps=sum(t and not old for t,old in pool.values()),methods=groups)
    if not proposals['all_pairs']:
        return item
    im,_,_,camera,base,_,_=current_inputs(job)
    truth,occupied,T,_=ref_inputs(job)
    before=registered(base,T)
    optional=set()
    if split in framed.EVALUATION:
        _,_,_,_,_,labels,ids,_,_,_,_=read_view(suite.CONTROL,split,f,prior,ctl)
        optional=horizontal_slivers(labels,ids)
    P,_=orientation(camera,im.size);eye=np.linalg.solve(P[:,:3],-P[:,3])
    for mode,added in proposals.items():
        if not added:continue
        removed=removals if mode=='exchange_checked' else set()
        retained=base-removed
        if mode in ('texture_checked','balanced','exchange_checked'):
            built=_keep_prior_visible(im.size,P,eye,retained,added)
            if not (built-retained):
                built=base
        else:
            built=retained|added
        world=registered(built,T)
        score=placement_scores(world,before,dict.fromkeys(truth),occupied)
        current={tuple(np.add(c,score['translation'])) for c in world}
        original={tuple(np.add(c,score['baseline_translation'])) for c in before}
        row={k:score['relative_geometry'][k] for k in FIELDS}
        row.update(required_gain=len(((current-original)&truth)-optional),
                   required_loss=len(((original-current)&truth)-optional),
                   added=sorted(built-base),removed=sorted(base-built))
        groups[mode]=row
    return item


def summarize(rows):
    result={}
    prio=read(HERE/'benchmark/reviews/facade-local-placement-01/priority.json')['methods']['clear_front']
    retired=read(HERE/'benchmark/reviews/facade-local-placement-01/summary.json')['historical_texture_controls']
    for cohort,splits in COHORTS.items():
        selected=[r for r in rows if r['split'] in splits]
        group={k:sum(r[k] for r in selected) for k in ('existing_pairs','existing_blocks','candidate_blocks','host_complete_candidates','exchange_pairs','beneficial_true_swaps')}
        group.update(views=len(selected))
        for method in METHODS:
            m=Counter({k:sum(r['methods'][method].get(k,0) for r in selected) for k in (*FIELDS,'required_gain','required_loss')})
            m['changed_views']=sum(any(r['methods'][method].get(k,0) for k in FIELDS[4:8]) for r in selected)
            m['f1']=2*m['correct']/(2*m['correct']+m['missed']+m['extras'])
            if cohort=='fixed':
                gain=m['required_gain']-m['required_loss']
                correct=prio['correct']+gain
                m['priority_f1']=2*correct/(2*correct+prio['missed']-gain+m['extras'])
                m['historical_112_f1']=2*m['correct']/(2*m['correct']+m['missed']+m['extras']+retired['visible_targets'])
            group[method]=dict(m)
        result[cohort]=group
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--in-dir',type=Path,default=HERE/'runs/thin-final')
    parser.add_argument('--report',type=Path,default=REPORT)
    args=parser.parse_args()
    jobs=list(suite.jobs([*framed.SIZES,'shelter']))
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(case,[(j,args.in_dir/(j[1]['id']+'.json')) for j in jobs]))
    args.report.mkdir(parents=True,exist_ok=True)
    save(args.report/'cases.json',rows)
    save(args.report/'summary.json',dict(base='abfc0be',cases_sha256=digest(args.report/'cases.json'),cohorts=summarize(rows)))
