"""Run projected-atlas anchors, depth reassignment, and decoded placements on frozen controls."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import argparse
import numpy as np
from benchmark import full_block_lattice_run as release,view_atlas_lattice as experiment
from benchmark.expanded_baseline import ROOT,SOURCE,COHORTS,frames,read,save,digest
from benchmark import expanded_fixed_handoffs as fixed


RUN=ROOT/'runs/view-atlas-lattice-01'
REVIEW=ROOT/'benchmark/reviews/view-atlas-lattice-01'
OLD=ROOT.parent/'full-block-lattice/runs/full-block-lattice-01'


def all_frames():
    return list(fixed.fixed_frames())+list(frames(SOURCE,COHORTS))


def frame(entry):
    compact=release.compact_row
    with patch.object(release,'infer',experiment.infer),patch.object(release,'RUN',RUN),patch.object(release,'compact_row',
              lambda row:dict(compact(row),registered=row['registration']['geometry'],registered_scores=row['registration']['scores'])):
        result=release.frame(entry)
    return result


def summary():
    entries=all_frames()
    standalone_root=ROOT.parent/'view-conditioned-atlas/runs/view-conditioned-atlas'
    rows=[]
    for e in entries:
        folder=RUN/e['dataset']/e['id']
        if not (folder/'final/receipt.json').exists():
            continue
        raw=read(folder/'result.json')
        final=read(folder/'final/metrics.json')
        receipt=read(folder/'final/receipt.json')
        old=read(OLD/e['dataset']/e['id']/'result.json')
        r={k:raw[k] for k in ('dataset','id','rgb_sha256','split','layout','dataset_manifest_sha256','grader_sha256','evidence','rescued','changed')}
        r.update(final['metrics'])
        a=read(standalone_root/'cases'/e['dataset']/(e['id']+'.json'))
        assert a['source_sha256']==r['rgb_sha256'] and a['baseline_pack_sha256']==r['baseline']['pack_sha256']
        assert a['geometry']==r['baseline']['geometry'] and a['baseline']==r['baseline']['scores']
        assert digest(standalone_root/'packs'/e['dataset']/(e['id']+'.zip'))==a['pack_sha256']
        assert all(r['grid'][k]==old['source'][k] for k in ('geometry','scores','translation','pack_sha256'))
        r['standalone']={'geometry':a['geometry'],'scores':a['source'],'pack_sha256':a['pack_sha256']}
        r['source']=r['joint_view']
        r['rescue_only']=r['joint_view'] if raw['rescued'] else r['baseline']
        r['grid_rescue']=r['grid'] if old['rescued'] else r['baseline']
        r['oracle']=max([r['baseline'],r['grid'],r['joint_view']],key=lambda opt:opt['geometry']['facade']['f1'])
        r['against_grid']=final['against_grid']
        r['against_baseline']=final['against_baseline']
        r['policies']=receipt['policies']
        rows.append(r)
    out={}
    for ds,n in {release.FIXED:104,**COHORTS}.items():
        group=[r for r in rows if r['dataset']==ds]
        if not group:continue
        def pool(rs):
            arms={}
            for name in ('baseline','standalone','source','grid','grid_point','grid_view','joint_anchor','joint_point','grid_rescue','rescue_only','oracle'):
                fields=['geometry','scores']
                if 'registered' in rs[0][name]:
                    fields+=['registered','registered_scores']
                arms[name]={k:release.typed.pooled_scores([r[name][k] for r in rs]) for k in fields}
                if all('targets' in r[name] for r in rs):
                    partial=Counter()
                    for r in rs:partial.update(r[name]['targets'].get('partial_blocks',{}))
                    if partial:arms[name]['partial_blocks']=dict(partial)
            return arms
        stats=pool(group)
        for k in ('against_grid','against_baseline'):
            deltas=Counter()
            for r in group:deltas.update(r[k])
            stats[k]=dict(deltas)
        stats.update(tested=len(group),available=n,rescued=sum(r['rescued'] for r in group),changed=sum(r['changed'] for r in group),
                     splits={split:dict(tested=len(rs),**pool(rs)) for split in {r['split'] for r in group}
                            if (rs:=[r for r in group if r['split']==split])})
        out[ds]=stats
    save(REVIEW/'summary.json',out)
    save(REVIEW/'cases.json.gz',rows)
    save(REVIEW/'manifest.json',dict(parent_grid='1c6af5c',sampler_source='70bd4c7',final_matcher_source='83a8589',
         hashes={p.name:digest(p) for p in sorted(list((ROOT/'benchmark').glob('view_atlas_lattice*.py'))+
                  [ROOT/'benchmark/view_conditioned_atlas.py',ROOT/'benchmark/view_conditioned_atlas_match.py'])},
         atlas=experiment.grid.atlas().provenance,cases_sha256=digest(REVIEW/'cases.json.gz'),
         oracle='facade-first per-image chooser over ordinary, frozen grid, combined; keep ordinary on ties',
         frames=[{k:e[k] for k in ('dataset','id','rgb_sha256','dataset_manifest_sha256','split')} for e in entries]))
    print({k:(v['tested'],v['source']['geometry']['strict']['f1']) for k,v in out.items()})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--ids',nargs='*')
    p.add_argument('--cohorts',nargs='*')
    p.add_argument('--summary',action='store_true')
    a=p.parse_args()
    if a.summary:return summary()
    todo=[e for e in all_frames() if (not a.ids or e['id'] in a.ids) and (not a.cohorts or e['dataset'] in a.cohorts)
          and not (RUN/e['dataset']/e['id']/'result.json').exists()]
    if a.workers==1:
        for e in todo:print(frame(e),flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            for result in pool.map(frame,todo):print(result,flush=True)


if __name__=='__main__':
    main()
