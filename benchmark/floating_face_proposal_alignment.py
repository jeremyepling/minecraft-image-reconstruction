"""Evaluate original and replacement rulers with the companion seed-audit grader."""
from concurrent.futures import ProcessPoolExecutor
from importlib import util
from pathlib import Path

import numpy as np

from benchmark.floating_face_proposal_report import REVIEW, extra_root
from benchmark.floating_face_proposal_run import GROUPS, OUT, ROOT, source_inputs
from benchmark.expanded_baseline import read, save, digest
from benchmark.relative_oracle_charts import fit_homography

AUDITOR = ROOT.parent/'floating-face-audit/benchmark/floating_face_audit.py'
spec = util.spec_from_file_location('frozen_face_auditor',AUDITOR)
audit = util.module_from_spec(spec)
spec.loader.exec_module(audit)


def run(job):
    g,case=job
    native=audit.NativeFaces(Path(case['frame']))
    name=case['id']
    _,ordinary,_,_=source_inputs(g,extra_root(g))
    seeds={c['id']:c for c in read(ordinary/'sky'/name/'charts.json')['charts']}
    packet=read(OUT/g/'offers'/(name+'.json.gz'))
    rows=[]
    for row in packet['charts']:
        chart=seeds[row['id']]
        old=[native.seed(chart,c,False) for c in chart['cells']]
        picked=row['picked']
        aligned=[]
        chosen=None
        for i,offer in enumerate(row['offers']):
            quad=np.array(offer['quad'])
            if i!=picked and native.overlap(quad,False)[0] < audit.THRESHOLDS['iou']-1e-6:
                continue
            H=fit_homography([0,0],quad)
            result=native.seed(dict(H=H,method=offer['method']),
                dict(u=0,v=0,block_id=offer['atlas'],confidence=offer['correlation']),False)
            if result['classification']=='aligned':
                aligned.append(result)
            if i==picked:
                chosen=result
        olds=[s for s in old if s['classification']=='aligned']
        rows.append(dict(id=row['id'],seeds=len(old),old_aligned=len(olds),
                         old_highest=max(old,key=lambda s:s.get('confidence') or 0.)['classification']=='aligned',
                         best_old=max(olds,key=lambda s:s['iou']) if olds else None,
                         offered=len(row['offers']),aligned_offers=len(aligned),chosen=chosen,
                         best_aligned=max(aligned,key=lambda s:s['iou']) if aligned else None))
    return dict(group=g,id=name,charts=rows)


def main():
    groups=(*GROUPS,'new6','new7') if (OUT/'new7').exists() else (*GROUPS,'new6')
    with ProcessPoolExecutor(6) as pool:
        cases=list(pool.map(run,[(g,c) for g in groups for c in source_inputs(g,extra_root(g))[3]]))
    save(REVIEW/'audit-alignment.json.gz',dict(grader_sha256=digest(AUDITOR),thresholds=audit.THRESHOLDS,cases=cases))
    rows=[r for case in cases for r in case['charts']]
    for cohort in ('all',*groups):
        use=rows if cohort=='all' else [r for c in cases if c['group']==cohort for r in c['charts']]
        pairs=[(bool(r['old_aligned']),r['chosen']['classification']=='aligned' if r['chosen'] else bool(r['old_aligned'])) for r in use]
        print(cohort,len(use),'old / best offered / source selected aligned / accepted / effective / gained / lost',
              sum(p[0] for p in pairs),sum(r['aligned_offers']>0 for r in use),
              sum(bool(r['chosen'] and r['chosen']['classification']=='aligned') for r in use),
              sum(bool(r['chosen']) for r in use),sum(p[1] for p in pairs),
              sum(b and not a for a,b in pairs),sum(a and not b for a,b in pairs))
    print(next(r['chosen'] for c in cases if c['group']=='sky' and 'chapel' in c['id'] for r in c['charts'] if r['id']=='mask-0'))


if __name__=='__main__':
    main()
