"""Measure each counter handoff on the same visible-reference subset as the geometry replays."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np

from benchmark import reference_surface_feedback as sim


def inspect(job):
    key=job[1]['id']
    im,parts,_,camera=sim.viewer.structure_stages(job,sim.releases()[1][key])
    labels,ids=sim.visible_faces(*job[:2])
    mapping=np.zeros(len(labels['faces'])+1,np.int32)
    groups=defaultdict(list)
    for f in labels['faces']:
        mapping[f['id']]=f['surface_id']
        groups[f['surface_id']].append(f)
    surface=mapping[ids]
    dims={s:sim.chart(fs,np.asarray(fs[0]['normal_world']),labels['camera'])[1] for s,fs in groups.items()}
    sizes=np.bincount(surface.ravel())
    possible=[]
    for p in parts:
        m=sim.native_mask(p,im.size)
        n=np.bincount(surface[m],minlength=len(sizes))
        iou=n/(sizes+m.sum()-n).clip(1)
        possible.extend((iou[s],p['id'],s) for s in np.flatnonzero(iou>=.5) if s)
    paired={}
    taken=set()
    for _,pid,s in sorted(possible,reverse=True):
        if pid not in paired and s not in taken:
            paired[pid]=int(s)
            taken.add(s)
    result={}
    for mode in ('baseline','mask','directions','chart'):
        if mode=='baseline':
            surfaces=sim.eave_pinholes.measurement_parts(sim.smooth_atlas_outline.measurement_parts(parts))
            mapping=paired
        else:
            surfaces,_,_=sim.oracle_parts(im,parts,labels,ids,mode)
            mapping={p['id']:int(p['id'].split('-')[1]) for p in surfaces}
        counts=sim.viewer.recount(im,surfaces,None)
        paired_rows=[r for r in counts['surfaces'] if r['id'] in mapping]
        def close(a,b):
            return a is not None and b is not None and np.allclose(a,b,rtol=0,atol=.01)
        by_id={p['id']:p for p in surfaces}
        result[mode]=dict(predicted=len(surfaces),reference=len(groups),paired=len(paired_rows),
                           width=sum(r['dimensions']['blocks'][0]==dims[mapping[r['id']]][0] for r in paired_rows),
                           height=sum(r['dimensions']['blocks'][1]==dims[mapping[r['id']]][1] for r in paired_rows),
                           exact=sum(r['dimensions']['blocks']==dims[mapping[r['id']]] for r in paired_rows),
                           count_available=sum(None not in r['counts'] for r in paired_rows),
                           chart_available=sum(r.get('quad') is not None for r in paired_rows),
                           chart_revised=sum(r.get('quad') is not None and by_id[r['id']].get('quad') is not None
                                             and not close(r.get('quad'),by_id[r['id']]['quad']) for r in paired_rows))
    return dict(id=key,split=job[0],counts=result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-split',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--out',type=Path,default=sim.ROOT/'count-handoffs.json')
    a=p.parse_args()
    groups=defaultdict(list)
    for j in sim.SUITE.jobs([*sim.SUITE.framed.SIZES,'shelter']):groups[j[0]].append(j)
    selected=[j for g in groups.values() for j in sorted(g,key=lambda j:j[1]['files']['rgb.png'])[:a.per_split]]
    with ProcessPoolExecutor(a.workers) as executor:
        rows=list(executor.map(inspect,selected))
    totals={m:{k:sum(row['counts'][m][k] for row in rows) for k in rows[0]['counts'][m]} for m in rows[0]['counts']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(totals=totals,rows=rows),indent=2)+'\n',encoding='utf-8')

