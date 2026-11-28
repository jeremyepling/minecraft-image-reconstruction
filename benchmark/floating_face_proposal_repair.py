"""Reference-only choice among frozen source quads for charts lacking an aligned seed."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_patch_combined as horizontal
from benchmark.expanded_baseline import read, save
from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.floating_bridges_report import summarize
from benchmark.floating_face_proposal_report import REVIEW, extra_root
from benchmark.floating_face_proposal_run import source_inputs, OUT, ROOT, GROUPS
from benchmark.floating_patch_report import baseline
from benchmark.relative_oracle_charts import fit_homography
from engine.surface_perimeter import _native_alpha

REPAIR=ROOT/'runs/face-proposal-repair'


def run(job):
    group,case,audit=job
    cv2.setNumThreads(1)
    source,ordinary,whole,_=source_inputs(group,extra_root(group))
    name=case['id']
    rel=Path('sky')/name/'charts.json'
    choices={r['id']:r['best_aligned'] for r in audit['charts'] if not r['old_aligned'] and r['best_aligned']}
    packet=read(whole/rel)
    parts={p['id']:p for p in read(source/'sky'/name/'source-handoff/surfaces.json')['planes']}
    image=Image.open(case['variants']['sky']['image']).convert('RGB')
    rgb=np.asarray(image)
    charts=[]
    for chart in packet['charts']:
        selected=choices.get(chart['id'])
        if selected:
            quad=np.array(selected['proposal_quad_px'])
            H=fit_homography([0,0],quad[[3,2,1,0]])
            alpha,(x,y,x1,y1)=_native_alpha(parts[chart['id']]['support'],image.size)
            mask=np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1]=np.asarray(alpha)>0
            points=np.column_stack(np.nonzero(mask)[::-1])+.5
            measured=resample(chart,H,mask,rgb,points)
            measured=preserve_and_rank(dict(chart,cells=[]),measured,rgb,mask)
            chart=dict(measured,confidence=chart['confidence'])
        charts.append(chart)
    root=REPAIR/group
    save(root/'charts'/rel,dict(packet,charts=charts,diagnostic='best native-aligned source quad when old seeds all miss'))
    with patch.object(horizontal,'inputs',lambda g:source_inputs(g,extra_root(g))):
        horizontal.run((group,case,root/'charts',root/'assembly'))
    return name,len(choices)


if __name__=='__main__':
    groups=(*GROUPS,'new6','new7')
    records={(r['group'],r['id']):r for r in read(REVIEW/'audit-alignment.json.gz')['cases']}
    jobs=[(g,c,records[g,c['id']]) for g in groups for c in source_inputs(g,extra_root(g))[3]]
    with ProcessPoolExecutor(7) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            print(f.result(),flush=True)
    report={}
    for g in groups:
        before=extra_root(g)/'horizontal-cells/assembly' if g in ('new6','new7') else baseline(g,True)
        report[g]=summarize(source_inputs(g,extra_root(g))[3],{'horizontal':before,'repair':REPAIR/g/'assembly'},'connected-atlas/sky')
    save(REVIEW/'reference-repair.json.gz',report)
    for g,data in report.items():
        for arm,s in data['cohorts']['all'].items():
            print(g,arm,[round(100*s[k]['f1'],2) for k in ('geometry_strict','geometry_facade','scores_strict','scores_facade')],
                  s['geometry_strict']['correct'],s['geometry_strict']['missed'],s['geometry_strict']['extras'],s['paired'])
