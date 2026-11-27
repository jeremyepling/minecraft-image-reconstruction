"""Summarize decoded, whole-build-registered placements from the expanded ruler ladder."""
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.expanded_wall_rulers import ARMS, NATIVE, OUT, cases
from benchmark.mask_blocks import decoded_pack
from benchmark.sky_crop_guard import pooled

REVIEW=ROOT/'benchmark/reviews/expanded-wall-rulers-01'
COMPARE=(('source','source_resample'),('source','source_material'),('source_resample','source_material'),
         ('native','counts'),('native','noop'),('noop','spacing'),('spacing','origin'))


def collect():
    rows=[]
    roots={'source':NATIVE.parent/'source','native':NATIVE,**{a:OUT/a for a in (*ARMS,'counts','source_resample','source_material')}}
    for case in cases():
        name=case['id']
        geometries={}
        truth={tuple(b['xyz_world']):b['block_id'] for b in read(Path(case['frame'])/'labels.json')['blocks']}
        arms={}
        for arm,root in roots.items():
            folder=root/'assembly/connected-atlas/sky'/name
            rec=read(folder/'score.json')
            assert rec['source_sha256']==case['rgb_sha256'] and rec['pack_sha256']==digest(folder/'reconstruction.zip')
            coords=read(folder/'placements.json')
            assert {tuple(p):m for p,m in coords['local']}==decoded_pack(folder/'reconstruction.zip')
            geo={tuple(p):m for p,m in coords['aligned']}
            geometries[arm]=geo
            s=rec['score']['fixed'] or rec['score']['relaxed_yaw']
            assert len(geo.keys()&truth.keys())==s['geometry']['strict']['correct']
            arms[arm]=dict(score=s,chart_sha256=rec['chart_sha256'],pack_sha256=rec['pack_sha256'],
                offered=rec['charts_seen'],kept=rec['kept_charts'],components=rec['components'])
        paired={}
        for a,b in COMPARE:
            prev=geometries[a].keys()
            now=geometries[b].keys()
            paired[a+'-'+b]=dict(gained=len((now-prev)&truth.keys()),lost=len((prev-now)&truth.keys()),
                extras_added=len((now-prev)-truth.keys()),extras_removed=len((prev-now)-truth.keys()))
        rows.append(dict(id=name,dataset=case['dataset'],layout=case['layout'],source_sha256=case['rgb_sha256'],
            reference_hashes=rec['reference_sha256'],arms=arms,paired=paired))
    cohorts={}
    for key in ('all',*sorted({r['dataset'] for r in rows}),*sorted({r['layout'] for r in rows})):
        cohort=[r for r in rows if key in ('all',r['dataset'],r['layout'])]
        arms={}
        for arm in roots:
            ss=[r['arms'][arm]['score'] for r in cohort]
            arms[arm]=pooled(ss)|dict(assigned=sum(s['scores']['assigned'] for s in ss),
                fallback=sum(s['scores']['fallback'] for s in ss))
        paired={}
        for a,b in COMPARE:
            label=a+'-'+b
            sums=Counter()
            for row in cohort:
                sums.update(row['paired'][label])
            for group in ('geometry','scores'):
                diffs=[r['arms'][b]['score'][group]['strict']['f1']-r['arms'][a]['score'][group]['strict']['f1'] for r in cohort]
                sums[group+'_improved']=sum(v>0 for v in diffs)
                sums[group+'_worse']=sum(v<0 for v in diffs)
                sums[group+'_tied']=sum(v==0 for v in diffs)
            paired[label]=sums
        cohorts[key]=dict(views=len(cohort),arms=arms,paired=paired)
    charts=[c for p in (OUT/'evidence').glob('*.json') for c in read(p)['charts']]
    count_records=[c for p in (OUT/'counts/evidence').glob('*.json') for c in read(p)['rows']]
    save(REVIEW/'handoffs.json.gz',{
        arm:[read(p) for p in sorted((OUT/directory).glob('*.json'))]
        for arm,directory in [('rulers','evidence'),('counts','counts/evidence'),('source','source_resample/evidence')]})
    score_file=REVIEW/'results.json.gz'
    save(score_file,dict(revision_base='d4f2e0959b6d7abe84000f05d139a9627b382b5f',
        cohorts=cohorts,cases=rows,measurements=dict(charts=len(charts),
        source_cells=sum(c['source_cells'] for c in charts),max_corner_error=max(c['max_corner_error'] for c in charts),
        atlas_sha256=read(next((OUT/'evidence').glob('*.json')))['atlas']['atlas_sha256'],
        median_absolute_relative_pitch_error=np.median(abs(1/np.array([c['pitch_factors'] for c in charts])-1),axis=0).tolist(),
        corrected_count_quads=len(count_records), previously_measured_both=sum(
            c['before'] is not None and len(c['before'])==2 and all(v is not None for v in c['before']) for c in count_records))))
    for key,cohort in cohorts.items():
        print(key,cohort['views'],{a:round(100*s['geometry_strict']['f1'],2) for a,s in cohort['arms'].items()})
    print('Pooled typed F1', {a:round(100*s['scores_strict']['f1'],2) for a,s in cohorts['all']['arms'].items()})
    print('Paired',cohorts['all']['paired'])


if __name__=='__main__':
    collect()
