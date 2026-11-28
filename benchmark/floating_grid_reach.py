"""Filter existing whole-wall cells by Chebyshev distance to original detected atlas cells."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import NATIVE, cases

OUT=ROOT/'runs/wall-reach'
FRESH=WORKTREES/'floating-fresh-validation/runs/floating-fresh'
SKY=WORKTREES/'floating-horizontal-bridges/runs/six-sky'
FIXED=WORKTREES/'floating-visible-builds/runs/floating-cloud-gate-01'
ARMS=('whole','radius2','radius4','sparse2','sparse4')


def limited(charts, original):
    before={c['id']:c for c in original}
    result={arm:[] for arm in ARMS}
    evidence=[]
    for chart in charts:
        seed=before[chart['id']]
        normal=seed.get('normal_camera')
        allowed=bool(seed['cells']) and normal is not None and abs(normal[1])<.67
        if not allowed or not chart['cells']:
            for arm in ARMS:
                result[arm].append(chart)
            continue
        assert np.array_equal(seed['H'],chart['H'])
        anchors=np.array(sorted({(c['u'],c['v']) for c in seed['cells']}),dtype=int)
        sparse=len(anchors)<=3 or np.linalg.matrix_rank(anchors-anchors[0])<2
        xy=np.array([(c['u'],c['v']) for c in chart['cells']],dtype=int)
        nearest=np.full(len(xy),np.iinfo(np.int64).max)
        for anchor in anchors:
            nearest=np.minimum(nearest,np.abs(xy-anchor).max(axis=1))
        sizes={}
        for arm,radius in zip(ARMS,(None,2,4,2 if sparse else None,4 if sparse else None)):
            cells=chart['cells'] if radius is None else [c for c,d in zip(chart['cells'],nearest) if d<=radius]
            result[arm].append(dict(chart,cells=cells))
            sizes[arm]=len(cells)
        evidence.append(dict(id=chart['id'],seed_cells=len(anchors),sparse=bool(sparse),
            retained_supported_seeds=int((nearest==0).sum()),counts=sizes))
    return result,evidence


def inputs(cohort):
    if cohort=='original':
        return NATIVE.parent/'source',NATIVE.parent/'source/atlas/connected',WORKTREES/'expanded-wall-rulers/runs/wall-rulers/source_material/charts',list(cases())
    if cohort=='sky':
        return FIXED,FIXED/'garden-anchor-extension/connected',SKY/'charts',read(assembly.SHARED/'fixtures.json')['cases']
    manifest='supplemental-manifest.json' if cohort=='supplemental' else 'manifest.json'
    return FRESH/'ordinary',FRESH/'ordinary/atlas/connected',FRESH/'source_material/charts',read(FRESH/manifest)['cases']


def run(job):
    cohort,case=job
    source,ordinary,whole,_=inputs(cohort)
    name=case['id']
    rel=Path('sky')/name/'charts.json'
    raw,packet=read(ordinary/rel),read(whole/rel)
    image=Path(case['variants']['sky']['image'])
    sha=digest(image)
    for item in (raw,packet):
        paths.check_handoff(item,source/'sky'/name/'source-handoff',sha)
    charts,ev=limited(packet['charts'],raw['charts'])
    root=OUT/cohort
    def checked_read(path):
        data=read(path)
        if case['dataset']=='visible-builds-v2' and path==Path(case['frame'])/'frame.json':
            audit=read(ROOT/'training/framing-audit-v2.json')
            proof=next(r for r in audit['frames'] if r['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            return dict(data,framing=proof)
        return data
    for arm in ('ordinary',*ARMS):
        folder=root/arm
        save(folder/'charts'/rel,raw if arm=='ordinary' else dict(packet,charts=charts[arm]))
        mode=(folder/'charts','charts.json',False,1,2.5,.5,False)
        with patch.object(paths,'RUN',source),patch.object(assembly,'ASSEMBLY',folder/'assembly'),patch.dict(assembly.MODES,{'connected-atlas':mode}),patch.object(expanded_scores,'read',checked_read):
            assembly.evaluate((case,'sky','connected-atlas'))
    loc=Path('connected-atlas/sky')/name/'placements.json'
    for arm,path in [('ordinary',SKY/'ordinary' if cohort=='sky' else source/'assembly'),('whole',SKY/'whole-wall' if cohort=='sky' else whole.parent/'assembly')]:
        assert read(root/arm/'assembly'/loc)==read(path/loc)
    save(root/'evidence'/(name+'.json'),dict(id=name,source_sha256=sha,source_chart_sha256=digest(ordinary/rel),whole_chart_sha256=digest(whole/rel),charts=ev))
    print(cohort,name,flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('cohorts',nargs='+',choices=('original','fresh','sky','supplemental'))
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    jobs=[(cohort,c) for cohort in args.cohorts for c in inputs(cohort)[3]]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))
