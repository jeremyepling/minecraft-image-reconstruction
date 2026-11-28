"""Build patch-filtered wall packs and their horizontal-cell combinations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark import relative_face_assembly as solver
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_wall_rulers import NATIVE, cases
from benchmark.floating_horizontal_bridges import horizontal_charts
from benchmark.floating_patch_measure import partition, rechart
from engine.surface_perimeter import _native_alpha

OUT = ROOT/'runs/patch-lattice'
BRIDGES = WORKTREES/'floating-horizontal-bridges/runs'
FRESH = WORKTREES/'floating-fresh-validation/runs/floating-fresh'
NEW = WORKTREES/'floating-patch-validation/runs/floating-patch-validation'


def inputs(cohort):
    if cohort == 'original':
        source = NATIVE.parent/'source'
        whole = WORKTREES/'expanded-wall-rulers/runs/wall-rulers/source_material'
        return source,source/'atlas/connected',whole/'charts',whole/'assembly',BRIDGES/'horizontal-cells/assembly',list(cases())
    if cohort == 'sky':
        source = WORKTREES/'floating-visible-builds/runs/floating-cloud-gate-01'
        sky = BRIDGES/'six-sky'
        return source,source/'garden-anchor-extension/connected',sky/'charts',sky/'whole-wall',sky/'horizontal-cells',read(assembly.SHARED/'fixtures.json')['cases']
    run = NEW if cohort == 'new' else FRESH
    manifest = 'supplemental-manifest.json' if cohort == 'supplemental' else 'manifest.json'
    source = run/'ordinary'
    whole = run/'source_material'
    horiz = run if cohort == 'new' else BRIDGES
    return source,source/'atlas/connected',whole/'charts',whole/'assembly',horiz/'horizontal-cells/assembly',read(run/manifest)['cases']


def run(job):
    cohort,case = job
    cv2.setNumThreads(1)
    source,original,whole,_,_,_ = inputs(cohort)
    name = case['id']
    rel = Path('sky')/name/'charts.json'
    image = Path(case['variants']['sky']['image'])
    rgb = np.asarray(Image.open(image).convert('RGB'))
    size = rgb.shape[1::-1]
    src = source/'sky'/name/'source-handoff'
    planes = read(src/'surfaces.json')['planes']
    plane_map = {p['id']:p for p in planes}
    seed,packet = read(original/rel), read(whole/rel)
    sha = digest(image)
    for p in (seed,packet):
        paths.check_handoff(p,src,sha)
    anchors = {c['id']:c for c in seed['charts']}
    filtered, added, evidence = [], [], []
    for chart in packet['charts']:
        mask = np.zeros(rgb.shape[:2], 'uint8')
        record = _native_alpha(plane_map[chart['id']]['support'],size)
        if record:
            alpha,(x,y,x1,y1) = record
            mask[y:y1,x:x1] = np.asarray(alpha)>0
        kept, obs = partition(rgb,mask,chart,anchors[chart['id']])
        replacements = rechart(rgb,mask,chart,obs)
        filtered.append(kept)
        added.extend(replacements)
        evidence.append(dict(id=chart['id'],before=len(chart['cells']),after=len(kept['cells']), regions=obs,
                             new_charts=[dict(id=r['id'], cells=len(r['cells'])) for r in replacements]))
    root = OUT/cohort
    save(root/'evidence'/(name+'.json.gz'),dict(id=name,source_image=str(image),seed_chart=str(original/rel),
         whole_chart=str(whole/rel),source_sha256=sha,seed_sha256=digest(original/rel),whole_sha256=digest(whole/rel),
         mask=str(src/'surfaces.json'),mask_sha256=digest(src/'surfaces.json'), charts=evidence))
    def checked(path):
        value = read(path)
        if case['dataset']=='visible-builds-v2' and path==Path(case['frame'])/'frame.json' and 'framing' not in value:
            audit=read(ROOT/'training/framing-audit-v2.json')
            proof=next(r for r in audit['frames'] if r['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            return dict(value,framing=proof)
        return value
    orient = solver.oriented
    def add_horizontal(charts):
        walls,hint = orient(charts)
        return walls+horizontal_charts(walls,planes,size,rgb),hint
    for arm,charts in [('trim',filtered),('rechart',filtered+added)]:
        chart_root=root/arm/'charts'
        save(chart_root/rel,dict(packet,charts=charts))
        for horiz in (False, True):
            folder=root/(arm+'-horizontal' if horiz else arm)
            mode=(chart_root,'charts.json',False,1,2.5,.5,False)
            with patch.object(paths,'RUN',source),patch.object(assembly,'ASSEMBLY',folder/'assembly'),patch.dict(assembly.MODES,{'connected-atlas':mode}),patch.object(expanded_scores,'read',checked),patch.object(solver,'oriented',add_horizontal if horiz else orient):
                assembly.evaluate((case,'sky','connected-atlas'))
    print(cohort,name,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('cohorts',nargs='+',choices=('original','fresh','supplemental','sky','new'))
    cli.add_argument('--workers',type=int,default=5)
    args=cli.parse_args()
    jobs=[(group,c) for group in args.cohorts for c in inputs(group)[-1]]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))
