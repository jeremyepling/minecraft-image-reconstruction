"""Select block pitch on observed vertical walls using only native RGB and inferred support."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, read, save, digest
from benchmark.expanded_wall_rulers import cases, NATIVE, SAMPLES
from benchmark.expanded_wall_source_resample import TEXELS, preserve_and_rank
from benchmark.full_block_lattice import atlas
from benchmark.shared_corner_rows import sample
from engine.atlas_count import rank_tile
from engine.geometry import UNIT, project
from engine.surface_perimeter import _native_alpha

OUT = ROOT / 'runs/grid-evidence'
FRESH = WORKTREES / 'floating-fresh-validation/runs/floating-fresh'
ORIGINAL = WORKTREES / 'expanded-wall-rulers/runs/wall-rulers/source_material'
ARMS = ('current', 'half', 'double', 'selected')
SCALES = {'current': 1., 'half': .5, 'double': 2.}


def options(chart, baseline, rgb, mask):
    points = np.column_stack(np.nonzero(mask)[::-1]) + .5
    H = np.array(chart['H'])
    old = np.array([(c['u'], c['v']) for c in chart['cells']])
    anchor = old[np.argmin(np.linalg.norm(old - np.median(old, axis=0), axis=1))]
    native_uv = project(np.linalg.inv(H), points)
    # Sample the same visible pixels for all rulers, with a deterministic stride.
    witnesses = np.linspace(0, len(points)-1, min(32, len(points)), dtype=int)
    records, candidates = {}, {}
    for arm, scale in SCALES.items():
        transform = np.diag([scale, scale, 1.])
        transform[:2, 2] = anchor * (1-scale)
        warp = H @ transform
        units = (native_uv - transform[:2, 2]) / scale
        keys = np.unique(np.floor(units[witnesses]).astype(int), axis=0)
        scores = []
        for key in keys:
            coords = np.floor(project(warp, key + TEXELS)).astype(int)
            x, y = coords[...,0], coords[...,1]
            inside = (x>=0) & (y>=0) & (x<mask.shape[1]) & (y<mask.shape[0])
            x,y=x.clip(0,mask.shape[1]-1),y.clip(0,mask.shape[0]-1)
            valid = inside & mask[y,x]
            if valid.mean() < .78:
                continue
            tile = rgb[y,x]
            ranked = rank_tile(dict(rgb=tile, mask=valid), atlas())
            if not ranked['block_id']:
                continue
            # Compare across each cell boundary and an equal-distance inside pair.
            # Repeated sprite interiors and facade openings can both supply evidence.
            seam, interior = [], []
            t = (np.arange(12)+.5)/12
            for axis in (0,1):
                bands=np.stack([np.stack([np.full_like(t,n),t],axis=1) for n in (-.065,.065,.435,.565)])
                if axis:
                    bands=bands[...,::-1]
                xy=np.floor(project(warp,key+bands)).astype(int)
                bx,by=xy[...,0],xy[...,1]
                ok=(bx>=0)&(by>=0)&(bx<mask.shape[1])&(by<mask.shape[0])
                bx,by=bx.clip(0,mask.shape[1]-1),by.clip(0,mask.shape[0]-1)
                colors=rgb[by,bx].astype(float)
                for i, values in ((0,seam),(2,interior)):
                    m=ok[i]&ok[i+1]&(mask[by[i],bx[i]]|mask[by[i+1],bx[i+1]])
                    d=np.abs(colors[i]-colors[i+1]).mean(1)
                    values.extend(d[m])
            contrast=(np.mean(seam)-np.mean(interior))/40 if seam and interior else 0.
            scores.append(ranked['evidence']['material_score'] + .20*np.clip(contrast,-1,1))
        records[arm] = dict(samples=len(scores), score=float(np.median(scores)) if len(scores)>=4 else None)
        if arm=='current':
            candidates[arm]=baseline
            continue
        bins = np.unique(np.floor(units).astype(np.int64), axis=0)
        bins = bins[sample(mask, project(warp,bins+.5),1)]
        cells=[]
        for start in range(0,len(bins),512):
            chunk=bins[start:start+512]
            quads=project(warp,chunk[:,None,:]+UNIT)
            length=np.linalg.norm(quads-np.roll(quads,1,axis=1),axis=-1).min(1)
            support=sample(mask,project(warp,chunk[:,None,None,:]+SAMPLES),1)
            coverage=support.mean((1,2))
            for i in np.flatnonzero((length>=3.5)&support[:,3,3]&(coverage>=.78)):
                cells.append(dict(u=int(chunk[i,0]),v=int(chunk[i,1]),mask_coverage=float(coverage[i])))
        candidates[arm]=preserve_and_rank(dict(cells=[]),dict(baseline,H=warp.tolist(),cells=cells),rgb,mask)
    current=records['current']['score']
    winner=max(SCALES,key=lambda arm: records[arm]['score'] if records[arm]['score'] is not None else -np.inf)
    if current is None or records[winner]['score'] < current+.06:
        winner='current'
    return candidates|{'selected':candidates[winner]}, dict(id=chart['id'],anchor=anchor.tolist(),chosen=winner,options=records)


def run(case, fresh=False):
    cv2.setNumThreads(1)
    name=case['id']
    root=OUT/('fresh' if fresh else 'original')
    source=FRESH/'ordinary' if fresh else NATIVE.parent/'source'
    baseline=FRESH/'source_material' if fresh else ORIGINAL
    image=Path(case['variants']['sky']['image'])
    sha=digest(image)
    assert sha==case['rgb_sha256']
    rgb=np.array(Image.open(image).convert('RGB'))
    handoff=source/'sky'/name/'source-handoff'
    original=source/'atlas/connected/sky'/name/'charts.json'
    material=baseline/'charts/sky'/name/'charts.json'
    packet=read(material)
    assert packet['source_packet_sha256']==digest(original)
    paths.check_handoff(packet,handoff,sha)
    parts={p['id']:p for p in read(handoff/'surfaces.json')['planes']}
    old={c['id']:c for c in read(original)['charts']}
    outputs={a:[] for a in ARMS}
    evidence=[]
    for chart in packet['charts']:
        c=old[chart['id']]
        normal=c.get('normal_camera')
        if not c['cells'] or normal is None or abs(normal[1])>=.67:
            variants=dict.fromkeys(ARMS,chart)
        else:
            alpha,(x,y,x1,y1)=_native_alpha(parts[c['id']]['support'],rgb.shape[1::-1])
            mask=np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1]=np.asarray(alpha)>0
            variants,obs=options(c,chart,rgb,mask)
            evidence.append(obs)
        for a in ARMS:
            outputs[a].append(variants[a])
    def checked_read(path):
        data=read(path)
        if case['dataset']=='visible-builds-v2' and path==image.with_name('frame.json'):
            audit=read(ROOT/'training/framing-audit-v2.json')
            proof=next(r for r in audit['frames'] if r['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==sha
            return dict(data,framing=proof)
        return data
    for arm in ARMS:
        folder=root/arm
        save(folder/'charts/sky'/name/'charts.json',dict(packet,charts=outputs[arm]))
        mode=(folder/'charts','charts.json',False,1,2.5,.5,False)
        with patch.object(paths,'RUN',source),patch.object(assembly,'ASSEMBLY',folder/'assembly'),patch.dict(assembly.MODES,{'connected-atlas':mode}),patch.object(expanded_scores,'read',checked_read):
            assembly.evaluate((case,'sky','connected-atlas'))
    result=Path('assembly/connected-atlas/sky')/name/'placements.json'
    assert read(root/'current'/result)==read(baseline/result)
    save(root/'evidence'/(name+'.json'),dict(source_sha256=sha,baseline_packet_sha256=digest(material),charts=evidence,atlas=atlas().provenance))
    print(name,len(evidence),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--fresh',action='store_true')
    p.add_argument('--ids',nargs='+')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    jobs=list(cases())
    if a.fresh:
        jobs=[]
        for dataset in ('visible-builds-v2','visible-builds-v3','decorative-builds-v1'):
            for rec in read(SOURCE/dataset/'manifest.json')['frames']:
                if a.ids and rec['id'] in a.ids:
                    frame=SOURCE/dataset/'data'/rec['split']/'frames'/rec['id']
                    jobs.append(dict(rec,dataset=dataset,frame=str(frame),variants={'sky':dict(image=str(frame/'rgb.png'),source_sha256=rec['rgb_sha256'])}))
    if a.ids:
        jobs=[case for case in jobs if case['id'] in a.ids]
        assert len(jobs)==len(a.ids)
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(partial(run,fresh=a.fresh),jobs))
