"""Test whole-mask supported-cell resampling on ordinary predicted surfaces and grids."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.expanded_wall_rulers import NATIVE, OUT, cases, resample
from benchmark.full_block_lattice import atlas
from engine.atlas_count import rank_tile
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

TEXELS=np.stack(np.meshgrid((np.arange(16)+.5)/16,(np.arange(16)+.5)/16),axis=-1)


def preserve_and_rank(old, chart, rgb, mask):
    original={(c['u'],c['v']):c for c in old['cells']}
    H=np.asarray(chart['H'])
    cells=[]
    for cell in chart['cells']:
        key=(cell['u'],cell['v'])
        if key in original:
            cells.append(original[key])
            continue
        xy=np.floor(project(H,TEXELS+key)).astype(int)
        x,y=xy[...,0],xy[...,1]
        height,width=mask.shape
        inside=(x>=0)&(x<width)&(y>=0)&(y<height)
        x,y=x.clip(0,width-1),y.clip(0,height-1)
        ranked=rank_tile(dict(rgb=rgb[y,x],mask=inside & mask[y,x]),atlas())
        if ranked['block_id']:
            cells.append(dict(cell,block_id=ranked['block_id'],fallback=False,
                confidence=max(0.,ranked['evidence']['correlation'])))
        else:
            cells.append(dict(cell,block_id='smooth_stone',fallback=True,confidence=0.))
    # The chart confidence controls tie breaks at assembly. Keep its value frozen.
    return dict(chart,cells=cells)


def run(case, material=False):
    cv2.setNumThreads(1)
    name=case['id']
    root=OUT/('source_material' if material else 'source_resample')
    source=NATIVE.parent/'source'
    src=Path(case['variants']['sky']['image'])
    assert digest(src)==case['rgb_sha256']
    rgb=np.array(Image.open(src).convert('RGB'))
    handoff=source/'sky'/name/'source-handoff'
    original=source/'atlas/connected/sky'/name/'charts.json'
    packet=read(original)
    paths.check_handoff(packet,handoff,digest(src))
    parts={p['id']:p for p in read(handoff/'surfaces.json')['planes']}
    output=[]
    evidence=[]
    if material:
        measured=read(OUT/'source_resample/charts/sky'/name/'charts.json')
        assert measured['source_packet_sha256']==digest(original)
        resampled={c['id']:c for c in measured['charts']}
    for chart in packet['charts']:
        p=parts[chart['id']]
        normal=chart.get('normal_camera')
        if not chart['cells'] or normal is None or abs(normal[1])>=.67:
            output.append(chart)
            continue
        alpha,(x,y,x1,y1)=_native_alpha(p['support'],rgb.shape[1::-1])
        mask=np.zeros(rgb.shape[:2],bool)
        mask[y:y1,x:x1]=np.asarray(alpha)>0
        if material:
            result=preserve_and_rank(chart,resampled[chart['id']],rgb,mask)
        else:
            points=np.column_stack(np.nonzero(mask)[::-1])+.5
            result=resample(chart,np.asarray(chart['H']),mask,rgb,points)
        output.append(result)
        evidence.append(dict(id=chart['id'],source_cells=len(chart['cells']),resampled_cells=len(result['cells'])))
    save(root/'charts/sky'/name/'charts.json',dict(packet,charts=output,source_packet_sha256=digest(original)))
    save(root/'evidence'/(name+'.json'),dict(id=name,charts=evidence))
    def read_v2(path):
        result=read(path)
        if case['dataset']=='visible-builds-v2' and path==src.with_name('frame.json'):
            audit=read(ROOT/'training/framing-audit-v2.json')
            proof=next(x for x in audit['frames'] if x['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            return result|dict(framing=proof)
        return result
    mode=(root/'charts','charts.json',False,1,2.5,.5,False)
    with patch.object(paths,'RUN',source), patch.object(assembly,'ASSEMBLY',root/'assembly'), \
         patch.dict(assembly.MODES,{'connected-atlas':mode}), patch.object(expanded_scores,'read',read_v2):
        assembly.evaluate((case,'sky','connected-atlas'))
    if material:
        export=Path('assembly/connected-atlas/sky')/name/'placements.json'
        previous=read(OUT/'source_resample'/export)
        current=read(root/export)
        for space in ('local','aligned'):
            assert [p for p,m in previous[space]]==[p for p,m in current[space]]
    print(name,len(evidence),flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--material',action='store_true')
    a=p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(partial(run,material=a.material),cases()))
