"""Conservative half-pitch gate frozen before fresh outputs. Consume the first experiment's exports."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from unittest.mock import patch

from benchmark.floating_grid_evidence import (
    FRESH, NATIVE, OUT, ROOT, TEXELS, assembly, atlas, cases, cv2, digest, expanded_scores,
    Image, np, paths, project, rank_tile, read, save, _native_alpha
)
from pathlib import Path


def gate(old, base, half, obs, mask, rgb):
    if obs['chosen']!='half':
        return False,'no fine-pitch proposal'
    if any(not c.get('fallback',False) and c.get('confidence',0)>=.90 for c in old['cells']):
        return False,'trusted existing sprite'
    if len(half['cells'])>len(base['cells'])+256:
        return False,'more than 256 additional cells'
    # Examine left and right halves separately. Use only the newly inserted
    # middle boundaries between half-cells. Sample the paired atlas crops on
    # a different checkerboard parity so their pixels are spatially separated.
    H=np.asarray(half['H'])
    inv=np.linalg.inv(np.asarray(base['H']))
    center=np.median([c['u'] for c in half['cells']])
    groups=[[],[]]
    for c in half['cells']:
        groups[int(c['u']>center)].append(c)
    audit=[]
    for group in groups:
        if len(group)<6:
            return False,'insufficient separated tiles'
        seam_votes,atlas_votes=[],[]
        for i in np.linspace(0,len(group)-1,min(48,len(group)),dtype=int):
            c=group[i]
            key=np.array([c['u'],c['v']])
            old_uv=project(inv,project(H,key+.5))
            if (int(np.floor(old_uv[0]))+int(np.floor(old_uv[1])))%2:
                # Require a new edge at this fine-cell boundary.
                for ax in (0,1):
                    side=0 if abs((old_uv[ax]-.25)%1-.5)<.1 else 1
                    t=(np.arange(12)+.5)/12
                    b=np.stack([np.stack([np.full_like(t,side+d),t],1) for d in (-.08,.08,.28,.44)])
                    if ax:
                        b=b[...,::-1]
                    xy=np.floor(project(H,key+b)).astype(int)
                    x,y=xy[...,0],xy[...,1]
                    valid=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
                    x,y=x.clip(0,mask.shape[1]-1),y.clip(0,mask.shape[0]-1)
                    use=(valid&mask[y,x]).all(0)
                    if use.sum()>=6:
                        color=rgb[y,x].astype(float)
                        dif=np.abs(color[0]-color[1])-np.abs(color[2]-color[3])
                        seam_votes.append(float(dif[use].mean()))
            else:
                scores=[]
                for warp,origin in ((H,key),(np.asarray(base['H']),np.floor(old_uv))):
                    xy=np.floor(project(warp,origin+TEXELS)).astype(int)
                    x,y=xy[...,0],xy[...,1]
                    valid=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
                    x,y=x.clip(0,mask.shape[1]-1),y.clip(0,mask.shape[0]-1)
                    valid &= mask[y,x]
                    if valid.mean()>=.78:
                        result=rank_tile(dict(rgb=rgb[y,x],mask=valid),atlas())
                        scores.append(result['evidence']['material_score'])
                if len(scores)==2:
                    atlas_votes.append(scores[0]-scores[1])
        if min(len(seam_votes),len(atlas_votes))<3:
            return False,'insufficient independent votes'
        audit.append((float(np.median(seam_votes)),float(np.median(atlas_votes))))
    return all(s>2 and a>.06 for s,a in audit),audit


def run(case,cohort):
    cv2.setNumThreads(1)
    source=FRESH/'ordinary' if cohort=='fresh' else NATIVE.parent/'source'
    root=OUT/cohort
    name=case['id']
    raw=read(source/'atlas/connected/sky'/name/'charts.json')
    evidence=read(root/'evidence'/(name+'.json'))
    evidence_by_id={c['id']:c for c in evidence['charts']}
    orig={c['id']:c for c in raw['charts']}
    packet=read(root/'current/charts/sky'/name/'charts.json')
    halves={c['id']:c for c in read(root/'half/charts/sky'/name/'charts.json')['charts']}
    parts={p['id']:p for p in read(source/'sky'/name/'source-handoff/surfaces.json')['planes']}
    image=Path(case['variants']['sky']['image'])
    rgb=np.asarray(Image.open(image).convert('RGB'))
    assert digest(image)==evidence['source_sha256']
    charts,decisions=[],[]
    for c in packet['charts']:
        pid=c['id']
        yes=False
        if pid in evidence_by_id:
            alpha,(x,y,x1,y1)=_native_alpha(parts[pid]['support'],rgb.shape[1::-1])
            mask=np.zeros(rgb.shape[:2],bool)
            mask[y:y1,x:x1]=np.asarray(alpha)>0
            yes,reason=gate(orig[pid],c,halves[pid],evidence_by_id[pid],mask,rgb)
            decisions.append(dict(id=pid,accepted=yes,reason=reason))
        charts.append(halves[pid] if yes else c)
    folder=root/'guarded'
    save(folder/'charts/sky'/name/'charts.json',dict(packet,charts=charts))
    mode=(folder/'charts','charts.json',False,1,2.5,.5,False)
    def checked_read(path):
        data=read(path)
        if case['dataset']=='visible-builds-v2' and path==image.with_name('frame.json'):
            proof=next(r for r in read(ROOT/'training/framing-audit-v2.json')['frames'] if r['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            return dict(data,framing=proof)
        return data
    with patch.object(paths,'RUN',source),patch.object(assembly,'ASSEMBLY',folder/'assembly'),patch.dict(assembly.MODES,{'connected-atlas':mode}),patch.object(expanded_scores,'read',checked_read):
        assembly.evaluate((case,'sky','connected-atlas'))
    save(folder/'evidence'/(name+'.json'),decisions)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('cohort',choices=('original','fresh'))
    parser.add_argument('--ids',nargs='+')
    a=parser.parse_args()
    jobs=list(cases()) if a.cohort=='original' else read(FRESH/'manifest.json')['cases']
    if a.ids:
        jobs=[c for c in jobs if c['id'] in a.ids]
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(partial(run,cohort=a.cohort),jobs))
