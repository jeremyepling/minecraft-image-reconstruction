"""Partition visibly mixed wall masks and verify a local texture ruler before replacing its grid."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import garden_anchor_extension as garden
from benchmark.expanded_wall_rulers import NATIVE
from benchmark.floating_verified_wall_expansion import (ROOT, WORKTREES, BASE, PREFIX, jobs,
    data, combined, read, save, digest, summarize, project, _native_alpha, full_face)
from benchmark.expanded_wall_source_resample import preserve_and_rank, resample, TEXELS, atlas
from engine.geometry import transform

OUT = ROOT/'runs/mixed-refit'
REVIEW = ROOT/'benchmark/reviews/floating-mixed-wall-refit-01'
GUARD = WORKTREES/'floating-verified-wall-expansion/runs/verified-wall-expansion'
ARMS = ('split', 'refit', 'visible-patches')


def depth_field(panel, case, source):
    group = Path(case['dataset'])/case['id']
    if panel == 'sky':
        folder = combined.assembly.SHARED/'baseline/sky'/group/'handoffs'
    elif panel == 'original':
        folder = WORKTREES/'full-expanded-controls/runs/expanded-baseline-01'/group/'handoffs'
        if not folder.is_dir():
            folder = NATIVE.parent/'fields'/group
    else:
        folder = source.parent/'fields'/group
    path = folder/'depth.npz'
    stamp=digest(path)
    for receipt in (folder/'receipt.json',folder.parent/'receipt.json'):
        if receipt.is_file():
            rec=read(receipt)
            assert rec['source_sha256']==case['variants']['sky']['source_sha256']
            if 'fields_sha256' in rec:
                assert stamp==rec['fields_sha256']['depth']
            break
    with np.load(path) as f:
        values = f['value'].squeeze()
    return values, stamp


def partition(rgb, mask, depth):
    """Group visible pixels by smoothed Lab appearance and local inverse-depth residual."""
    y,x = np.nonzero(mask)
    x0,x1,y0,y1 = x.min(),x.max()+1,y.min(),y.max()+1
    inside = mask[y0:y1,x0:x1]
    blur = lambda x: cv2.GaussianBlur(x,(0,0),3)
    weight = np.maximum(blur(inside.astype('f')),1.e-6)
    lab = cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2LAB).astype('f')
    avg = blur(lab*inside[:,:,None])/weight[:,:,None]
    d = cv2.resize(depth.astype('f'),rgb.shape[1::-1],interpolation=cv2.INTER_LINEAR)[y,x]
    A = np.stack([np.ones(len(x)),x-x.mean(),y-y.mean()],axis=1)
    trend = np.linalg.lstsq(A[::13],d[::13],rcond=None)[0]
    residual = d - A@trend
    scale = max(np.percentile(abs(residual),85),.04*np.median(abs(d)),1.e-6)
    f = np.column_stack([avg[inside], 16*np.clip(residual/scale,-2,2)]).astype('f')
    # Deterministic farthest-point initialization on an evenly sampled population.
    samples = f[::17]
    centers = [np.median(samples,axis=0)]
    while len(centers)<4:
        dist = np.min([np.sum((samples-c)**2,axis=1) for c in centers],axis=0)
        centers.append(samples[np.argmin(abs(dist-np.percentile(dist,95)))])
    for _ in range(14):
        labels = np.argmin([np.sum((samples-c)**2,axis=1) for c in centers],axis=0)
        centers = [np.mean(samples[labels==k],axis=0) if np.any(labels==k) else c for k,c in enumerate(centers)]
    color = np.zeros(inside.shape,'uint8')
    color[inside] = 1+np.argmin([np.sum((f-c)**2,axis=1) for c in centers],axis=0)
    color = cv2.medianBlur(color,5)
    # Preserve every input pixel. Grow small islands from their nearest large same-source component.
    pieces=np.zeros_like(color,dtype='int32')
    n=0
    for k in range(1,5):
        ct,cc,stats,_=cv2.connectedComponentsWithStats((inside&(color==k)).astype('uint8'))
        for i in range(1,ct):
            if stats[i,cv2.CC_STAT_AREA]>=max(400, .009*len(x)):
                n+=1;pieces[cc==i]=n
    if n==0:
        return mask.astype('int32')
    missing=inside&(pieces==0)
    if missing.any():
        # Only reassign observed pixels, and only within their original connected silhouette.
        _, islands=cv2.connectedComponents(inside.astype('uint8'))
        for key in np.unique(islands[missing]):
            area=islands==key
            available=area&(pieces>0)
            if available.any():
                _,idx=cv2.distanceTransformWithLabels((~available).astype('uint8'),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
                lut=np.zeros(idx.max()+1,int);lut[idx[available]]=pieces[available]
                pieces[area&missing]=lut[idx[area&missing]]
            else:
                n+=1;pieces[area]=n
    result=np.zeros(mask.shape,'int32');result[y0:y1,x0:x1]=pieces
    assert np.array_equal(result>0,mask)
    return result


def rectify(quad, ratio, rgb, region):
    """Use the source quadrilateral and its measured aspect as a texture sampling frame."""
    if ratio is None or not np.isfinite(ratio) or not .05<ratio<20:
        return None
    q=np.array(quad)
    tall=np.mean(np.linalg.norm(q[[0,1]]-q[[3,2]],axis=1))
    H=transform(q)@np.diag([1/(tall*ratio),1/tall,1.])
    yy,xx=np.nonzero(region)
    uv=project(np.linalg.inv(H),np.column_stack([xx+.5,yy+.5]))
    low=np.floor(uv.min(0)).astype(int)-2;size=np.ceil(uv.max(0)).astype(int)-low+2
    if min(size)<12 or max(size)>2000 or np.prod(size)>1000000:
        return None
    H=H@np.array([[1.,0,low[0]],[0,1.,low[1]],[0,0,1.]])
    y,x=np.indices(size[::-1]);xy=(project(H,np.stack([x+.5,y+.5],axis=-1))-.5).astype('float32')
    image=cv2.remap(rgb,xy,None,cv2.INTER_LINEAR)
    support=cv2.remap(region.astype('uint8'),xy,None,cv2.INTER_NEAREST)
    return H,image,support


def partial_witnesses(h, uv, region, rgb, texture):
    """Compare the proposed phase against quarter-tile competitors in visible native source patches."""
    patterns=atlas().patterns[texture].reshape(8,-1)
    correlations=[]
    for offset in ((0,0),(.25,0),(-.25,0),(0,.25),(0,-.25)):
        xy=np.floor(project(h,uv[:,None,None,:]+TEXELS+offset)).astype(int)
        x,y=xy[...,0],xy[...,1]
        valid=(x>=0)&(x<region.shape[1])&(y>=0)&(y<region.shape[0])
        x,y=x.clip(0,region.shape[1]-1),y.clip(0,region.shape[0]-1)
        valid=(valid&region[y,x]).reshape(len(uv),-1)
        m=valid.astype(float)
        n=np.maximum(1,m.sum(1))
        a=rgb[y,x].mean(-1).reshape(len(uv),-1)
        a=(a-(a*m).sum(1)[:,None]/n[:,None])*m
        var=m@(patterns**2).T-(m@patterns.T)**2/n[:,None]
        score=(a@patterns.T)/np.sqrt(np.maximum(1,(a*a).sum(1)[:,None]*var))
        score[n<128]=-1
        correlations.append(score)
    scores=np.stack(correlations)
    controls=np.where(scores[1:]>-1,scores[1:],np.nan)
    # Require a majority of the competing shifts to remain visible for a fair comparison.
    count=np.isfinite(controls).sum(0)
    control=np.nan_to_num(controls,nan=-1).max(0)
    passed=(scores[0]>=.60)&(scores[0]-control>=.07)&(count>=3)
    orientation=int(np.argmax(passed.sum(0)))
    return [dict(uv=pt.tolist(),material=atlas().labels[texture],corr=float(v),phase=float(v-z))
            for pt,ok,v,z in zip(uv,passed[:,orientation],scores[0,:,orientation],control[:,orientation]) if ok]


def region_ruler(chart, region, rgb, quad, aspect):
    aligned=rectify(quad,aspect,rgb,region)
    if aligned is None:
        return {},dict(reason='small rectification')
    H,img,mask=aligned
    steps,peaks=garden.periods(img,mask)
    anchors=garden.seeds(img,mask)
    if not anchors or not steps:
        return {},dict(reason='no resolved source period',peaks=peaks)
    # Learn phase at one patch; verify on native tiles separated from that seed.
    train=anchors[:1]
    proposals=garden.anchors(img,mask,steps[:8],train)
    hypotheses={arm:[] for arm in ARMS[1:]}
    max_hits={arm:0 for arm in ARMS[1:]}
    visited=set()
    for _,_,idx,ox,oy,dx,dy in proposals:
        phase=(round(dx,1),round(dy,1),round((ox/dx)%1,1),round((oy/dy)%1,1))
        if len(visited)>=12:
            break
        if phase in visited:
            continue
        visited.add(phase)
        h=H@np.array([[dx,0,ox],[0,-dy,oy+dy],[0,0,1.]])
        points=np.column_stack(np.nonzero(region)[::-1])+.5
        uv=np.unique(np.floor(project(np.linalg.inv(h),points)).astype(int),axis=0)
        train_xy=project(H,np.array([(t[0],t[1]) for t in train]))
        train_uv=project(np.linalg.inv(h),train_xy)
        candidates=np.flatnonzero(np.max(abs(uv+.5-train_uv[0]),axis=1)>1.5)
        if len(candidates)<2:
            continue
        indices=candidates[np.linspace(0,len(candidates)-1,min(20,len(candidates)),dtype=int)]
        hits=[]
        for i in indices:
            e=full_face(h,uv[i],region,rgb)
            if e.get('ok'):
                hits.append(dict(uv=uv[i].tolist(),material=e['material'],corr=e['correlation'],phase=e['phase_advantage']))
        partial=partial_witnesses(h,uv[indices],region,rgb,idx)
        for arm,witness in [('refit',hits),('visible-patches',partial)]:
            max_hits[arm]=max(max_hits[arm],len(witness))
            if len(witness)>=2 and max(np.ptp([hit['uv'] for hit in witness],axis=0))>=2:
                hypotheses[arm].append((len(witness),np.mean([hit['corr'] for hit in witness]),h,witness,dx,dy))
    result={};evidence={}
    for arm,hyp in hypotheses.items():
        if not hyp:
            continue
        _,_,h,hits,dx,dy=max(hyp,key=lambda p:p[:2])
        blank=dict(chart,cells=[])
        sampled=resample(blank,h,region,rgb,points)
        fixed=preserve_and_rank(blank,sampled,rgb,region)
        if fixed['cells']:
            result[arm]=fixed
            evidence[arm]=dict(holdout=hits,pitch=[dx,dy],cells=len(fixed['cells']))
    return result,dict(reason='verified' if result else 'no separated native confirmations',
                       proposals=len(visited),peaks=peaks,max_hits=max_hits,evidence=evidence)


def process(group,case):
    cv2.setNumThreads(1)
    name=case['id'];rel=Path('sky')/name;src,_,_,_=data(group)
    baseline=BASE/group/'large-contraction'
    chartfile=baseline/'charts'/rel/'charts.json';packet=read(chartfile)
    audited=read(GUARD/group/'audit'/(name+'.json.gz'))
    assert audited['base']==digest(chartfile) and audited['source']==packet['source_sha256']
    flagged={cid for cid,ev in audited['charts'].items() if ev['peer_guard']}
    variants={arm:[] for arm in ARMS};details={}
    sha=None
    if flagged:
        path=Path(case['variants']['sky']['image'])
        assert digest(path)==packet['source_sha256']
        rgb=np.array(Image.open(path).convert('RGB'));depth,sha=depth_field(group,case,src)
        planes={p['id']:p for p in read(src/rel/'source-handoff/surfaces.json')['planes']}
        counts={r['id']:r for r in read(src/rel/'source-handoff/counts.json')['surfaces']}
        for c in packet['charts']:
            cid=c['id']
            if cid not in flagged:
                for v in variants.values():v.append(c)
                continue
            alpha,(x,y,r,b)=_native_alpha(planes[cid]['support'],rgb.shape[1::-1])
            mask=np.zeros(rgb.shape[:2],bool);mask[y:b,x:r]=np.asarray(alpha)>0
            parts=partition(rgb,mask,depth)
            centers=project(np.array(c['H']),np.array([(v['u']+.5,v['v']+.5) for v in c['cells']])).astype(int)
            ids=parts[centers[:,1],centers[:,0]]
            if np.any(ids==0):
                raise ValueError(f'Unowned original cell center on {name} {cid}')
            record=[]
            for key,freq in Counter(ids.tolist()).most_common():
                region=parts==key
                sub=dict(c,id=f'{cid}~r{key}',cells=[v for v,j in zip(c['cells'],ids) if j==key])
                new,ev=region_ruler(sub,region,rgb,counts[cid]['quad'],counts[cid]['evidence'].get('aspect'))
                variants['split'].append(sub)
                for arm in ARMS[1:]:variants[arm].append(new.get(arm,sub))
                record.append(dict(part=key,pixels=int(region.sum()),baseline_cells=freq,**ev))
            details[cid]=record
            # Every originally observed cell survived the grouping intervention.
            assert sum(r['baseline_cells'] for r in record)==len(c['cells'])
            save(OUT/group/'source'/name/'region-legend.json',record)
            Image.fromarray(parts.astype('uint16')).save(OUT/group/'source'/name/'region-ids.png')
    roots=dict(base=baseline/'assembly')
    for arm in ARMS:
        if flagged:
            dest=OUT/group/arm;save(dest/'charts'/rel/'charts.json',dict(packet,charts=variants[arm]))
            with patch.object(combined,'inputs',data):
                combined.run((group,case,dest/'charts',dest/'assembly'))
            roots[arm]=dest/'assembly'
        else:
            roots[arm]=baseline/'assembly'
    scored=summarize([case],roots,PREFIX)['rows'][0]
    ev=dict(panel=group,source=packet['source_sha256'],base=digest(chartfile),depth=sha,regions=details,
            pack={k:digest(v/PREFIX/name/'reconstruction.zip') for k,v in roots.items()})
    save(OUT/group/'rows'/(name+'.json.gz'),dict(scored,audit=ev))
    print(group,name,[(k,v['gained'],v['lost'],v['added'],v['removed']) for k,v in scored['paired'].items()] if flagged else '',flush=True)
    return name


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=3)
    cli.add_argument('--affected',action='store_true')
    args=cli.parse_args()
    work=jobs()
    if args.affected:
        work=[(g,c) for g,c in work if read(GUARD/g/'audit'/(c['id']+'.json.gz'))['arms']['peer-guard']['changed']]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(process,*zip(*work)))
    if not args.affected:
        rows=[read(OUT/g/'rows'/(c['id']+'.json.gz')) for g,c in work]
        save(REVIEW/'rows.json.gz',rows)
        from benchmark.floating_mixed_wall_report import publish
        publish(rows)
