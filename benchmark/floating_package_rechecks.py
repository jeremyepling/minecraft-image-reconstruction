"""Replay source depth splitting and a projected sky veto on the same selected expanded build."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_depth_before_atlas as depth
from benchmark import floating_mixed_wall_report as report
from benchmark.floating_bridges_report import read_arm
from benchmark.floating_verified_wall_expansion import (
    BASE, PREFIX, ROOT, WORKTREES, _native_alpha, choose, combined, data,
    digest, jobs, peer_pitches, project, read, save, summarize)
from engine.floating_sky import sky_support

PACKAGE = WORKTREES / 'floating-selected-package/runs/selected-package'
OLD_MASKS = ROOT / 'benchmark/reviews/floating-depth-before-atlas-01'
OLD_SKY = WORKTREES / 'floating-projected-sky/benchmark/reviews/floating-projected-sky-01'
OUT = ROOT / 'runs/package-rechecks'
REVIEW = ROOT / 'benchmark/reviews/floating-package-rechecks-01'
ARMS = ('split', 'border', 'split-border', 'split-original-target')
SAMPLES = np.stack(np.meshgrid((np.arange(9)+.5)/9, (np.arange(9)+.5)/9), -1)


def native_mask(plane, size):
    alpha, (x,y,r,b) = _native_alpha(plane['support'],size)
    mask = np.zeros(size[::-1],bool)
    mask[y:b,x:r] = np.asarray(alpha)>0
    return mask


def guard_targets(charts, seeds, planes, rgb):
    """Reevaluate the historical extrapolation and peer guards on changed packets."""
    flags = set()
    for chart in charts:
        cid = chart['id']
        if cid not in seeds or len(chart['cells'])<256:
            continue
        support = native_mask(planes[cid],rgb.shape[1::-1])
        options, ev = choose(chart,seeds[cid],support,rgb)
        if not options:
            continue
        peer = peer_pitches(chart,charts)
        anchor = {(x['uv'][0]//6,x['uv'][1]//6) for x in ev['seeds'] if x['ok']}
        if len(peer)>=2 and min(peer.values())>=2 and len(ev['supported_regions'])<=len(anchor):
            flags.add(cid)
    return flags


def region_ids(group,case,cid,plane,rgb,src,archived):
    mask = native_mask(plane,rgb.shape[1::-1])
    if archived:
        png = OLD_MASKS / (case['id']+'-'+cid+'-ids.png')
        meta = archived['audit'][cid]['source']
        assert digest(png)==meta['ids_sha256'] and meta['source_sha256']==case['variants']['sky']['source_sha256']
        ids = np.asarray(Image.open(png))
    else:
        # New eligibility can activate an image whose masks were never cached.
        normal,dist,_ = depth.source_fields(group,case,src,rgb.shape[1::-1])
        geometry,info = depth.partition(rgb,mask,normal,dist)
        ids,_,_ = depth.refine_adjacent(rgb,mask,normal,dist,geometry,info)
    assert np.array_equal(ids>0,mask)
    return ids


def partition_charts(charts, flags, fields):
    out=[]
    details={}
    for c in charts:
        cid=c['id']
        if cid not in flags:
            out.append(c)
            continue
        cells=c['cells']
        uv=np.array([[p['u']+.5,p['v']+.5] for p in cells])
        xy=project(np.asarray(c['H']),uv).astype(int)
        ids=fields[cid]
        assert np.all((xy>=0)&(xy<ids.shape[::-1]))
        codes=ids[xy[:,1],xy[:,0]]
        # A changed grid can move a center outside source support. Preserve any
        # such input in an explicit residual chart, without inventing coverage.
        grouped={}
        for cell,k in zip(cells,codes):
            grouped.setdefault(int(k),[]).append(cell)
        for k,rows in sorted(grouped.items()):
            out.append(dict(c,id=f'{cid}~p{k}',cells=rows))
        assert sum(map(len,grouped.values()))==len(cells)
        details[cid]={k:len(v) for k,v in grouped.items()}
    return out,details


def border_filter(charts, sky):
    """The original 81 native footprint votes on these single-homography chart cells."""
    out=[]
    drops={}
    for c in charts:
        assert 'export_uv' not in c and 'export_cells' not in c
        cells=c['cells']
        if not cells:
            out.append(c)
            continue
        assert not any(v.get('normal_offset',0) for v in cells)
        uv=np.array([[p['u'],p['v']] for p in cells])
        pix=np.floor(project(np.asarray(c['H']),uv[:,None,None,:]+SAMPLES)).astype(int)
        x,y=pix[...,0],pix[...,1]
        h,w=sky.shape
        inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        votes=(sky[y.clip(0,h-1),x.clip(0,w-1)]&inside).sum((1,2))
        rejected=votes>40
        if rejected.any():
            drops[c['id']]=[[int(u),int(v),int(n)] for (u,v),n in zip(uv[rejected],votes[rejected])]
            # export_uv preserves all original charts and cells for corner joins,
            # anchoring and component priority; only the final exported set changes.
            out.append(dict(c,export_uv=uv[~rejected].tolist()))
        else:
            out.append(c)
    return out,drops


def sites(case, base, other):
    labels=read(Path(case['frame'])/'labels.json')
    states={b['id']:b['block_id'] for b in labels['blocks']}
    truth={tuple(f['xyz_world']):states[f['block']] for f in labels['faces']}
    result=[]
    for xyz in sorted(base.keys()|other.keys()):
        before,after=base.get(xyz),other.get(xyz)
        if before!=after:
            result.append(dict(xyz=xyz,before=before,after=after,visible_target=truth.get(xyz)))
    return result


def replay(job):
    group,case,old_split,old_sky=job
    cv2.setNumThreads(1)
    name=case['id']
    package=PACKAGE/group
    rel=Path('sky')/name/'charts.json'
    handoff=read(package/'handoffs'/(name+'.json'))
    packet=read(package/'charts'/rel)
    sha=digest(package/'charts'/rel)
    assert sha==handoff['chart_sha256']
    src,seeds_root,_,_=data(group)
    image=Path(case['variants']['sky']['image'])
    surface_path=src/'sky'/name/'source-handoff/surfaces.json'
    assert digest(surface_path)==handoff['surface_sha256']
    assert digest(image)==packet['source_sha256']==handoff['source_sha256']==old_split['source_sha256']
    original=BASE/group/'large-contraction/charts'/rel
    guard=read(depth.prior.GUARD/group/'audit'/(name+'.json.gz'))
    assert digest(original)==guard['base']==handoff['baseline_chart_sha256']
    before=read(original)['charts']
    charts=packet['charts']
    assert not any('export_cells' in c or 'export_uv' in c for c in charts)
    original_targets={k for k,ev in guard['charts'].items() if ev['peer_guard']}
    assert original_targets==set(old_split['audit'])
    with Image.open(image) as im:
        rgb=np.asarray(im.convert('RGB'))
        sky=sky_support(im)
    planes={p['id']:p for p in read(surface_path)['planes']}
    if charts!=before:
        seeds_file=seeds_root/rel
        assert digest(seeds_file)==guard['seeds']
        seeds={c['id']:c for c in read(seeds_file)['charts']}
        current=guard_targets(charts,seeds,planes,rgb)
    else:
        current=original_targets
    fields={cid:region_ids(group,case,cid,planes[cid],rgb,src,old_split if cid in original_targets else None)
            for cid in current|original_targets}
    split,active=partition_charts(charts,current,fields)
    original_split,original_active=partition_charts(charts,original_targets,fields)
    border,drop=border_filter(charts,sky)
    both,split_drop=border_filter(split,sky)
    candidates={'split':split,'border':border,'split-border':both,'split-original-target':original_split}
    base=package/'assembly'
    receipt_path=base/PREFIX/name/'score.json'
    rec=read(receipt_path)
    assert rec['chart_sha256']==sha and rec['source_sha256']==packet['source_sha256']
    roots={'base':base}
    # Identical proposals on a given frame use the same measured pack.
    applied=[(charts,base)]
    for arm,values in candidates.items():
        same=next((path for seen,path in applied if values==seen),None)
        if same is not None:
            roots[arm]=same
            continue
        root=OUT/group/arm
        save(root/'charts'/rel,dict(packet,charts=values))
        with patch.object(combined,'inputs',data):
            combined.run((group,case,root/'charts',root/'assembly'))
        roots[arm]=root/'assembly'
        applied.append((values,roots[arm]))
    scored=summarize([case],roots,PREFIX)['rows'][0]
    coords={}
    hashes={}
    changed={}
    for a,path in roots.items():
        folder=path/PREFIX/name
        receipt=read(folder/'score.json')
        assert receipt['reference_sha256']==rec['reference_sha256']
        hashes[a]=receipt['pack_sha256']
        if path not in coords:
            _,coords[path]=read_arm(folder)
        if hashes[a]!=hashes['base'] or path!=base:
            changed[a]=sites(case,coords[base],coords[path])
    ev=dict(package_chart_sha256=sha,source_sha256=handoff['source_sha256'],baseline_chart_sha256=guard['base'],
            changed_package_charts=handoff['changed_charts'],surface_sha256=handoff['surface_sha256'],rule=handoff['rule'],
            current_split=active,original_target_split=original_active,border_votes=drop,split_border_votes=split_drop,
            old_depth_border_votes=old_sky['drops']['border'],old_split_delta=old_split['paired']['split'],
            old_depth_border_delta=old_sky['paired']['border'],pack_sha256=hashes)
    row=dict(**scored,panel=group,split=case['split'],audit=ev,changed_sites=changed)
    save(OUT/group/'cases'/(name+'.json.gz'),row)
    if current or original_targets or drop or charts!=before:
        print(group,name,scored['paired'],flush=True)


def verify_base(rows):
    published=read(PACKAGE.parent.parent/'benchmark/reviews/floating-selected-package-01/conditions.json.gz')
    by={(r['panel'],r['id']):r for r in published}
    assert len(by)==len(rows)==84
    for row in rows:
        reference=by[row['panel'],row['id']]
        assert row['score']['base']==reference['package']
        assert row['audit']['pack_sha256']['base']==reference['pack_sha256']
        assert row['audit']['package_chart_sha256']==reference['chart_sha256']



def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=5)
    arg=cli.parse_args()
    a={(r['panel'],r['id']):r for r in read(OLD_MASKS/'conditions.json.gz')}
    b={(r['panel'],r['id']):r for r in read(OLD_SKY/'conditions.json.gz')}
    work=jobs()
    keys={(p,c['id']) for p,c in work}
    assert len(keys)==84 and keys==a.keys()==b.keys()
    with ProcessPoolExecutor(arg.workers) as pool:
        list(pool.map(replay,[(p,c,a[p,c['id']],b[p,c['id']]) for p,c in work]))
    rows=[read(OUT/g/'cases'/(c['id']+'.json.gz')) for g,c in work]
    verify_base(rows)
    cohorts={'84':rows,'six':[r for r in rows if r['panel']=='sky']}
    assert len(cohorts['six'])==6
    for ds in sorted({r['dataset'] for r in rows}):
        cohorts[ds]=[r for r in rows if r['dataset']==ds]
    save(REVIEW/'conditions.json.gz',rows)
    with patch.object(report,'ARMS',ARMS):
        save(REVIEW/'summary.json',{k:dict(count=len(v),arms=report.summarize_rows(v)) for k,v in cohorts.items()})


if __name__=='__main__':
    main()
