"""Compare direct and connected source-supported strict cells at a frozen ruler plane."""
import argparse
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from benchmark.local_wall_rulers import CONTROLS, ROOT, WORK, atlas, groups, inputs, quad, rays, read, support
from benchmark.local_wall_ruler_exports import entries
from benchmark.expanded_baseline import save
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.camera_horizon_replay import replay
from engine import visible_grid, visible_placement
from engine.atlas_count import _shortlist
import numpy as np
from PIL import Image, ImageDraw

OLD = ROOT.parent/'local-atlas-wall-rulers/runs/local-wall-rulers-native-support-01'
OUT = ROOT/'runs/wall-ruler-local-transfer-01'
REVIEW = ROOT/'benchmark/reviews/wall-ruler-local-transfer-01'
EXTENSION = os.environ.get('MCIR_LOCAL_RULER_EXTENSION')=='1'
if EXTENSION:
    CONTROLS=ROOT.parent/'wall-ruler-transfer-controls/benchmark/reviews/wall-ruler-transfer-controls-01/extension.json'
    OUT=ROOT/'runs/wall-ruler-local-transfer-extension-01'
    OLD=OUT/'candidates'


def frozen_candidates(f):
    from benchmark import local_wall_rulers
    OLD.mkdir(parents=True,exist_ok=True)
    if f['baseline']['camera_usable']:
        with patch.object(local_wall_rulers,'OUT',OLD):
            local_wall_rulers.run(f)
    else:
        save(OLD/(f['id']+'.json'),dict(id=f['id'],records=[],source_sha256=f['rgb']['sha256']))


SUB = np.array([[-.25,-.25],[.25,-.25],[-.25,.25],[.25,.25]])


def chart(P, eye, axis, level, xy):
    ray = rays(P[:,:3], xy)
    free = [i for i in range(3) if i != axis]
    return eye[free] + ray[:,free]*((level-eye[axis])/ray[:,axis])[:,None]


def footprint(P, eye, axis, level, cell, mask, rgb):
    corners = quad(P, axis, level, np.array(cell))
    lo, hi = np.floor(corners.min(0)).astype(int), np.ceil(corners.max(0)).astype(int)
    if min(lo)<0 or any(hi>mask.shape[::-1]) or np.prod(hi-lo)>80000:
        return
    x,y = np.meshgrid(np.arange(lo[0],hi[0]), np.arange(lo[1],hi[1]))
    xy = np.c_[x.ravel()+.5,y.ravel()+.5]
    uv = chart(P,eye,axis,level,xy)-cell
    inside = ((uv>=0)&(uv<1)).all(1)
    if inside.sum()<130 or not mask[y.ravel()[inside],x.ravel()[inside]].all():
        return
    xy = xy[inside]
    xy = xy[np.linspace(0,len(xy)-1,min(384,len(xy)),dtype=int)]
    subs = chart(P,eye,axis,level,(xy[:,None,:]+SUB).reshape(-1,2)).reshape(-1,4,2)-cell
    return subs, rgb[xy[:,1].astype(int),xy[:,0].astype(int)]


def match(patch, row, axis):
    """Choose a side texture on the center stripe and verify independent seam-side stripes."""
    uv,rgb=patch
    middle=np.abs(uv.mean(1)[:,0]-.5)<.24
    if min(middle.sum(),(~middle).sum())<16:
        return False
    lib=atlas()
    allowed=set().union(*(lib.face_choices[k] for k in ('north','south','east','west')))
    short,_=_shortlist(rgb[middle],np.ones(middle.sum(),bool),lib)
    indices=[list(lib.labels).index(row['texture'])]
    indices.extend(int(k) for k in short if lib.labels[k] in allowed and lib.spreads[k].mean()>2)
    turns=[0,2,4,6] if axis==2 else [1,3,5,7]
    sprites=lib.patterns[np.ix_(list(dict.fromkeys(indices)),turns)].reshape(-1,256)
    def score(sprites,offset,which):
        ij=np.floor((uv[which]+offset)*16).astype(int)%16
        observed=rgb[which].mean(1)
        template=sprites[:,ij[:,:,1]*16+ij[:,:,0]].mean(2)
        template-=template.mean(1,keepdims=True)
        observed-=observed.mean()
        return (template@observed)/np.maximum(.001,np.linalg.norm(template,axis=1)*np.linalg.norm(observed))
    train=score(sprites,0,middle)
    winner=train.argmax()
    if train[winner]<.62:
        return False
    selected=sprites[winner:winner+1]
    held=score(selected,0,~middle)[0]
    if held<.52:
        return False
    return all(held-score(selected,phase,~middle)[0]>=.03 for phase in ([.25,0],[-.25,0],[0,.25],[0,-.25]))


def regions(f):
    key=f['id']
    original=read(OLD/(key+'.json'))
    raw=(WORK/f['rgb']['path']).read_bytes()
    assert sha256(raw).hexdigest()==original['source_sha256']==f['rgb']['sha256']
    rgb=np.array(Image.open(WORK/f['rgb']['path']).convert('RGB'))
    parts,status=inputs(f)
    parts={p['id']:p for p in parts}
    P=np.array(status['P'])
    eye=-np.linalg.solve(P[:,:3],P[:,3]) if original['records'] else None
    records=[]
    overlay=Image.fromarray(rgb)
    draw=ImageDraw.Draw(overlay)
    for r in original['records']:
        proposal=r['arms']['native_single']
        if not (proposal['accepted'] and proposal['legal_plane']):
            continue
        axis,level=r['axis'],proposal['level']
        part=parts[r['surface']]
        mask=support(part,rgb.shape[:2])
        _,patches=groups(P,eye,part,dict(axis=axis,level=r['baseline']),rgb.shape[:2])
        direct=set()
        for xy in [patches[0],patches[2]]:
            cs,ns=np.unique(np.floor(chart(P,eye,axis,level,xy)).astype(int),axis=0,return_counts=True)
            c=tuple(cs[ns.argmax()].tolist())
            if ns.max()/len(xy)>=.80 and footprint(P,eye,axis,level,c,mask,rgb) is not None:
                direct.add(c)
        connected=direct.copy()
        tested=direct.copy()
        queue=deque(sorted(direct))
        while queue:
            c=queue.popleft()
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                neighbor=c[0]+dx,c[1]+dy
                if neighbor in tested:
                    continue
                tested.add(neighbor)
                tile=footprint(P,eye,axis,level,neighbor,mask,rgb)
                if tile is not None and match(tile,proposal,axis):
                    connected.add(neighbor)
                    queue.append(neighbor)
        for cell in sorted(connected):
            pts=[tuple(p) for p in quad(P,axis,level,np.array(cell))]
            draw.line(pts+[pts[0]], fill='#ffc04d' if cell in direct else '#31e9ea',width=2)
        records.append(dict(surface=r['surface'],axis=axis,level=level,front=int(np.sign(eye[axis]-level)),
                            direct=sorted(direct),connected=sorted(connected),tested=len(tested)))
    out=OUT/key
    out.mkdir(parents=True,exist_ok=True)
    overlay.save(out/'source.jpg',quality=87)
    save(out/'regions.json',dict(source_sha256=original['source_sha256'],records=records))
    print('regions',key,sum(len(r['connected']) for r in records),flush=True)


class SameStrictHandoff(Exception):
    pass


def build(task):
    f,e,arm=task
    key=f['id']
    folder=OUT/key/arm
    records=read(OUT/key/'regions.json')['records']
    selection='connected' if arm in ('overlap','whole') else arm
    by_id={r['surface']:r for r in records if arm!='noop' and r[selection]}
    get_carriers,normal_blocks=visible_placement._carriers,visible_placement._blocks
    ids=[]
    events=[]
    previous=read(OUT/key/'connected/case.json') if EXTENSION and arm=='overlap' else None
    def carriers(image,parts,masks,P):
        ids[:]=[p['id'] for p in parts]
        result=get_carriers(image,parts,masks,P)
        if arm=='whole':
            # Replay the previous whole-wall carrier replacement verbatim.
            selected={r['surface']:r for r in records}
            for name,mask in zip(ids,masks,strict=True):
                if name not in selected:
                    continue
                r=selected[name]
                plane=r['axis'],r['level'],r['front']
                old=next((v[1] for v in result if v[0] is mask),None)
                if old!=plane:
                    result=[v for v in result if v[0] is not mask]
                    result.insert(0,(mask,plane,'source atlas wall grid'))
                    events.append(dict(surface=name,old=old,new=plane))
        return result
    def blocks(masks,carriers,P,cache):
        strict,resolved,details=normal_blocks(masks,carriers,P,cache)
        if arm=='whole':
            return strict,resolved,details
        removals={}
        additions=set()
        for index,(name,mask) in enumerate(zip(ids,masks,strict=True)):
            if name not in by_id:
                continue
            r=by_id[name]
            axis,level,front=r['axis'],r['level'],r['front']
            plane=axis,level,front
            if plane not in cache:
                cache[plane]=visible_grid._cell_map(P,mask.shape,plane)
            occupied=cache[plane].occupied(mask)
            requested=set()
            for cell in r[selection]:
                xyz=list(cell)
                xyz.insert(axis,level-int(front>0))
                requested.add(tuple(xyz))
            permitted=requested & occupied
            removed=set()
            old,_=visible_grid._carrier(mask,carriers)
            if arm=='overlap' and old is not None and old!=plane and permitted:
                # Remove an old strict vote only when its complete source footprint
                # is inside the newly certified cells on this same surface.
                cellmap=cache[plane]
                free=[i for i in range(3) if i!=axis]
                pairs={tuple(xyz[i] for i in free) for xyz in permitted}
                offered=np.array([tuple(pair) in pairs for pair in cellmap.cells])
                region=np.zeros(mask.size,dtype=bool)
                region[cellmap.valid_pixels]=offered[cellmap.inverse]
                removed=cache[old].occupied(region.reshape(mask.shape)&mask)
                removals[index]=removed
            additions.update(permitted)
            events.append(dict(surface=name,requested=len(requested),strict_supported=len(permitted),old_removed=len(removed),
                               new_strict=len(permitted-strict)))
        if removals:
            strict=set()
            for index,mask in enumerate(masks):
                old,_=visible_grid._carrier(mask,carriers)
                if cache.get(old) is not None:
                    strict.update(cache[old].occupied(mask)-removals.get(index,set()))
        if previous is not None and len(events)==len(previous['events']):
            fields=('surface','requested','strict_supported','new_strict')
            same=all(tuple(a[k] for k in fields)==tuple(b[k] for k in fields)
                     for a,b in zip(events,previous['events'],strict=True))
            if same and all(e['old_removed']==0 for e in events):
                raise SameStrictHandoff
        return strict|additions,resolved,details
    folder.mkdir(parents=True,exist_ok=True)
    _,status=inputs(f)
    try:
        with patch.object(visible_placement,'_carriers',carriers),patch.object(visible_placement,'_blocks',blocks):
            if e['dataset']=='fixed-104':
                result=fixed.replay_placement(fixed.inference_inputs(e))
                (folder/'reconstruction.zip').write_bytes(result.pack)
                case,_=fixed.score_result(e,result)
            else:
                camera=dict(P=status['P'],scale_status='supported',reason=status['reason'],camera_source=status['camera_source'])
                case=replay(e,camera,WORK/f['replay']['root'],folder)
    except SameStrictHandoff:
        save(folder/'case.json',dict(case=previous['case'],events=events,reused='connected'))
        return
    save(folder/'case.json',dict(case=case,events=events))
    print('export',key,arm,case['geometry']['strict']['correct'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['candidates','regions','export','noop','whole','castle-overlap'])
    p.add_argument('--split',choices=['development','transfer'],required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    pframes=read(CONTROLS)['frames'] if EXTENSION else read(CONTROLS)['pilot']['frames']
    frames=[f for f in pframes if (f['split']=='train')==(args.split=='development')]
    with ProcessPoolExecutor(args.workers) as pool:
        if args.stage=='candidates':
            assert EXTENSION
            list(pool.map(frozen_candidates,frames))
        elif args.stage=='regions':
            list(pool.map(regions,frames))
        else:
            items=entries()
            tasks=[]
            for f in frames:
                records=read(OUT/f['id']/'regions.json')['records']
                if args.stage=='castle-overlap' and EXTENSION:
                    if any(r['connected'] for r in records):
                        tasks.append((f,items[f['id']],'overlap'))
                    continue
                if args.stage=='castle-overlap':
                    if f['id']=='test-castle-01-ashlar-high-right':
                        tasks.append((f,items[f['id']],'overlap'))
                    continue
                if args.stage=='whole':
                    if records:
                        tasks.append((f,items[f['id']],'whole'))
                    continue
                if args.stage=='noop':
                    tasks.append((f,items[f['id']],'noop'))
                    continue
                for arm in ('direct','connected'):
                    if arm=='connected' and all(r['connected']==r['direct'] for r in records):
                        continue
                    if not any(r[arm] for r in records):
                        continue
                    tasks.append((f,items[f['id']],arm))
            list(pool.map(build,tasks))
