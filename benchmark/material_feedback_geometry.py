"""Check whether a placed cube hides a better-supported visible textured plane.

Run features without labels; run oracle before the development rule and frozen evaluation.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
from pathlib import Path

from benchmark import typed_structures as typed
import numpy as np
from PIL import Image
from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import checked, digest, read
from engine import placed_materials
from engine.projected_appearance import Scene
from engine.thin_atlas_faces import _mips

HERE = Path(__file__).resolve().parents[1]
RELEASE = HERE/'benchmark/reviews/typed-structures-01'
OUT = HERE/'benchmark/reviews/material-feedback-geometry-01'
CACHE = HERE.parent/'typed-structure-scores/runs/typed-structures'


@lru_cache(maxsize=1)
def published():
    summary = read(RELEASE/'summary.json')
    sites = json.loads(gzip.decompress(checked(RELEASE/'sites.json.gz',summary['sites_sha256']).read_bytes()))
    rows = read(checked(RELEASE/'cases.json',summary['cases_sha256']))
    return sites, {row['id']: row for row in rows}


@lru_cache(maxsize=1)
def atlas_maps():
    atlas = typed.viewer.calibrated_atlas()
    return atlas, _mips(atlas)


def source_inputs(key, cache):
    inp = read(cache/'inputs'/(key+'.json'))
    sites, rows = published()
    site = sites[key]
    for field in ('source_hash','geometry_pack_sha256'):
        if inp[field] != site[field] or inp[field] != rows[key][field]:
            raise ValueError(f'{field} mismatch in {key}')
    if site['pack_sha256'] != rows[key]['pack_sha256']:
        raise ValueError(f'typed pack mismatch in {key}')
    cells = [tuple(p) for p in inp['cells']]
    ids = {tuple(row[:3]):row[3:] for row in site['rows']}
    if set(ids) != set(cells):
        raise ValueError(f'Changed cells in {key}')
    return inp, cells, ids


def fit_face(samples, colors, sprites, maps):
    indices, uv = map(np.concatenate, zip(*samples))
    if len(indices) < 12 or not len(sprites):
        return None
    sub = np.linspace(0,len(indices)-1,min(384,len(indices)),dtype=int)
    values, uv = colors[indices[sub]], uv[sub]
    center = np.median(values,axis=0)
    light = (values-center).mean(1)
    templates = maps[sprites].reshape(-1,256)
    observed = templates[:,uv]
    offset = np.median(observed, axis=1)
    varied = observed-offset[:,None]
    gain = np.clip((varied @ light) / np.maximum(1,(varied**2).sum(axis=1)),0,1.3)
    cost = np.abs(varied*gain[:,None]-light).mean(1)
    best = int(cost.argmin())
    return center, (templates[best]-offset[best])*gain[best]


def propose(image, camera, cells, names):
    if not cells or camera.get('P') is None:
        return []
    atlas,maps = atlas_maps()
    scene = Scene(image,[],camera,cells)
    ids = [names[c][0].removeprefix('minecraft:') for c in scene.cells]
    backed = [names[c][1] != 'fallback' for c in scene.cells]
    colors = scene.rgb.reshape(-1,3).astype(np.float32)
    yy,xx = np.indices(scene.shape)
    test = ((xx//3+yy//3)%2 == 1).ravel()
    samples = defaultdict(list)
    witnesses = defaultdict(set)
    def facekey(n,axis):
        c = scene.cells[n]
        return (ids[n],int(axis),c[axis],scene.eye[axis] > c[axis]+1)
    for n,indices in scene.owned.items():
        if not backed[n]:
            continue
        for axis in np.unique(scene.axes[indices]):
            part = indices[(scene.axes[indices]==axis)&~test[indices]]
            if len(part) < 3:
                continue
            key = facekey(n,axis)
            samples[key].append((part,scene.uv(part,scene.cells[n],axis)))
            witnesses[key].add(n)
    models = {}
    def render(n,axis,idx):
        key = facekey(n,axis)
        if key not in models:
            dirs = ('up',) if axis==1 and key[3] else ('down',) if axis==1 else ('north','south','east','west')
            sprites = [i for i,tile in enumerate(atlas.labels)
                       if any(key[0] in atlas.face_choices[d].get(tile,()) for d in dirs)]
            models[key] = fit_face(samples[key],colors,sprites,maps) if samples[key] else None
        model = models[key]
        if model is None:
            return None
        center,texture = model
        return np.abs(colors[idx]-center-texture[scene.uv(idx,scene.cells[n],axis),None]).mean(1)
    proposals=[]
    for n,idx in scene.owned.items():
        if not backed[n] or len(idx)<20:
            continue
        # At most 5% of the front cube may bound the silhouette. On the rest
        # there must already be a second cube at that pixel behind its face.
        if np.mean(scene.back[idx]>=0)<.95:
            continue
        visible = idx[test[idx]]
        compare = visible[scene.back[visible]>=0]
        if len(compare)<10:
            continue
        errors=[]
        for axis in np.unique(scene.axes[compare]):
            front = compare[scene.axes[compare]==axis]
            rear_pairs = np.column_stack((scene.back[front],scene.back_axis[front]))
            for rear,other_axis in np.unique(rear_pairs,axis=0):
                if not backed[rear]:
                    continue
                test_idx = front[(scene.back[front]==rear)&(scene.back_axis[front]==other_axis)]
                # Require an independently visible witness on the rear plane.
                if not (witnesses[facekey(rear,other_axis)]-{n}):
                    continue
                before = render(n,axis,test_idx)
                after = render(rear,other_axis,test_idx)
                if before is not None and after is not None:
                    errors.extend(zip(before.tolist(),after.tolist()))
        if len(errors)<10 or len(errors)<.75*len(visible):
            continue
        a,b=np.array(errors).T
        proposals.append(dict(cell=list(scene.cells[n]),samples=len(a),front=round(float(a.mean()),3),
                              rear=round(float(b.mean()),3),gain=round(float((a-b).mean()),3),
                              vote=round(float((b<a).mean()),4)))
    return proposals


def features(args):
    key,cache,work=args
    inp,cells,ids=source_inputs(key,cache)
    path=checked(Path(inp['rgb']),inp['source_hash'])
    with Image.open(path) as image:
        proposals=propose(image.convert('RGB'),inp['camera'],cells,ids)
    typed.compact(work/(key+'.json'),dict(id=key,source_hash=inp['source_hash'],
        geometry_pack_sha256=inp['geometry_pack_sha256'],pack_sha256=published()[1][key]['pack_sha256'],
        proposals=proposals))
    return key


# This menu uses only interpretable units: held-out source samples, RGB error
# advantage and fraction of samples won. Select on development buildings.
RULES = [(n,g,v) for n in (16,32) for g in (2,5,10,20) for v in (.65,.8,.9)]


def select(proposals,rule):
    n,g,v=rule
    eligible=[r for r in proposals if r['samples']>=n and r['gain']>=g and r['vote']>=v]
    # Evaluate one independent geometry change at a time. Two proposed removals
    # may rely on each other as visible evidence.
    return max(eligible,key=lambda r:(r['gain'],r['vote'],r['samples']),default=None)


def records(key,cache,work):
    props=read(work/(key+'.json'))
    base=published()[1][key]
    for field in ('source_hash','geometry_pack_sha256','pack_sha256'):
        if props[field]!=base[field]:
            raise ValueError(f'Feature binding changed: {key} {field}')
    inp,cells,ids=source_inputs(key,cache)
    ref=read(cache/'reference'/(key+'.json'))
    world=dict(zip(cells,map(tuple,ref['aligned']),strict=True))
    truth={tuple(v[:3]):v[3] for v in ref['truth']}
    optional=set(map(tuple,ref['optional']))
    return props['proposals'],world,truth,optional,base


def oracle(keys,cache,work):
    counts=Counter()
    for key in keys:
        props,world,truth,optional,_=records(key,cache,work)
        kinds=Counter('extra' if world[tuple(r['cell'])] not in truth else
                      'optional' if world[tuple(r['cell'])] in optional else 'required' for r in props)
        counts.update(kinds)
        counts['proposals']+=len(props)
        counts['frames_with_extra']+=int(kinds['extra']>0)
        counts['frames_with_proposal']+=int(bool(props))
    return counts


def train(keys,cache,work):
    leaderboard={rule: Counter() for rule in RULES}
    for key in keys:
        props,world,truth,_,_=records(key,cache,work)
        for rule in RULES:
            proposal=select(props,rule)
            if proposal:
                matched=world[tuple(proposal['cell'])] in truth
                leaderboard[rule]['correct_removed' if matched else 'extra_removed']+=1
    # Even if no rule generalizes, retain the best challenger that actually fires.
    active=[r for r,c in leaderboard.items() if sum(c.values())]
    def utility(r):
        counts=leaderboard[r]
        return (counts['extra_removed']-1.2*counts['correct_removed'], -counts['correct_removed'],r)
    if not active:
        raise ValueError('No candidate fires on development cases')
    winner=max(active,key=utility)
    return dict(rule=winner, scores=[dict(rule=r,**c) for r,c in leaderboard.items()])


def trial(args):
    key,cache,work,rule=args
    props,world,truth,optional,base=records(key,cache,work)
    row={k:base[k] for k in ('id','split','source_hash','geometry_pack_sha256','pack_sha256')}
    row['proposals']=len(props)
    proposal=select(props,rule)
    if not proposal:
        return dict(row,geometry=base['geometry'],result=base['result'])
    inp,cells,ids=source_inputs(key,cache)
    at=tuple(proposal['cell']); point=world[at]
    cells.remove(at)
    rgb=checked(Path(inp['rgb']),inp['source_hash'])
    with Image.open(rgb) as image:
        blocks,evidence=placed_materials.assign(image,inp['camera'],cells,atlas_maps()[0])
    payload,decoded=typed.typed_pack(blocks)
    positioned={world[p]:name for p,name in decoded.items()}
    shift,_=best_translation(set(positioned),set(truth))
    positioned={tuple(p[i]+shift[i] for i in range(3)):name for p,name in positioned.items()}
    assigned={tuple(world[p][i]+shift[i] for i in range(3)) for p,s in evidence.items() if s!='fallback'}
    import hashlib
    return dict(row,choice=proposal,removed='extra' if point not in truth else 'optional' if point in optional else 'required',
                extra_translation=list(shift),candidate_sha256=hashlib.sha256(payload).hexdigest(),
                geometry=typed.geometry_score(positioned,truth,optional),result=typed.score(positioned,truth,optional,assigned))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','oracle','train','evaluate'])
    parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--work',type=Path,default=HERE/'runs/material-feedback-geometry')
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    cases=published()[1]; keys=sorted(cases)
    groups={'fixed':[k for k in keys if cases[k]['shape_sha256']], 'train':[k for k in keys if cases[k]['split']=='train']}
    groups['other']=[k for k in keys if k not in groups['fixed']]
    groups['other_unseen']=[k for k in groups['other'] if k not in groups['train']]
    assert [len(groups[k]) for k in ('fixed','train','other','other_unseen')]==[104,72,100,28]
    args.out.mkdir(exist_ok=True,parents=True);args.work.mkdir(exist_ok=True,parents=True)
    if args.stage=='features':
        with ProcessPoolExecutor(args.workers) as pool:
            for _ in pool.map(features,[(k,args.cache,args.work) for k in keys]):
                pass
    elif args.stage=='oracle':
        typed.compact(args.out/'oracle.json',{k:oracle(v,args.cache,args.work) for k,v in groups.items()})
    elif args.stage=='train':
        typed.compact(args.out/'train.json',train(groups['train'],args.cache,args.work))
    else:
        rule=read(args.out/'train.json')['rule']
        with ProcessPoolExecutor(args.workers) as pool:
            rows=list(pool.map(trial,[(k,args.cache,args.work,rule) for k in keys]))
        typed.compact(args.out/'cases.json',rows)
        index={r['id']:r for r in rows}
        def pool(group):
            changed=[index[k] for k in group if 'choice' in index[k]]
            return dict(views=len(group), changed=len(changed), removed=dict(Counter(r['removed'] for r in changed)),
                        geometry=typed.pooled_scores([index[k]['geometry'] for k in group]),
                        result=typed.pooled_scores([index[k]['result'] for k in group]),
                        baseline_geometry=typed.pooled_scores([cases[k]['geometry'] for k in group]),
                        baseline_result=typed.pooled_scores([cases[k]['result'] for k in group]))
        typed.compact(args.out/'summary.json',dict(revision='5818a37',cases_sha256=digest(args.out/'cases.json'),
                                                  rule=rule,cohorts={k:pool(v) for k,v in groups.items()}))


if __name__=='__main__':
    main()

