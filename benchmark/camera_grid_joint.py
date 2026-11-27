"""Source-only alternating camera phase, metric scale and visible-wall level fit."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, SOURCE, COHORTS, WORKTREES, digest, frames, read, save

RUN = ROOT / 'runs/camera-grid-joint-01'
REVIEW = ROOT / 'benchmark/reviews/camera-grid-joint-01'
BASE = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01'
CONTROL = WORKTREES / 'camera-model-controls/benchmark/reviews/camera-model-controls-01'
MOGE = WORKTREES / 'camera-moge-surface-first/runs/camera-moge-surface-first-01'
GEO = WORKTREES / 'camera-geocalib/runs/camera-geocalib-01'
MOGE_SHA = '24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0'
GEO_SHA = '86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac'


def entries():
    frozen = read(CONTROL / 'frames.json')
    assert len(frozen) == 288
    sources = {e['id']: e for e in frames(SOURCE, COHORTS)}
    for row in frozen:
        if row['repeat_supported'] or read(GEO/'matched-control/metric'/(row['id']+'.json'))['camera']['P'] is not None:
            continue
        entry = sources[row['id']]
        assert entry['rgb_sha256'] == row['rgb_sha256']
        yield entry


def intrinsic(focal, size):
    return np.array([[focal[0], 0, size[0]/2], [0, focal[1], size[1]/2], [0, 0, 1]])


def rotations(linear, focal, geo, size):
    R = np.linalg.solve(intrinsic([focal]*2, size), linear)
    yield 'moge', linear, R
    up = np.array(geo['up_cam'])
    up /= np.linalg.norm(up)
    x = R[:, 0] - up * (up @ R[:, 0])
    x /= np.linalg.norm(x)
    corrected = np.column_stack((x, up, np.cross(up, x)))
    yield 'geocalib', intrinsic(geo['focal_xy_px'], size) @ corrected, corrected


def mask(support, size):
    w,h = support['size']
    image = np.zeros((h,w), np.uint8)
    for y,a,b in support['runs']:
        image[y,a:b] = 1
    return np.array(Image.fromarray(image).resize(size, Image.Resampling.NEAREST)).astype(bool)


def signature(part):
    return hashlib.sha256(json.dumps(part['support'],sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def patches(anchors):
    # Different candidate masks can cover the same motif. Reject touching boxes before splitting.
    result, boxes = [], []
    for a in sorted(anchors, key=lambda a:(-a['corr'],a['surface'])):
        if a['axis']==1:
            continue
        quad = np.array(a['quad'])
        lo, hi = quad.min(0)-2, quad.max(0)+2
        if any(np.all(lo < b) and np.all(hi > c) for c,b in boxes):
            continue
        boxes.append((lo,hi))
        result.append(a)
    return result


def wall_evidence(linear, R, anchors, parts, xyz, valid, normals, size):
    h,w = valid.shape
    observed = np.zeros_like(valid)
    prepared = {}
    for part in parts:
        if part.get('support'):
            m = mask(part['support'], (w,h))
            observed |= m
            prepared[part['id']] = m
    points = xyz @ R
    floor = valid & ~observed & ((normals @ R[:,1]) > .85)
    floor[:int(.74*h)] = False
    if floor.sum()<60:
        return None
    height = -np.median(points[:,:,1][floor])
    if height<=0:
        return None
    walls = {}
    for a in anchors:
        key, ax = a['surface'], a['axis']
        if key in walls or key not in prepared:
            continue
        m = prepared[key] & valid
        if m.sum()<25 or abs(np.median(normals[m],axis=0) @ R[:,ax])<.65:
            continue
        # A bottom edge with immediately adjacent floor pixels and no depth jump.
        bottom = h-1-np.argmax(m[::-1],axis=0)
        contacts = []
        for col in np.flatnonzero(m.any(0))[::3]:
            row = bottom[col]
            if row+2<h and floor[row+2,col] and abs(xyz[row,col,2]/xyz[row+2,col,2]-1)<.05:
                contacts.append([(col+.5)*size[0]/w,(row+1)*size[1]/h])
        walls[key] = (ax,float(np.median(points[:,:,ax][m])),contacts)
    return height,walls


def solve(linear, height, walls, anchors, seeds):
    available = [a for a in anchors if a['surface'] in walls]
    if len(available)<2:
        return []
    groups = sorted({a['surface'] for a in available},
                    key=lambda k:(-sum(a['surface']==k for a in available), k))
    # Most populated source wall supplies the fit. All other walls are withheld.
    fit = [a['surface']==groups[0] for a in available]
    qs = np.array([a['quad'] for a in available])
    axes = np.array([a['axis'] for a in available])
    rays = np.concatenate((qs,np.ones((*qs.shape[:2],1))),axis=2) @ np.linalg.inv(linear).T
    rays /= np.take_along_axis(rays,axes[:,None,None],axis=2)
    offsets = np.rint(np.array([np.array(a['xyz'])-np.array(a['xyz'])[0] for a in available]))
    model = np.array([walls[a['surface']][1] for a in available])
    idx = np.flatnonzero(fit)
    verify = np.flatnonzero(~np.array(fit))
    candidates = []
    seeds = [f * multiplier for f in seeds.get(groups[0], []) for multiplier in (.5, 1., 2.)]
    for seed in seeds:
        scale = seed
        rel = rays * (scale*model[:,None,None])
        phase = -np.angle(np.mean(np.exp(2j*np.pi*rel[idx]),axis=(0,1)))/(2*np.pi)
        eye = phase.copy()
        eye[1] += round(height*scale-eye[1])
        initial = np.rint(eye[axes]+scale*model).astype(int)
        for _ in range(3):
            def lattice(levels):
                rel = rays*(levels-eye[axes])[:,None,None]
                return np.rint((rel+eye-offsets).mean(1))[:,None,:]+offsets
            alternatives = initial[:,None]+np.array([-1,0,1])
            errors = []
            for ls in alternatives.T:
                cell = lattice(ls)
                residual = (cell-eye) - rays*(ls-eye[axes])[:,None,None]
                errors.append(np.mean(residual**2,axis=(1,2))+.02*(ls-eye[axes]-scale*model)**2)
            # Independent test surfaces get their level solely from the depth field.
            levels = np.rint(eye[axes]+scale*model).astype(int).clip(initial-1,initial+1)
            levels[idx] = alternatives[idx, np.argmin(np.mean(np.asarray(errors)[:,idx], axis=1))]
            cells = lattice(levels)
            # Solve eye and uniform model scale with fixed snapped wall-grid assignments.
            A,B = [],[]
            for i in idx:
                ax = axes[i]
                for rr,cc in zip(rays[i],cells[i]):
                    mat = np.eye(3)
                    mat[:,ax] -= rr
                    A.extend(np.column_stack((mat,np.zeros(3))))
                    B.extend(cc-rr*levels[i])
                v = np.zeros(4); v[ax]=1; v[3]=model[i]
                A.append(v); B.append(levels[i])
            A.append([0,1,0,-height]); B.append(0)
            sol = np.linalg.lstsq(A,B,rcond=None)[0]
            eye = np.clip(sol[:3],phase+[-.5,round(height*seed-phase[1])-.75,-.5],
                                      phase+[.5,round(height*seed-phase[1])+.75,.5])
            scale = float(np.clip(sol[3],seed*.88,seed*1.12))
        cells = lattice(levels)
        projected = (cells-eye) @ linear.T
        if np.any(projected[:,:,2]<=.02):
            continue
        uv = projected[:,:,:2]/projected[:,:,2,None]
        error = np.linalg.norm(uv-qs,axis=2)
        held_px = float(np.quantile(error[verify],.9)) if len(verify) else 1e4
        contact_errors = []
        overrides = {}
        for k,(ax,dist,contacts) in walls.items():
            matching = [i for i,a in enumerate(available) if a['surface']==k]
            if not matching:
                continue
            level = int(levels[matching[0]])
            if any(levels[i]!=level for i in matching):
                break
            overrides[k]=[ax,level]
            if len(contacts)>=4:
                ray = np.column_stack((contacts,np.ones(len(contacts)))) @ np.linalg.inv(linear).T
                cy = eye[1] + ray[:,1]*(level-eye[ax])/ray[:,ax]
                contact_errors.append(float(np.quantile(abs(cy),.75)))
        else:
            floor_error = min(contact_errors,default=1e4)
            fit_px = float(np.quantile(error[idx],.9))
            # Both a separate wall of complete native sprites and a floor contact must agree.
            accepted = len(verify)>=2 and held_px<=3 and fit_px<=3 and floor_error<=.16
            P = np.column_stack((linear,-linear@eye))
            candidates.append(dict(camera=dict(P=P.tolist(),scale_status='supported',camera_source='joint-grid',
                    reason='source-only camera and visible wall lattice hypothesis'),overrides=overrides,
                    fit_p90_px=fit_px,held_p90_px=held_px,ground_error_blocks=floor_error,
                    fit_n=len(idx),held_n=len(verify),accepted=bool(accepted),scale=scale,
                    rank=float(fit_px+min(held_px,100)+10*min(floor_error,10)),
                    fit_wall=groups[0],native_quads=qs.tolist(),reprojected=uv.tolist(),fit=fit))
    return candidates


def predict(e):
    d = MOGE/e['dataset']/e['id']
    p,meta = read(d/'proposal.json'),read(d/'field.json')
    source = Path(e['frame'])/'rgb.png'
    geo = read(GEO/'proposals'/(e['id']+'.json'))
    assert digest(source)==p['source_sha256']==meta['source_sha256']==geo['rgb_sha256']==e['rgb_sha256']
    assert meta['checkpoint_sha256']==MOGE_SHA and geo['weights_sha256']==GEO_SHA
    assert meta['revision']=='e50ffda41565591092adea54c6ac83d6212e1e23' and geo['upstream_commit']=='1dedb29f94fc11b896c5f572005978aeebccd67f'
    assert meta['crop'] is None and meta['tokens']==600
    base = BASE/e['dataset']/e['id']
    assert read(base/'receipt.json')['source_sha256']==e['rgb_sha256']
    parts = read(base/'handoffs/placement-input.json.gz')['parts']
    anchors = patches(p.get('anchors',[]))
    candidates = []
    with np.load(d/'field.npz') as f:
        size = tuple(meta['native_size'])
        assert tuple(f['native_size'])==size==Image.open(source).size
        assert f['points'].shape==f['normal'].shape==(*f['mask'].shape,3)
        assert list(f['mask'].shape[::-1])==meta['resized_size']==[448,280]
        assert np.isclose(meta['focal_px'],f['focal_px']) and np.isclose(meta['z_shift'],f['z_shift'])
        xyz = f['points'].copy()
        xyz[:,:,2] += f['z_shift']
        valid = f['mask'] & (xyz[:,:,2]>.01)
        factors = {s['surface']: [s['factor']] for s in p.get('scales',[]) if s['kind']=='whole-sprite'}
        if p.get('linear') and len(anchors)>=2 and factors:
            for name,L,R in rotations(np.array(p['linear']),f['focal_px'],geo,size):
                evidence = wall_evidence(L,R,anchors,parts,xyz,valid,f['normal'].astype(float),size)
                if evidence:
                    for c in solve(L,*evidence,anchors,factors):
                        c.update(rotation=name, supports={part['id']:signature(part) for part in parts if part['id'] in c['overrides']})
                        candidates.append(c)
    candidates.sort(key=lambda c:c['rank'])
    keep = []
    for c in candidates:
        if not any(np.allclose(c['camera']['P'],other['camera']['P']) and c['overrides']==other['overrides'] for other in keep):
            keep.append(c)
        if len(keep)==2:
            break
    result = dict(id=e['id'],source_sha256=e['rgb_sha256'],field_sha256=digest(d/'field.npz'),
                  candidate_count=len(candidates),anchors=len(anchors),best=keep,accepted=bool(keep and keep[0]['accepted']))
    if len(keep)==2 and keep[1]['accepted'] and abs(keep[0]['rank']-keep[1]['rank'])<.5:
        result['accepted']=False
    save(RUN/e['id']/'choices.json',result)
    return e['id'],len(candidates),result['accepted']


if __name__=='__main__':
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=['train','held'])
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(predict,[e for e in entries() if (e['split']=='train')==(args.split=='train')]):
            print(*row,flush=True)
