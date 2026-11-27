"""Paired rectified/native atlas rulers on identical photographed wall pixels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import cv2
import numpy as np
from PIL import Image, ImageDraw
from benchmark.full_block_lattice import UNIT, atlas, rays
from engine.atlas_count import _shortlist
from engine.geometry import project

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
CONTROLS = ROOT.parent/'surface-depth-round-controls/benchmark/reviews/surface-depth-round-controls-01/controls.json'
OUT = ROOT/'runs/local-wall-rulers-native-support-01'
ARMS = ['crop_single', 'crop_shared', 'native_single', 'native_shared']
cv2.setNumThreads(1)


def read(path):
    return json.loads(Path(path).read_bytes())


def quad(P, axis, level, origin, pitch=1):
    xyz = np.full((4, 3), float(level))
    xyz[:, [i for i in range(3) if i != axis]] = origin + pitch * UNIT
    return project(P, xyz)


def support(part, shape):
    width,height=part['support']['size']
    mask = np.zeros((height,width), np.uint8)
    for y, a, b in part['support']['runs']:
        mask[y, a:b] = 1
    if mask.shape==shape:
        return mask
    y=((np.arange(shape[0])+.5)*height/shape[0]).astype(int)
    x=((np.arange(shape[1])+.5)*width/shape[1]).astype(int)
    return mask[np.ix_(y,x)]


def groups(P, eye, part, decision, shape):
    """Choose a contiguous three-tile window using only complete ordinary support."""
    axis, level = decision['axis'], decision['level']
    mask = support(part, shape)
    y, x = np.nonzero(mask[::4, ::4])
    if not len(x):
        return
    free = [i for i in range(3) if i != axis]
    ray = rays(P[:, :3], np.c_[x*4+.5, y*4+.5])
    uv = eye[free] + ray[:, free] * ((level-eye[axis])/ray[:,axis])[:,None]
    coords, counts = np.unique(np.floor(uv).astype(int), axis=0, return_counts=True)
    cells = {tuple(c): int(n) for c,n in zip(coords, counts) if n >= 10}
    choices = []
    for c in cells:
        for delta in ((0,1),(1,0)):
            triple = [tuple(np.array(c)+j*np.array(delta)) for j in range(3)]
            if all(t in cells for t in triple):
                choices.append((min(cells[t] for t in triple), triple))
    for _, trio in sorted(choices, reverse=True)[:140]:
        patches = []
        for cell in trio:
            q = quad(P, axis, level, np.array(cell))
            lo = np.floor(q.min(0)).astype(int)
            hi = np.ceil(q.max(0)).astype(int)
            if min(lo)<0 or np.any(hi>shape[::-1]) or np.prod(hi-lo)>80000:
                break
            x,y = np.meshgrid(np.arange(lo[0],hi[0]),np.arange(lo[1],hi[1]))
            xy = np.c_[x.ravel()+.5,y.ravel()+.5]
            r = rays(P[:,:3],xy)
            u = eye[free]+r[:,free]*((level-eye[axis])/r[:,axis])[:,None]-cell
            take = ((u>=0)&(u<1)).all(1)
            inside = mask[y.ravel()[take],x.ravel()[take]].astype(bool)
            if len(inside)<130 or inside.mean()<.96:
                break
            pts = xy[take][inside]
            patches.append(pts[np.linspace(0,len(pts)-1,min(384,len(pts)),dtype=int)])
        if len(patches)==3:
            return trio, patches


def correlation(pred, obs):
    p = pred-pred.mean(-1,keepdims=True)
    o = obs-obs.mean()
    return (p@o)/np.maximum(.001,np.linalg.norm(p,axis=-1)*np.linalg.norm(o))


def arm_scores(uv, gray, sprites):
    # Native response averages the four atlas samples under each pixel.
    addresses = np.floor(uv*16).astype(int)%16
    idx = addresses[:,:,1]*16+addresses[:,:,0]
    native = sprites[:,idx].mean(2)
    # Rectification bins exactly the same observations at canonical texels.
    center = np.floor(uv.mean(1)*16).astype(int)%16
    bin_id = center[:,1]*16+center[:,0]
    counts = np.bincount(bin_id,minlength=256)
    valid = counts>0
    observed = np.bincount(bin_id,gray,minlength=256)[valid]/counts[valid]
    return np.array([correlation(sprites[:,valid],observed),correlation(native,gray)])


def choose(P, rgb, part, d):
    eye = -np.linalg.solve(P[:,:3],P[:,3])
    group = groups(P,eye,part,d,rgb.shape[:2])
    if not group:
        return
    trio, patches = group
    axis,base = d['axis'],d['level']
    free = [i for i in range(3) if i!=axis]
    center = np.array(trio[0])+.5
    sub = np.array([[-.25,-.25],[.25,-.25],[-.25,.25],[.25,.25]])
    colors = [rgb[xy[:,1].astype(int),xy[:,0].astype(int)] for xy in patches]
    allowed = set().union(*(set(atlas().face_choices[k]) for k in ('north','south','east','west')))
    # Choose a held tile's sprite on its center stripe only. Its seam-side strips
    # are excluded from both the RGB shortlist and template selection.
    held_ray = rays(P[:,:3],patches[2])
    held_uv = eye[free]+held_ray[:,free]*((base-eye[axis])/held_ray[:,axis])[:,None]-trio[2]
    direction = int(np.flatnonzero(np.array(trio[1])-trio[0])[0])
    middle = np.abs(held_uv[:,direction]-.5)<.24
    if min(middle.sum(),(~middle).sum())<16:
        return
    shortlists=[]
    for pixels in [colors[0],colors[1],colors[2][middle]]:
        short,_ = _shortlist(pixels, np.ones(len(pixels),bool),atlas())
        shortlists.append([int(k) for k in short if atlas().labels[k] in allowed and atlas().spreads[k].mean()>2][:6])
    if not all(shortlists):
        return
    keys = list(dict.fromkeys(k for subset in shortlists for k in subset))
    choices=[np.array([keys.index(k)*4+j for k in subset for j in range(4)]) for subset in shortlists]
    # UV is expressed in world-axis order. The up texture axis is world Y.
    orientations = [0,2,4,6] if axis==2 else [1,3,5,7]
    sprites = atlas().patterns[np.ix_(keys,orientations)].reshape(-1,256)
    rr = [rays(P[:,:3],(xy[:,None,:]+sub).reshape(-1,2)).reshape(-1,4,3) for xy in patches]
    # Candidate levels retain the ordinary level as the first option.
    options, values = [], []
    for offset in (0,-1,1,-3,3):
        level = base+offset
        if (level-eye[axis])*(base-eye[axis])<=0:
            continue
        pivot = eye[free]+(center-eye[free])*(level-eye[axis])/(base-eye[axis])
        uvs = [eye[free]+r[:,:,free]*((level-eye[axis])/r[:,:,axis])[:,:,None] for r in rr]
        for pitch,phase in [(1,(0,0)),(1,(.25,0)),(1,(0,.25)),(.5,(0,0)),(2,(0,0))]:
            origin = (np.zeros(2) if pitch==1 else pivot-.5*pitch)+phase
            options.append((level,pitch,origin))
            patches_and_edges=list(zip(uvs,colors))+[(uvs[2][m],colors[2][m]) for m in (middle,~middle)]
            values.append([arm_scores((u-origin)/pitch,c.mean(1),sprites) for u,c in patches_and_edges])
    # options x patches x domain x templates. The third tile never participates in selection.
    scores = np.array(values)
    record = dict(surface=d['id'],axis=axis,baseline=base,train_tiles=np.array(trio).tolist(),arms={})
    r = rays(P[:,:3],np.mean(patches[2],axis=0)[None,:])[0]
    def held_quad(level,pitch,origin):
        uv = eye[free]+r[free]*(level-eye[axis])/r[axis]
        return quad(P,axis,level,origin+pitch*np.floor((uv-origin)/pitch),pitch)
    record['options']=[dict(level=level,pitch=pitch,phase=origin.tolist(),quad=held_quad(level,pitch,origin).tolist())
                       for level,pitch,origin in options]
    for k,name in enumerate([*ARMS,'native_mixed']):
        if name=='native_mixed':
            domain=1
            fits=[scores[:,j,domain][:,choices[j]] for j in range(2)]
            fit=np.mean([s.max(1) for s in fits],axis=0)
            i=fit.argmax()
            fitted=[int(choices[j][s[i].argmax()]) for j,s in enumerate(fits)]
            t=int(choices[2][scores[i,3,domain,choices[2]].argmax()])
            valid=float(scores[i,4,domain,t])
            fit_score=float(fit[i])
        else:
            domain,shared = divmod(k,2)
            fit = scores[:,:shared+1,domain,:].mean(1)[:,choices[0]]
            i,j = np.unravel_index(fit.argmax(), fit.shape)
            t=choices[0][j]
            valid = float(scores[i,2,domain,t])
            fit_score = float(fit[i,j])
            fitted=[int(t)]*(shared+1)
        level,pitch,origin = options[i]
        q = held_quad(level,pitch,origin)
        record['arms'][name] = dict(level=level,pitch=pitch,phase=origin.tolist(),quad=q.tolist(),
            texture=atlas().labels[keys[t//4]],turn=orientations[t%4],fit=fit_score,verify=valid,
            fitting_textures=[atlas().labels[keys[v//4]] for v in fitted],
            accepted=bool(fit_score>=.62 and valid>=.52),
            legal_plane=bool(pitch==1 and not np.any(origin)))
    return record


def inputs(f):
    dest=f['destination']
    if 'root' in dest:
        root=WORK/dest['root']
        return read(root/'surfaces.json')['planes'],read(root/'build-manifest.json')['placement']
    artifacts=f['replay']['artifacts']
    if 'surface_sha256' in artifacts:
        return read(WORK/artifacts['surface_sha256']['path'])['planes'],dest['historical_placement']
    original=read(WORK/artifacts['input_sha256']['path'])
    return original['parts'],dest['historical_placement']


def run(f):
    data=(WORK/f['rgb']['path']).read_bytes()
    assert sha256(data).hexdigest()==f['rgb']['sha256']
    from io import BytesIO
    rgb=np.array(Image.open(BytesIO(data)).convert('RGB'))
    parts,status=inputs(f)
    lookup={p['id']:p for p in parts if p.get('support',{}).get('runs')}
    walls=[d for d in status.get('plane_offsets',[]) if d['axis']!=1 and d['id'] in lookup]
    walls.sort(key=lambda d:-support(lookup[d['id']],rgb.shape[:2]).sum(dtype=np.int64))
    records=[]
    for d in walls[:8]:
        rec=choose(np.array(status['P']),rgb,lookup[d['id']],d)
        if rec:
            records.append(rec)
    path=OUT/(f['id']+'.json')
    path.write_text(json.dumps(dict(id=f['id'],dataset=f['dataset'],split=f['split'],source_sha256=f['rgb']['sha256'],
        pack_sha256=f['pack']['sha256'],walls=len(walls),records=records)))
    print(f['id'],len(walls),len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=['development','transfer','all'],default='all')
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    fs=[f for f in read(CONTROLS)['pilot']['frames'] if args.split=='all' or (f['split']=='train')==(args.split=='development')]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,fs))
