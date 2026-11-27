"""Test a second Block Counter pass using the final typed, visible cubes on one wall."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

from benchmark import concrete_shadow_boundaries as surfaces, typed_structures as typed
from benchmark.texture_camera_eval import checked, digest, read, save
from engine.geometry import project, transform
from engine.pillar_seams import native_mask
from engine.projected_appearance import Scene
from engine.thin_atlas_faces import _corr, _mips

OUT = Path('benchmark/reviews/material-feedback-grid-01')


@lru_cache(maxsize=1)
def typed_inputs():
    release = typed.REVIEW
    meta = read(release/'summary.json')
    return json.loads(gzip.decompress(checked(release/'sites.json.gz',meta['sites_sha256']).read_bytes()))


@lru_cache(maxsize=1)
def textures():
    atlas = surfaces.viewer.calibrated_atlas()
    sprites = defaultdict(list)
    for i, tile in enumerate(atlas.labels):
        names = {n for wall in ('north','south','east','west') for n in atlas.face_choices[wall].get(tile,[])}
        for name in names:
            sprites['minecraft:'+name].append(i)
    return _mips(atlas), sprites


def plane_coords(scene, xy, axis, level):
    rays = np.column_stack((xy,np.ones(len(xy)))) @ scene.inv.T
    t = (level-scene.eye[axis]) / rays[:,axis]
    free = [d for d in range(3) if d!=axis]
    return scene.eye[free] + t[:,None]*rays[:,free]


def material_strength(scene, idx, axis, level, name):
    maps, sprites = textures()
    if name not in sprites or len(idx)<14:
        return None
    y,x = np.divmod(idx,scene.shape[1])
    uv = plane_coords(scene, np.column_stack((x,y))*scene.step+.5, axis,level)
    colors = scene.rgb[y,x].mean(-1)
    if colors.std()<1.4:
        return None
    scores=[]
    for scale in (1.,.5,2.):
        texels = np.floor(uv*scale*16).astype(int)%16
        flat = texels[:,1]*16+texels[:,0]
        keys,inverse,counts=np.unique(flat,return_inverse=True,return_counts=True)
        if len(keys)<10:
            return None
        value=(np.bincount(inverse,weights=colors)/counts).astype(np.float32)
        choices=maps[sprites[name]][:,:2]
        scores.append(max(float(_corr(choices[..., (keys//16+dy)%16*16+(keys+dx)%16],value).max())
                          for dx,dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1))))
    return scores[0],scores[0]-max(scores[1:])


def propose(image, parts, counts, camera, site):
    output = {}
    if camera.get('P') is None or not site['rows']:
        return output
    names = {tuple(r[:3]):r[3:] for r in site['rows']}
    scene=Scene(image,[],camera,names,step=2)
    cells=np.asarray(scene.cells)
    strength={}
    for part,record in zip(parts,counts['surfaces'],strict=True):
        if part['id']!=record['id']:
            raise ValueError('Surface inventory differs from count rows')
        initial=record.get('dimensions',{}).get('blocks',[None,None])
        quad=record.get('quad')
        if None not in initial or quad is None:
            continue
        mask=native_mask(part,image.size)[::scene.step,::scene.step].ravel()
        hits=np.flatnonzero(mask & (scene.owner>=0))
        if len(hits)<32:
            continue
        num,axes=scene.owner[hits],scene.axes[hits].astype(int)
        planes=cells[num,axes]+(scene.eye[axes]>cells[num,axes]+1)
        keys,pop=np.unique(np.column_stack((axes,planes)),axis=0,return_counts=True)
        winner=int(np.argmax(pop))
        axis,level=map(int,keys[winner])
        if axis==1 or pop[winner]<.70*mask.sum():
            continue
        native=np.flatnonzero(mask)
        y,x=np.divmod(native,scene.shape[1])
        coords=plane_coords(scene,np.column_stack((x,y))*scene.step+.5,axis,level)
        endpoints=np.quantile(coords,[.002,.998],axis=0)
        rounded=np.rint(endpoints)
        if abs(endpoints-rounded).max()>.28:
            continue
        span=(rounded[1]-rounded[0]).astype(int)
        if min(span)<1 or max(span)>40:
            continue
        chart=plane_coords(scene,np.asarray(quad),axis,level)
        order=int(np.argmax(abs(chart[1]-chart[0])))
        dims=[int(span[order]),int(span[1-order])]
        if any(old is not None and old!=new for old,new in zip(initial,dims)):
            continue
        contributing=[]
        for n in np.unique(num[(axes==axis)&(planes==level)]):
            ids=scene.owned[n]
            face=ids[scene.axes[ids]==axis]
            inside=face[mask[face]]
            if len(inside)<14 or len(inside)<.75*len(face):
                continue
            cell=scene.cells[n]
            if names[cell][1]=='fallback':
                continue
            signature=(int(n),axis)
            if signature not in strength:
                strength[signature]=material_strength(scene,face,axis,level,names[cell][0])
            value=strength[signature]
            if value:
                contributing.append((cell,value))
        supported=[c for c,(cor,lead) in contributing if cor>=.46 and lead>=.085]
        free=[d for d in range(3) if d!=axis]
        support_uv={tuple(c[d] for d in free) for c in supported}
        corners={(int(i),int(j)) for i in (rounded[0,0],rounded[1,0]-1)
                 for j in (rounded[0,1],rounded[1,1]-1)}
        texture=len(supported)>=2 and len(supported)>=.5*len(contributing)
        verified_axes=[]
        for dim in (order,1-order):
            observed={c[dim] for c in support_uv}
            lo, hi = int(rounded[0,dim]), int(rounded[1,dim])-1
            verified_axes.append(texture and lo in observed and hi in observed and
                                 len(observed & set(range(lo,hi+1))) >= .6*(hi-lo+1))
        output[part['id']]=dict(dims=dims,eligible=len(contributing),supported=len(supported),
            texture=texture,corners=texture and corners<=support_uv, ends=verified_axes,
            plane=[axis,level],snap=float(abs(endpoints-rounded).max()))
    return output


def pair(parts,size,labels,ids):
    lut=np.zeros(len(labels['faces'])+1,int)
    groups=defaultdict(list)
    for face in labels['faces']:
        sid=face['surface_id']
        lut[face['id']]=sid
        groups[sid].append(face)
    truth=lut[ids]
    areas=np.bincount(truth.ravel())
    pairs=[]
    for p in parts:
        mask=native_mask(p,size)
        n=np.bincount(truth[mask],minlength=len(areas))
        iou=n/np.maximum(1,mask.sum()+areas-n)
        pairs.extend((float(iou[k]),p['id'],int(k)) for k in np.flatnonzero(iou>=.5) if k)
    result={}
    used=set()
    for _,key,sid in sorted(pairs,reverse=True):
        if key not in result and sid not in used:
            result[key]=groups[sid]
            used.add(sid)
    return result


def expected(faces,quad):
    axis=int(np.argmax(abs(np.asarray(faces[0]['normal_world']))))
    if axis==1 or quad is None:
        return None
    free=[d for d in range(3) if d!=axis]
    pos=np.array([f['xyz_world'] for f in faces])[:,free]
    span=np.ptp(pos,axis=0)+1
    face=max(faces,key=lambda f:f['visible_pixels'])
    w=np.array(face['corners_world'])[:,free]; px=np.array(face['corners_px'])
    projection=transform(w)@np.linalg.inv(transform(px))
    q=project(projection,np.asarray(quad))
    across=np.argmax(abs(q[1]-q[0]))
    return [int(span[across]),int(span[1-across])]


def frame(job):
    split,fr,_,_=job
    key=fr['id']
    site=typed_inputs()[key]
    if site['source_hash']!=fr['files']['rgb.png']:
        raise ValueError(f'Wrong source in typed export for {key}')
    image,parts,counts,camera=surfaces.viewer.structure_stages(job,surfaces.baseline()[key])
    proposals=propose(image,parts,counts,camera,site)
    labels,ids=surfaces.visible_faces(split,fr)
    pairs=pair(parts,image.size,labels,ids)
    rows=[]
    for r in counts['surfaces']:
        name=r['id']
        actual=expected(pairs[name],r.get('quad')) if name in pairs else None
        init=r.get('dimensions',{}).get('blocks',[None,None])
        if actual is not None or name in proposals:
            rows.append(dict(surface=name,current=init,actual=actual,matched=name in pairs,proposal=proposals.get(name)))
    return dict(id=key,split=split,source_hash=site['source_hash'],typed_pack=site['pack_sha256'],rows=rows)


def summarize(frames):
    result={}
    for mode in ('geometry','texture','ends','corners'):
        out=Counter()
        for f in frames:
            for row in f['rows']:
                p,a,b=row['proposal'],row['actual'],row['current']
                c=list(b)
                if p:
                    for axis in range(2):
                        eligible=(p['ends'][axis] if mode=='ends' else mode=='geometry' or p[mode])
                        if c[axis] is None and eligible:
                            c[axis]=p['dims'][axis]
                use=c!=b
                if use:
                    out['proposals']+=1
                    if a is None:
                        out['unpaired_or_wrong_orientation']+=1
                    elif None not in c:
                        out['added_correct' if c==a else 'added_incorrect']+=1
                    else:
                        out['partial_only']+=1
                if a is not None:
                    out['wall_charts']+=1
                    out['baseline_correct']+=b==a
                    out['baseline_returned']+=None not in b
                    out['result_correct']+=c==a
                    out['result_returned']+=None not in c
                    out['oracle_correct_proposals']+=bool(p and p['dims']==a)
                    for axis,label in enumerate(('across','down')):
                        out[label+'_baseline_known']+=b[axis] is not None
                        out[label+'_baseline_correct']+=b[axis]==a[axis]
                        if b[axis] is None and c[axis] is not None:
                            out[label+('_added_right' if c[axis]==a[axis] else '_added_wrong')]+=1
        result[mode]=dict(out)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--set',choices=['development','confirmation'],required=True)
    ap.add_argument('--workers',type=int,default=5)
    args=ap.parse_args()
    splits=['train'] if args.set=='development' else [s for s in [*typed.SUITE.framed.SIZES,'shelter'] if s!='train']
    jobs=list(typed.SUITE.jobs(splits))
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(frame,jobs))
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/(args.set+'-cases.json'),rows)
    fixed=set(typed.SUITE.framed.EVALUATION)
    groups=dict(development=rows) if args.set=='development' else dict(fixed=[r for r in rows if r['split'] in fixed],other28=[r for r in rows if r['split'] not in fixed])
    summary={k:summarize(rr) for k,rr in groups.items()}
    save(OUT/(args.set+'-summary.json'),summary)
    print(summary,flush=True)


if __name__=='__main__':
    main()

