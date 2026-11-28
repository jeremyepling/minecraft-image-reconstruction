"""Separate photographed wall surfaces by relative depth and surface direction before matching tiles."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_mixed_wall_refit as prior
from benchmark.floating_verified_wall_expansion import BASE, PREFIX, ROOT, WORKTREES
from benchmark.floating_verified_wall_expansion import data, read, save, digest, jobs, combined, summarize
from engine.normal_regions import _axes, _unit
from engine.surface_fields import MODELS
from engine.surface_perimeter import _native_alpha

OUT = ROOT / 'runs/depth-before-atlas'
REVIEW = ROOT / 'benchmark/reviews/floating-depth-before-atlas-01'
ARMS = ('split', 'complete', 'visible-patches', 'verified')
OLD = WORKTREES / 'floating-mixed-wall-refit/runs/mixed-refit'
STRAIGHT = False


def source_fields(panel, case, src, size):
    group = Path(case['dataset']) / case['id']
    if panel == 'sky':
        folder = combined.assembly.SHARED / 'baseline/sky' / group / 'handoffs'
    elif panel == 'original':
        folder = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01' / group / 'handoffs'
        if not folder.exists():
            folder = prior.NATIVE.parent / 'fields' / group
    else:
        folder = src.parent / 'fields' / group
    rec = next(read(p) for p in [folder / 'receipt.json', folder.parent / 'receipt.json'] if p.exists())
    assert rec['source_sha256'] == case['variants']['sky']['source_sha256']
    fields, hashes = {}, {}
    for k in ('normal', 'depth'):
        path = folder / (k + '.npz')
        hashes[k] = digest(path)
        assert rec['model_sha256'][k] == MODELS[k][2]
        if 'fields_sha256' in rec:
            assert hashes[k] == rec['fields_sha256'][k]
        with np.load(path) as raw:
            fields[k] = cv2.resize(raw['value'].astype('f'), size)
    return _unit(fields['normal']), fields['depth'], hashes


def signed_residual(depth, mask):
    """Fit the common slope from neighboring same-direction pixels, rejecting depth jumps."""
    slope = []
    for axis in (1, 0):
        d = depth if axis else depth.T
        m = mask if axis else mask.T
        ok = m[:, 6:] & m[:, :-6] & m[:, 3:-3]
        gradient = (d[:, 6:] - d[:, :-6])[ok] / 6
        slope.append(float(np.median(gradient)) if len(gradient) else 0.)
    y, x = np.indices(depth.shape)
    return depth - slope[0] * x - slope[1] * y


def bands(values, depth_noise):
    """Split only histogram valleys between substantial, separated depth modes."""
    lo, hi = np.quantile(values, [.01, .99])
    if hi - lo < 3 * depth_noise:
        return []
    counts, edges = np.histogram(values, bins=48, range=(lo, hi))
    smoothed = cv2.GaussianBlur(counts.astype('f')[None], (9, 1), 1.5)[0]
    peaks = [i for i in range(len(counts)) if smoothed[i] >= max(smoothed[max(0,i-2):i+3])]
    peaks.sort(key=lambda p: -smoothed[p])
    chosen = []
    for p in peaks:
        if smoothed[p] < .07 * max(smoothed) or any(abs(edges[p] - edges[q]) < 2 * depth_noise for q in chosen):
            continue
        if all(min(smoothed[min(p,q):max(p,q)+1]) < .76 * min(smoothed[p],smoothed[q]) for q in chosen):
            chosen.append(p)
        if len(chosen) == 4:
            break
    ordered = sorted(chosen)
    return [edges[a + np.argmin(smoothed[a:b+1]) + 1] for a,b in zip(ordered,ordered[1:])]


def partition(rgb, mask, normal, depth):
    """Use signed depth inside each camera-normal family. Snap boundaries to adjacent image edges."""
    axes = _axes(normal, mask)
    directions = np.concatenate((axes, -axes))
    y, x = np.nonzero(mask)
    x0, x1, y0, y1 = x.min(), x.max()+1, y.min(), y.max()+1
    sl = np.s_[y0:y1,x0:x1]
    inside, camera, distance = mask[sl], normal[sl], depth[sl]
    dots = camera @ directions.T
    orient = np.argmax(dots, axis=2)
    seeds = np.zeros(inside.shape, 'int32')
    seeds[~inside] = 1
    meta = {}
    index = 1
    minimum = max(72, int(.0015 * len(x)))
    for k, direct in enumerate(directions):
        area = inside & (orient == k)
        if area.sum() < minimum:
            continue
        residual = signed_residual(distance, area)
        noise = max(.012 * abs(float(np.median(distance[inside]))), .7 * float(np.median(abs(distance - cv2.GaussianBlur(distance, (0,0), 5))[area])), 1.e-6)
        cuts = bands(residual[area], noise)
        labels = np.digitize(residual, cuts)
        for b in range(len(cuts)+1):
            core = cv2.erode((area & (labels==b)).astype('uint8'), np.ones((5,5),'uint8'))
            count, ids, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
            for i in range(1,count):
                if stats[i,cv2.CC_STAT_AREA] < minimum:
                    continue
                picked = ids == i
                index += 1
                seeds[picked] = index
                meta[index] = dict(normal_camera=_unit(np.median(camera[picked],axis=0)).tolist(),
                                   axis=int(k), depth_band=b, residual=float(np.median(residual[picked])),
                                   depth_noise=noise)
    if not meta:
        meta[2] = dict(normal_camera=_unit(np.median(camera[inside],axis=0)).tolist(),axis=-1,
                       depth_band=0,residual=0.,depth_noise=0.)
        seeds[inside] = 2
    # Each observed silhouette island has its own seed. Exterior pixels never become candidates.
    _, islands = cv2.connectedComponents(inside.astype('uint8'), connectivity=4)
    for i in np.unique(islands[inside]):
        region = islands == i
        if not (seeds[region] > 1).any():
            index = max(meta) + 1
            seeds[region] = index
            meta[index] = dict(normal_camera=_unit(np.median(camera[region],axis=0)).tolist(),
                               axis=-1,depth_band=0,residual=0.,depth_noise=0.)
    padded = cv2.copyMakeBorder(cv2.GaussianBlur(rgb[sl], (5,5), 1.1),1,1,1,1,cv2.BORDER_REPLICATE)
    result = cv2.watershed(padded, np.pad(seeds,1,constant_values=1))[1:-1,1:-1].copy()
    result[~inside] = 0
    for i in np.unique(islands[inside]):
        island = islands == i
        valid = island & (result > 1)
        if not valid.any():
            valid = island & (seeds > 1)
            result[valid] = seeds[valid]
        missing = island & ~valid
        if missing.any():
            _, keys = cv2.distanceTransformWithLabels((~valid).astype('uint8'),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
            lut = np.zeros(keys.max()+1,'int32')
            lut[keys[valid]] = result[valid]
            result[missing] = lut[keys[missing]]
    out = np.zeros(mask.shape,'uint16')
    out[sl] = result
    assert np.array_equal(out>0,mask)
    details = []
    for k in np.unique(out[mask]):
        yy,xx = np.nonzero(out == k)
        details.append(dict(id=int(k),pixels=len(xx),bbox=[int(v) for v in (xx.min(),yy.min(),xx.max()+1,yy.max()+1)],
                            center=[float(np.median(xx)),float(np.median(yy))],**meta[int(k)]))
    return out, details


def straight_edge_evidence(rgb):
    """Record long, independently detected photographic edges and their measured directions."""
    field=np.zeros((*rgb.shape[:2],2),'float32')
    lengths=np.zeros(rgb.shape[:2],'float32')
    edges=prior.garden.line_segments(cv2.GaussianBlur(rgb,(5,5),1.1).astype('uint8'))
    for line in sorted(edges,key=lambda e:np.linalg.norm(e[1]-e[0])):
        delta=line[1]-line[0]
        length=np.linalg.norm(delta)
        if length<max(22.,.045*max(rgb.shape[:2])):
            continue
        a,b=(tuple(v) for v in np.rint(line).astype(int))
        cv2.line(lengths,a,b,float(length),7)
        unit=delta/length
        cv2.line(field,a,b,(float(unit[0]),float(unit[1])),7)
    return field,lengths


def horizontal_pencils(rgb,vp):
    """The upright camera's vertical direction follows from two measured horizontal vanishing directions."""
    if vp is None:
        return None
    points=vp[:,:2]/vp[:,2:]
    center=np.asarray(rgb.shape[1::-1])/2
    rays=points-center
    candidates=[]
    for i,j in ((0,1),(0,2),(1,2)):
        f2=-rays[i]@rays[j]
        if not (.28*rgb.shape[0])**2<f2<(3*rgb.shape[0])**2 or rays[i,0]*rays[j,0]>=0:
            continue
        if any(abs(rays[t,1])>.32*abs(rays[t,0]) for t in (i,j)):
            continue
        mismatch=abs(points[i,1]-points[j,1])/rgb.shape[0]
        if mismatch<=.045:
            candidates.append((mismatch,i,j,np.sqrt(f2)))
    if not candidates:
        return vp
    _,i,j,f=min(candidates)
    a,b=np.column_stack((rays[[i,j]]/f,np.ones(2)))
    v=np.cross(a,b)
    vertical=np.r_[center*v[2]+v[:2]*f,v[2]]
    if abs(vertical[2])<1.e-7:
        return vp
    return np.stack((vp[i],vp[j],vertical))



def refine_adjacent(rgb, mask, normal, depth, coarse, geometry):
    """Use appearance contours as proposals and keep cuts with adjacent signed depth or normal turns."""
    appearance = prior.partition(rgb,mask,depth)
    encoded = np.where(mask, coarse.astype('int32')*(int(appearance.max())+1)+appearance,0)
    small = np.zeros(mask.shape,'int32')
    parents = [0]
    for code in np.unique(encoded[mask]):
        count, cc = cv2.connectedComponents((encoded == code).astype('uint8'),connectivity=4)
        for component in range(1,count):
            membership = cc==component
            small[membership] = len(parents)
            parents.append(int(coarse[membership][0]))
    y0,x0 = np.nonzero(mask)
    box = np.s_[y0.min():y0.max()+1,x0.min():x0.max()+1]
    ids = small[box]
    d = depth[box]
    normal = normal[box]
    appearance_rgb = cv2.GaussianBlur(rgb[box].astype('f'),(0,0),2)
    if STRAIGHT:
        line_dir,line_length=straight_edge_evidence(rgb[box])
    residual = signed_residual(d,mask[box])
    noise = max(.004*abs(float(np.median(d[mask[box]]))), 1.e-6)
    distances = [None] + [cv2.distanceTransform((ids==i).astype('uint8'),cv2.DIST_L2,5) for i in range(1,len(parents))]
    pairs = {}
    for axis in (0,1):
        a,b = (ids[:-1],ids[1:]) if axis==0 else (ids[:,:-1],ids[:,1:])
        y,x=np.nonzero((a!=b)&(a>0)&(b>0))
        for i,j in set(zip(a[y,x].tolist(),b[y,x].tolist())):
            subset=(a[y,x]==i)&(b[y,x]==j)
            yi,xi=y[subset],x[subset]
            pair=tuple(sorted((i,j)))
            pairs.setdefault(pair,[]).append(np.stack((yi,xi),axis=-1))
    edges = []
    roots = list(range(len(parents)))
    def root(i):
        while roots[i]!=i:
            roots[i]=roots[roots[i]]
            i=roots[i]
        return i
    for (i,j), pieces in pairs.items():
        by,bx=np.concatenate(pieces).T
        # Local normal to the source contour; each comparison stays on its own side.
        delta = distances[i] - distances[j]
        vx = cv2.Sobel(delta,cv2.CV_32F,1,0,ksize=3)[by,bx]
        vy = cv2.Sobel(delta,cv2.CV_32F,0,1,ksize=3)[by,bx]
        length = np.maximum(1.e-6,np.hypot(vx,vy))
        vx,vy=vx/length,vy/length
        record=dict(parts=(i,j),contacts=len(by),signed_depth=0.,normal_turn_deg=0.,color_delta=0.,agreement=0.,probes=0)
        for radius in (9,5,3):
            yp,ym=[np.rint(by+s*radius*vy).astype(int).clip(0,len(ids)-1) for s in (1,-1)]
            xp,xm=[np.rint(bx+s*radius*vx).astype(int).clip(0,ids.shape[1]-1) for s in (1,-1)]
            usable=(ids[yp,xp]==i)&(ids[ym,xm]==j)
            if usable.sum()<8:
                continue
            a=(yp[usable],xp[usable]); b=(ym[usable],xm[usable])
            jumps=residual[a]-residual[b]
            med=float(np.median(jumps))
            turns=np.arccos(np.clip(np.sum(normal[a]*normal[b],axis=1),-1,1))
            colors=np.linalg.norm(appearance_rgb[a]-appearance_rgb[b],axis=1)
            picks=np.linspace(0,len(jumps)-1,min(len(jumps),16),dtype=int)
            record['probe_pixels']=[[[int(ax+x0.min()),int(ay+y0.min())],[int(bx_+x0.min()),int(by_+y0.min())]]
                                    for ax,ay,bx_,by_ in zip(a[1][picks],a[0][picks],b[1][picks],b[0][picks])]
            record.update(probes=int(usable.sum()),signed_depth=med,agreement=float(np.mean(np.sign(jumps)==np.sign(med))),
                          normal_turn_deg=float(np.rad2deg(np.median(turns))),color_delta=float(np.median(colors)))
            break
        relief=(abs(record['signed_depth'])>=noise and record['agreement']>=.70)
        normal_turn=record['normal_turn_deg']>=12
        strong=parents[i]!=parents[j] or ((relief or normal_turn) and record['color_delta']>=8 and record['probes']>=8)
        if STRAIGHT:
            # A long, straight photographic discontinuity is useful even when the
            # relative-depth field smooths both sides to a single value.
            align=abs(line_dir[by,bx,0]*vx+line_dir[by,bx,1]*vy)
            votes=(line_length[by,bx]>0)&(align<.35)
            record['straight_border_samples']=int(votes.sum())
            strong |= votes.sum()>=15 and votes.mean()>=.40 and record['color_delta']>=12
        record['separate']=bool(strong)
        edges.append(record)
    # A weak local contact cannot absorb a different pair with a confirmed boundary.
    forbidden={tuple(e['parts']) for e in edges if e['separate']}
    members={i:{i} for i in range(1,len(parents))}
    for edge in sorted(edges,key=lambda e:-e['contacts']):
        i,j=map(root,edge['parts'])
        if edge['separate'] or i==j:
            continue
        if any(tuple(sorted((a,b))) in forbidden for a in members[i] for b in members[j]):
            continue
        roots[j]=i
        members[i]|=members.pop(j)
    result=np.asarray([root(i) for i in range(len(roots))],dtype='uint16')[small]
    details=[]
    by_id={e['id']:e for e in geometry}
    for k in sorted(members):
        yy,xx=np.nonzero(result==k)
        ev=by_id[parents[k]]
        details.append(dict(ev,id=k,pixels=len(xx),center=[float(np.median(xx)),float(np.median(yy))],
                            bbox=[int(v) for v in (xx.min(),yy.min(),xx.max()+1,yy.max()+1)],
                            normal_camera=_unit(np.median(normal[ids==k],axis=0)).tolist() if len(members[k])==1
                            else _unit(np.median(normal[np.isin(ids,list(members[k]))],axis=0)).tolist(),
                            source_fragments=sorted(members[k])))
    assert np.array_equal(result>0,mask)
    return result,details,edges



def load_source(panel,case):
    name = case['id']
    src = data(panel)[0]
    path = BASE / panel / 'large-contraction/charts/sky' / name / 'charts.json'
    packet = read(path)
    guard = read(prior.GUARD/panel/'audit'/(name+'.json.gz'))
    assert guard['base']==digest(path) and guard['source']==packet['source_sha256']
    flags = {k for k,v in guard['charts'].items() if v['peer_guard']}
    return src,packet,flags


def masks(panel,case):
    cv2.setNumThreads(1)
    src,packet,flags = load_source(panel,case)
    if not flags:
        return
    name = case['id']
    folder = OUT/panel/'source'/name
    folder.mkdir(parents=True,exist_ok=True)
    image = Path(case['variants']['sky']['image'])
    assert digest(image) == packet['source_sha256']
    rgb = np.array(Image.open(image).convert('RGB'))
    normal,depth,hashes = source_fields(panel,case,src,rgb.shape[1::-1])
    surfaces = read(src/'sky'/name/'source-handoff/surfaces.json')
    for plane in surfaces['planes']:
        if plane['id'] not in flags:
            continue
        alpha,(x,y,x1,y1) = _native_alpha(plane['support'],rgb.shape[1::-1])
        mask = np.zeros(rgb.shape[:2],bool)
        mask[y:y1,x:x1] = np.asarray(alpha)>0
        parts, details = partition(rgb,mask,normal,depth)
        coarse_count=len(details)
        parts,details,boundaries=refine_adjacent(rgb,mask,normal,depth,parts,details)
        Image.fromarray(parts).save(folder/(plane['id']+'-ids.png'))
        save(folder/(plane['id']+'.json'),dict(source_sha256=packet['source_sha256'],fields_sha256=hashes,
             plane=plane['id'],pixels=int(mask.sum()),parts=details,boundaries=boundaries,geometry_seeds=coarse_count,
             ids_sha256=digest(folder/(plane['id']+'-ids.png'))))
        print(panel,name,plane['id'],len(details),'surfaces',flush=True)


def mask_boundary(labels, inside):
    result=np.zeros(inside.shape,bool)
    for axis in (0,1):
        src,dst=(np.s_[:-1,:],np.s_[1:,:]) if axis==0 else (np.s_[:,:-1],np.s_[:,1:])
        edges=inside[src]&inside[dst]&(labels[src]!=labels[dst])
        result[src]|=edges
        result[dst]|=edges
    return result


def ownership(case, ids, boundaries=()):
    """Score predicted partitions only after pixel masks have been frozen."""
    frame=Path(case['frame'])
    labels=read(frame/'labels.json')
    with np.load(frame/'labels.npz') as raw:
        face_id=raw['face_id']
    groups=['background']
    table=np.zeros(int(face_id.max())+1,'int32')
    for face in labels['faces']:
        coordinate=float(np.median(np.asarray(face['corners_world'])@face['normal_world']))
        plane=face['direction']+':'+str(round(coordinate,3))
        if plane not in groups:
            groups.append(plane)
        table[face['id']]=groups.index(plane)
    truth=table[face_id]
    inside=ids>0
    pred_edges=mask_boundary(ids,inside)
    true_edges=mask_boundary(truth,inside)
    def matches(a,b):
        if not b.any():
            return 0
        return int((a & (cv2.distanceTransform((~b).astype('B'),cv2.DIST_L2,5)<=3)).sum())
    owner=[]
    counts={}
    for k in np.unique(ids[inside]):
        vals,n=np.unique(truth[ids==k],return_counts=True)
        hist=Counter({groups[v]:int(c) for v,c in zip(vals,n)})
        counts[int(k)]=hist
        owner.append(dict(part=int(k),pixels=int(n.sum()),planes=hist.most_common(5)))
    totals=Counter()
    for h in counts.values():
        totals.update(h)
    order=Counter()
    for e in boundaries:
        if not e['separate'] or e.get('agreement',0)<.70 or not e.get('signed_depth'):
            continue
        for (ax,ay),(bx,by) in e.get('probe_pixels',[]):
            a,b=groups[truth[ay,ax]],groups[truth[by,bx]]
            if 'background' in (a,b):
                continue
            da,za=a.split(':');db,zb=b.split(':')
            if da!=db:
                order['other_direction']+=1
                continue
            actual=float(za)-float(zb)
            if not actual:
                order['same_plane']+=1
                continue
            order['ordered_parallel']+=1
            order['correct_order']+=int(np.sign(actual)==np.sign(e['signed_depth']))
    matched=cv2.distanceTransform((~pred_edges).astype('B'),cv2.DIST_L2,5)<=3
    edge_kinds={}
    orientations=np.array([p.split(':')[0] for p in groups])
    for axis in (0,1):
        a,b=(np.s_[:-1,:],np.s_[1:,:]) if axis==0 else (np.s_[:,:-1],np.s_[:,1:])
        valid=inside[a]&inside[b]&(truth[a]!=truth[b])
        fa,fb=orientations[truth[a][valid]],orientations[truth[b][valid]]
        aligned=(matched[a]|matched[b])[valid]
        for da,db in set(zip(fa.tolist(),fb.tolist())):
            subset=(fa==da)&(fb==db)
            kind='/'.join(sorted((da,db)))
            counter=edge_kinds.setdefault(kind,Counter())
            counter['truth']+=int(subset.sum());counter['recovered']+=int(aligned[subset].sum())
    return dict(parts=owner,pixels=int(inside.sum()),dominant=sum(max(h.values()) for h in counts.values()),
                signed_order=order,boundary_direction_pairs=edge_kinds,
                native_planes={p:dict(pixels=v,largest_piece=max(h.get(p,0) for h in counts.values())) for p,v in totals.items()},
                edges=dict(predicted=int(pred_edges.sum()),truth=int(true_edges.sum()),matched_predicted=matches(pred_edges,true_edges),
                           matched_truth=matches(true_edges,pred_edges),tolerance_px=3))


def tangent_frame(rgb,mask,normal,vanishing):
    """Recover two orthogonal camera tangents on the piece from global source-image lines."""
    if vanishing is None:
        return None
    vp=vanishing[:,:2]/vanishing[:,2:]
    center=np.array(rgb.shape[1::-1])/2
    rays=vp-center
    focal2=[-rays[i]@rays[j] for i,j in ((0,1),(1,2),(2,0))]
    possible=[v for v in focal2 if (.28*rgb.shape[0])**2<v<(3*rgb.shape[0])**2]
    if not possible:
        return None
    focal=np.sqrt(np.median(possible))
    vectors=_unit(np.column_stack((rays/focal,np.ones(3))))
    dots=abs(vectors@normal)
    tangents=np.argsort(dots)[:2]
    if max(dots[tangents])>.42 or abs(vectors[tangents[0]]@vectors[tangents[1]])>.25:
        return None
    yy,xx=np.nonzero(mask)
    origin=np.array([np.median(xx)+.5,np.median(yy)+.5])
    basis=vp[tangents]-origin
    # Use right/down numbering for the local image frame before converting to atlas UVs.
    if abs(basis[0,1])/np.linalg.norm(basis[0])>abs(basis[1,1])/np.linalg.norm(basis[1]):
        tangents=tangents[::-1]
        basis=basis[::-1]
    scale=np.median(np.linalg.norm(basis,axis=1)*vectors[tangents,2])
    hs=[np.r_[vp[k],1.]*vectors[k,2]*np.sign(basis[i,i])/scale for i,k in enumerate(tangents)]
    h=np.column_stack([*hs,np.r_[origin,1.]])
    if abs(np.linalg.det(h))<1.e-9:
        return None
    uv=prior.project(np.linalg.inv(h),np.column_stack((xx+.5,yy+.5)))
    low=np.floor(uv.min(0)).astype(int)-2
    size=np.ceil(uv.max(0)).astype(int)-low+2
    if min(size)<12 or max(size)>2000 or np.prod(size)>1000000:
        return None
    h=h@np.array([[1.,0,low[0]],[0,1.,low[1]],[0,0,1.]])
    y,x=np.indices(size[::-1])
    xy=(prior.project(h,np.stack([x+.5,y+.5],axis=-1))-.5).astype('float32')
    return h,cv2.remap(rgb,xy,None,cv2.INTER_LINEAR),cv2.remap(mask.astype('B'),xy,None,cv2.INTER_NEAREST)


def native_grid_score(case,chart):
    """Check each accepted source tile on frontmost owners, keeping answers outside the fitter."""
    frame=Path(case['frame'])
    info=read(frame/'labels.json')
    faces={f['id']:f for f in info['faces']}
    blocks={b['id']:b['block_id'] for b in info['blocks']}
    with np.load(frame/'labels.npz') as raw:
        owner=raw['face_id']
    h=np.array(chart['H'])
    cells=chart['cells']
    uv=np.array([(c['u']+.5,c['v']+.5) for c in cells])
    xy=prior.project(h,uv).astype(int)
    ids=owner[xy[:,1],xy[:,0]]
    planes=Counter()
    correct=0
    materials=Counter()
    all_planes={}
    for cell,i in zip(cells,ids):
        if i not in faces:
            continue
        f=faces[i]
        coord=round(float(np.median(np.array(f['corners_world'])@f['normal_world'])),3)
        plane=f['direction']+':'+str(coord)
        planes[plane]+=1
        all_planes[i]=plane
        material=blocks[f['block']]
        materials[material]+=1
        correct+=int(cell['block_id'].removeprefix('minecraft:')==material.removeprefix('minecraft:'))
    if not planes:
        return dict(cells=len(cells),distinct_faces=0,correct_ids=0)
    primary=planes.most_common(1)[0][0]
    selected=[i for i,p in all_planes.items() if p==primary]
    quad=prior.project(np.linalg.inv(h),np.array([faces[i]['corners_px'] for i in selected]))
    x,y=quad.transpose(2,0,1)
    areas=abs(np.sum(x*np.roll(y,-1,axis=1)-y*np.roll(x,-1,axis=1),axis=1))*.5
    return dict(cells=len(cells),distinct_faces=len(all_planes),correct_ids=correct,materials=materials,planes=planes,
                dominant_ys=sorted({faces[i]['xyz_world'][1] for i in selected}),median_cells_per_dominant_face=float(np.median(areas)))



def build(panel,case):
    cv2.setNumThreads(1)
    src,packet,flags=load_source(panel,case)
    name=case['id']
    base=BASE/panel/'large-contraction/assembly'
    roots={'base':base}
    plans={k:[] for k in ARMS}
    audit={}
    if flags:
        image=Path(case['variants']['sky']['image'])
        assert digest(image)==packet['source_sha256']
        rgb=np.array(Image.open(image).convert('RGB'))
        counts={s['id']:s for s in read(src/'sky'/name/'source-handoff/counts.json')['surfaces']}
        lines=prior.garden.line_segments(rgb)
        vanishing=prior.garden.pencils(rgb,lines)
        if STRAIGHT:
            vanishing=horizontal_pencils(rgb,vanishing)
        for c in packet['charts']:
            cid=c['id']
            if cid not in flags:
                for plan in plans.values():
                    plan.append(c)
                continue
            folder=OUT/panel/'source'/name
            ids=np.asarray(Image.open(folder/(cid+'-ids.png')))
            metadata=read(folder/(cid+'.json'))
            assert metadata['source_sha256']==packet['source_sha256']
            assert metadata['ids_sha256']==digest(folder/(cid+'-ids.png'))
            centers=prior.project(np.asarray(c['H']),np.array([(p['u']+.5,p['v']+.5) for p in c['cells']])).astype(int)
            assignment=ids[centers[:,1],centers[:,0]]
            assert (assignment>0).all()
            selected={k:[] for k in ARMS}
            changes=[]
            whole=counts[cid]
            for part in metadata['parts']:
                k=part['id']
                region=ids==k
                rows=[p for p,j in zip(c['cells'],assignment) if j==k]
                sub=dict(c,id=f'{cid}~p{k}',cells=rows)
                measured=np.array(part['normal_camera'])
                options={}
                detail={}
                # Fit a fresh source period and phase in each plane frame. The parent
                # supplies tangent directions only when measured normals agree.
                frames={}
                if np.dot(measured,c['normal_camera'])>.92:
                    frames['wall tangents']=prior.rectify(whole['quad'],whole['evidence'].get('aspect'),rgb,region)
                frames['line tangents']=tangent_frame(rgb,region,measured,vanishing)
                for frame_kind,frame in frames.items():
                    if frame is None:
                        detail[frame_kind]=dict(reason='unresolved plane tangents')
                        continue
                    updated=dict(sub,normal_camera=measured.tolist())
                    found,ev=prior.region_ruler(updated,region,rgb,None,None,rectified=frame)
                    detail[frame_kind]=ev
                    for arm,chart in found.items():
                        quality=ev['evidence'][arm]['holdout']
                        rank=(len(quality),float(np.mean([v['corr'] for v in quality])))
                        if arm not in options or rank>options[arm][0]:
                            options[arm]=(rank,chart,frame_kind)
                if rows:
                    selected['split'].append(sub)
                for arm,option in [('complete','refit'),('visible-patches','visible-patches')]:
                    if option in options:
                        selected[arm].append(options[option][1])
                    elif rows:
                        selected[arm].append(sub)
                changes.append(dict(part=k,base_cells=len(rows),witness=detail,
                                    accepted={arm:dict(frame=v[2],cells=len(v[1]['cells'])) for arm,v in options.items()}))
            assert sum(len(c['cells']) for c in selected['split'])==len(c['cells'])
            # Use the complete-face proposal only when assembly can retain the altered
            # grid and the other sampled regions together. Ordinal relief gives no
            # metric origin for detached charts.
            complete=selected['complete']
            witnessed=[x for x in changes if 'refit' in x['accepted']]
            accepted=False
            if witnessed:
                from benchmark.relative_face_assembly import assemble
                groups=assemble(complete,min_edges=1,edge_pixels=2.5,coplanar_pixels=.5)['groups']
                accepted=bool(groups) and set(groups[0]['origins'])=={s['id'] for s in complete}
            selected['verified']=complete if accepted else [c]
            for arm,v in selected.items():
                plans[arm].extend(v)
            old=np.asarray(Image.open(OLD/panel/'source'/name/'region-ids.png'))
            assert np.array_equal(old>0,ids>0)
            grid_scores={}
            for arm,option in [('complete','refit'),('visible-patches','visible-patches')]:
                picked={f'{cid}~p{p["part"]}' for p in changes if option in p['accepted']}
                grid_scores[arm]={ch['id']:native_grid_score(case,ch) for ch in selected[arm] if ch['id'] in picked}
            audit[cid]=dict(fitted=changes,verified=accepted,source=metadata,grid_scores=grid_scores,
                            score_parts=ownership(case,ids,metadata['boundaries']),score_previous=ownership(case,old))
    for arm,charts in plans.items():
        if not flags or charts==packet['charts']:
            roots[arm]=base
            continue
        folder=OUT/panel/arm
        save(folder/'charts/sky'/name/'charts.json',dict(packet,charts=charts))
        with patch.object(combined,'inputs',data):
            combined.run((panel,case,folder/'charts',folder/'assembly'))
        roots[arm]=folder/'assembly'
    score=summarize([case],roots,PREFIX)['rows'][0]
    sha={a:digest(r/PREFIX/name/'reconstruction.zip') for a,r in roots.items()}
    components={}
    for arm,r in roots.items():
        if flags:
            rec=read(r/PREFIX/name/'score.json')
            components[arm]=dict(sizes=rec['component_sizes'],selected=list(rec['selected_origins']))
    save(OUT/panel/'rows'/(name+'.json.gz'),dict(**score,panel=panel,split=case['split'],audit=audit,
         source_sha256=packet['source_sha256'],packs=sha,components=components))
    if flags:
        print(name,{k:(v['geometry']['strict']['correct'],v['geometry']['strict']['extras']) for k,v in score['score'].items()},flush=True)


def publish(work):
    rows=[read(OUT/g/'rows'/(c['id']+'.json.gz')) for g,c in work]
    save(REVIEW/'conditions.json.gz',rows)
    from benchmark import floating_mixed_wall_report as report
    groups={'all':rows,'six':[r for r in rows if r['panel']=='sky']}
    for dataset in sorted({r['dataset'] for r in rows}):
        groups[dataset]=[r for r in rows if r['dataset']==dataset]
    with patch.object(report,'ARMS',ARMS):
        save(REVIEW/'summary.json',{k:dict(conditions=len(v),arms=report.summarize_rows(v)) for k,v in groups.items()})



def mode(straight):
    global OUT, REVIEW, STRAIGHT
    STRAIGHT=straight
    tag='depth-before-atlas'+('-straight' if straight else '')
    OUT=ROOT/'runs'/tag
    REVIEW=ROOT/'benchmark/reviews'/('floating-'+tag+'-01')


def run_job(panel,case,*,straight,stage):
    mode(straight)
    (masks if stage=='masks' else build)(panel,case)


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=3)
    cli.add_argument('--stage',choices=('masks','build','all'),default='all')
    cli.add_argument('--straight',action='store_true')
    arg = cli.parse_args()
    work=jobs()
    mode(arg.straight)
    with ProcessPoolExecutor(arg.workers) as pool:
        if arg.stage in ('masks','all'):
            list(pool.map(partial(run_job,straight=arg.straight,stage='masks'),*zip(*work)))
        if arg.stage in ('build','all'):
            list(pool.map(partial(run_job,straight=arg.straight,stage='build'),*zip(*work)))
    if arg.stage in ('build','all'):
        publish(work)
