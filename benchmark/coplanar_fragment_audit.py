"""Measure adjoining fragments of each recorded coplanar, edge-connected visible surface."""
import argparse
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked,read,save,digest
from engine.pillar_seams import native_mask,bounds

CONTROL=Path('../visible-error-budget/runs/primary-production')
REVIEW=Path('benchmark/reviews/coplanar-fragments-01')
THRESHOLDS={'inclusive':(.70,.02,32),'standard':(.80,.05,64),'strict':(.90,.10,128)}


def edge_contact(a,b,boxa,boxb):
    """Unique pixels in A with a B pixel within four native pixels along a cardinal axis."""
    h,w=a.shape
    l,t=max(boxa[0],boxb[0])-4,max(boxa[1],boxb[1])-4
    r,bt=min(boxa[2],boxb[2])+4,min(boxa[3],boxb[3])+4
    if l>=r or t>=bt:
        return None
    l,t,r,bt=max(0,l),max(0,t),min(w,r),min(h,bt)
    a,b=a[t:bt,l:r],b[t:bt,l:r]
    a,b=a&~b,b&~a
    edge=np.zeros_like(a)
    for shift in range(1,5):
        edge[:-shift] |= a[:-shift]&b[shift:]
        edge[shift:] |= a[shift:]&b[:-shift]
        edge[:,:-shift] |= a[:,:-shift]&b[:,shift:]
        edge[:,shift:] |= a[:,shift:]&b[:,:-shift]
    y,x=np.nonzero(edge)
    if len(x)<8:
        return None
    return y+t,x+l


def connected(edges):
    adjacency=defaultdict(set)
    for a,b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    remaining=set(adjacency)
    groups=[]
    while remaining:
        seed=min(remaining)
        component={seed}
        waiting=[seed]
        while waiting:
            nxt=adjacency[waiting.pop()]-component
            component.update(nxt)
            waiting.extend(nxt)
        remaining-=component
        groups.append(sorted(component))
    return groups


def load_case(job):
    s,f,h,row=job
    if s in ('new','extra'):
        from training.visible_labels import visible_labels
        from benchmark.mask_blocks import registered,decoded_pack
        photo,parts,counts,cam=framed.photo(s,f,h)
        root=framed.NEW_INPUTS if s=='new' else framed.transfer.BUILDS
        path=root/f['directory']/'captures'/f['source_id']
        pose=read(checked(path.with_suffix('.json'),f['files']['camera.json']))
        blocks=read(checked(path.with_name(path.name+'-blocks.json'),f['files']['world-readback.json']))
        labels,ids,_=visible_labels(pose,{tuple(b[k] for k in 'xyz'):b for b in blocks},(0,0,0))
        truth,_,reg,hashes=framed.references('old',f,framed.NATIVE_INPUTS[s],framed.TRAINING_DATA,framed.DEPTH,
                                            dict(old=root))
        assert hashes==row['input_hashes']
        raw=registered(set(decoded_pack(checked(CONTROL/s/f['id']/'reconstruction.zip',row['pack_sha256']))),reg)
        placed=framed.spacing.shift(raw,row['translation'])
    else:
        photo,parts,_,cam,_,labels,ids,truth,_,placed,_=read_view(CONTROL,s,f,h,row)
    assert len(truth-placed)==row['relative_geometry']['missed']
    return photo,parts,labels,ids,truth-placed


def inspect(job):
    s,f,h,row=job
    photo,parts,labels,ids,missed=load_case(job)
    faces=labels['faces']
    luts=np.zeros(len(faces)+1,dtype='int32')
    coords={}
    by_surface=defaultdict(set)
    for face in faces:
        i,sid=face['id'],face['surface_id']
        coords[i]=tuple(face['xyz_world'])
        luts[i]=sid
        by_surface[sid].add(coords[i])
    target=luts[ids]
    sizes=np.bincount(target.ravel())
    masks=[native_mask(p,photo.size) for p in parts]
    candidates=defaultdict(list)
    metadata={}
    for i,mask in enumerate(masks):
        if not mask.any():
            continue
        pixels=np.bincount(target[mask],minlength=len(sizes))
        sid=int(np.argmax(pixels[1:])+1)
        n=int(pixels[sid])
        purity=n/mask.sum()
        fraction=n/max(1,sizes[sid])
        if purity<.70 or fraction<.02 or n<32:
            continue
        candidates[sid].append(i)
        metadata[i] = (purity,fraction,n,mask & (target==sid))
    result=dict(id=f['id'],split=s,reference_hashes=row['reference_hashes'],rgb_sha256=row['rgb_sha256'],
                photo_hash=hashlib.sha256(photo.tobytes()).hexdigest(),pack_sha256=row['pack_sha256'],
                visible_surfaces=len(by_surface),missed=len(missed),thresholds={})
    for name,(purity,fraction,n) in THRESHOLDS.items():
        counted=set()
        on_piece=set()
        touching=set()
        pieces=[]
        for sid,pool in candidates.items():
            keep=[i for i in pool if metadata[i][0]>=purity and metadata[i][1]>=fraction and metadata[i][2]>=n]
            if len(keep)<2:
                continue
            edges={}
            for a,b in combinations(keep,2):
                ma,mb=metadata[a][3],metadata[b][3]
                if np.count_nonzero(ma&mb)>.10*min(metadata[a][2],metadata[b][2]):
                    continue
                ab=edge_contact(ma,mb,bounds(ma),bounds(mb))
                if ab is not None:
                    ba=edge_contact(mb,ma,bounds(mb),bounds(ma))
                    edges[a,b]=set(np.unique(ids[ab])) | (set(np.unique(ids[ba])) if ba is not None else set())
            for component in connected(edges):
                m=np.logical_or.reduce([metadata[i][3] for i in component])
                blocks={coords[z] for z in np.unique(ids[m])}
                border={coords[z] for (a,b),zs in edges.items() if a in component and b in component for z in zs}
                counted.update(by_surface[sid]&missed)
                on_piece.update(blocks&missed)
                touching.update(border&missed)
                pieces.append(dict(surface=sid,mask_ids=[parts[i]['id'] for i in component],
                                   on_face=len(by_surface[sid]&missed),on_pieces=len(blocks&missed),
                                   at_seam=len(border&missed)))
        result['thresholds'][name]=dict(groups=pieces,surfaces=len({g['surface'] for g in pieces}),
                                on_surface=len(counted),on_pieces=len(on_piece),at_seam=len(touching))
    return result


def run(args):
    meta=read(CONTROL/'summary.json')
    index={r['id']:r for r in read(checked(CONTROL/'cases.json',meta['cases_sha256']))}
    jobs=[(s,f,h,index[f['id']]) for s,f,h in framed.cases(args.splits)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(inspect,jobs))
    grouped={'fixed':framed.EVALUATION,'development':['train'],'alternate':['extra'],'new':['new']}
    totals={}
    for cohort,splits in grouped.items():
        sub=[r for r in rows if r['split'] in splits]
        if not sub:
            continue
        totals[cohort]={}
        for name in THRESHOLDS:
            stat=Counter()
            for row in sub:
                rec=row['thresholds'][name]
                stat.update({k:rec[k] for k in ('surfaces','on_surface','on_pieces','at_seam')})
                stat.update(views=1,affected_views=int(bool(rec['groups'])),total_surfaces=row['visible_surfaces'],
                            missed=row['missed'],pieces=sum(len(g['mask_ids']) for g in rec['groups']))
            totals[cohort][name]=stat
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'cases.json',rows)
    save(args.out/'summary.json',dict(control='edb10a3',source_cases=meta['cases_sha256'],
                                     cases_sha256=digest(args.out/'cases.json'),groups=totals))
    print(totals)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--splits',nargs='+',default=list(framed.SIZES))
    parser.add_argument('--out',type=Path,default=REVIEW)
    run(parser.parse_args())
