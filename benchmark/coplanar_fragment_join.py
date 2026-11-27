"""Join adjoining masks only when a repeating photographic texel map agrees across the seam."""
from itertools import combinations

import numpy as np
from PIL import ImageFilter

from benchmark.coplanar_fragment_audit import connected,edge_contact
from benchmark.wall_texture_trial import rebuild
from engine.pillar_seams import bounds,native_mask,orientation
from engine.plane_consensus import with_mask


def join_regions(image,parts,groups):
    masks={p['id']:native_mask(p,image.size) for p in parts}
    by_id={p['id']:p for p in parts}
    joined={k for group in groups for k in group}
    remaining=[p for p in parts if p['id'] not in joined]
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    for group in groups:
        largest=max(group,key=lambda k:masks[k].sum())
        parent=by_id[largest]
        if parent.get('normal_camera') is None:
            remaining.extend(by_id[k] for k in group)
            continue
        fields={k:v for k,v in parent.items() if k not in ('context_crop','source_crop')}
        merged=np.logical_or.reduce([masks[k] for k in group])
        part=with_mask(fields,merged,image.size,gray,np.ones(2))
        remaining.append(dict(part,id=largest+'-phase-union'))
    return remaining


def planes(rgb,mask,axis,p,eye,inv,cells):
    # Cut out one native pixel at the contour before fitting texture.
    padded=np.pad(mask,1)
    interior=mask&padded[1:-1,:-2]&padded[1:-1,2:]&padded[:-2,1:-1]&padded[2:,1:-1]
    y,x=np.nonzero(interior)
    if len(y)<80:
        return None
    if len(y)>3200:
        take=np.random.default_rng(19).choice(len(y),3200,replace=False)
        y,x=y[take],x[take]
    color=rgb[y,x].astype(float)
    intensity=color.mean(1)
    if intensity.std()<3.:
        return None
    ray=np.c_[x+.5,y+.5,np.ones(len(x))]@inv.T
    if np.min(abs(ray[:,axis]))<1.e-7:
        return None
    free=[k for k in range(3) if k!=axis]
    levels=[v[axis] for v in cells]
    tests=[]
    for k in range(min(levels)-2,max(levels)+3):
        level=k+int(eye[axis]>k+1)
        d=(level-eye[axis])/ray[:,axis]
        if d.min()<=0 or k<=eye[axis]<=k+1:
            continue
        uv=eye[free]+ray[:,free]*d[:,None]
        tile=np.floor(uv).astype(int)
        fold=tile.sum(1)%2
        texel=np.floor(uv*16).astype(int)%16
        code=16*texel[:,1]+texel[:,0]
        maps=[]; counts=[]
        for side in (0,1):
            take=fold==side
            n=np.bincount(code[take],minlength=256)
            maps.append(np.bincount(code[take],weights=intensity[take],minlength=256)/n.clip(1))
            counts.append(n)
        eligible=(counts[0]>=2)&(counts[1]>=2)
        if eligible.sum()<28:
            continue
        corr=correlation(maps[0][eligible],maps[1][eligible])
        template=(maps[0]*counts[0]+maps[1]*counts[1])/(counts[0]+counts[1]).clip(1)
        tests.append((corr,level,template,eligible))
    tests.sort(key=lambda o:o[0],reverse=True)
    if len(tests)<2:
        return None
    best,level,template,valid=tests[0]
    return dict(corr=best,level=level,template=template,valid=valid,margin=best-tests[1][0])


def correlation(a,b):
    if len(a)<8:
        return -1.
    a=a-a.mean(); b=b-b.mean()
    return float(a@b/max(1.e-7,np.linalg.norm(a)*np.linalg.norm(b)))


def photograph(image,parts,counts,camera,prior):
    if not prior or camera.get('P') is None:
        return prior,dict(groups=[])
    p,rot=orientation(camera,image.size)
    inv=np.linalg.inv(p[:,:3]);eye=-inv@p[:,3]
    masks=[native_mask(part,image.size) for part in parts]
    boxes=[bounds(m) if m.sum()>=80 else None for m in masks]
    rgb=np.asarray(image.convert('RGB'))
    fits={}
    edges=[]
    for a,b in combinations(range(len(parts)),2):
        if boxes[a] is None or boxes[b] is None:
            continue
        na=parts[a].get('normal_camera');nb=parts[b].get('normal_camera')
        if na is None or nb is None or np.dot(na,nb)<.985:
            continue
        axis=int(np.argmax(abs(rot.T@na)))
        if axis!=int(np.argmax(abs(rot.T@nb))):
            continue
        contact=edge_contact(masks[a],masks[b],boxes[a],boxes[b])
        if contact is None or len(contact[0])<24:
            continue
        for i in (a,b):
            if i not in fits:
                fits[i]=planes(rgb,masks[i],axis,p,eye,inv,prior)
        A,B=fits[a],fits[b]
        if not A or not B or A['level']!=B['level']:
            continue
        if min(A['corr'],B['corr'])<.70 or min(A['margin'],B['margin'])<.12:
            continue
        common=A['valid']&B['valid']
        if common.sum()<28:
            continue
        corr=correlation(A['template'][common],B['template'][common])
        same_color=abs(A['template'][common].mean()-B['template'][common].mean())<=8
        if corr>=.72 and same_color:
            edges.append((a,b))
    groups=[[parts[i]['id'] for i in component] for component in connected(edges)]
    if not groups:
        return prior,dict(groups=[])
    cells,detail=rebuild(image,counts,camera,join_regions(image,parts,groups))
    return cells,dict(groups=groups,**detail)
