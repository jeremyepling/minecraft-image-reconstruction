"""Measure integer pillar depths independently on alternating photographed block rows."""
from engine import floating_grid_policy as floating
import numpy as np
from PIL import Image,ImageFilter

from engine.atlas_phase import ATLAS_SHA256
from engine.pillar_seams import native_mask,orientation,split_pillars
from engine.thin_atlas_faces import _match,_mips,_corr
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def sample_mask(region,size):
    mask=native_mask(region,size)
    padded=np.pad(mask,1)
    inner=mask & padded[1:-1,:-2] & padded[1:-1,2:] & padded[:-2,1:-1] & padded[2:,1:-1]
    y,x=np.nonzero(inner)
    if len(x)>4000:
        take=np.random.default_rng(5681).choice(len(x),4000,replace=False)
        y,x=y[take],x[take]
    return y,x


def predict(image,region,camera,cells,atlas,maps,*,audit=False):
    y,x=sample_mask(region,image.size)
    if len(x)<60:
        return None
    p,rotation=orientation(camera,image.size)
    axis=int(np.argmax(abs(rotation.T@region['normal_camera'])))
    if axis==1:
        return None
    free=[i for i in range(3) if i!=axis]
    yi=free.index(1)
    inv=np.linalg.inv(p[:,:3])
    eye=-inv@p[:,3]
    rays=np.column_stack((x+.5,y+.5,np.ones(len(x))))@inv.T
    if np.min(abs(rays[:,axis]))<1.e-7:
        return None
    rgb=np.asarray(image.convert('RGB'))[y,x].astype(float)
    levels=[q[axis] for q in cells]
    plans=[]
    for level in range(min(levels)-3,max(levels)+4):
        plane=level+int(eye[axis]>level+1)
        if level<=eye[axis]<=level+1 or np.min((plane-eye[axis])/rays[:,axis])<=0:
            continue
        xyz=eye+(plane-eye[axis])*rays/rays[:,[axis]]
        texel=np.floor(xyz[:,free]*16).astype(int)
        ijk=np.floor(xyz).astype(int)
        ijk[:,axis]=level
        if np.quantile(ijk[:,1],.98)-np.quantile(ijk[:,1],.02)<1:
            continue
        phase=texel%16
        keys=16*phase[:,1]+phase[:,0]
        folds=ijk[:,1]%2
        metrics=[]
        for fold in (0,1):
            seen=folds==fold
            if min(seen.sum(),(~seen).sum())<32:
                break
            # Fit a sprite only on alternating block courses, then compare
            # the unused courses with its projected texels at that depth.
            fit=_match(texel[seen],rgb[seen],atlas,maps)
            if not fit:
                break
            tested=~seen
            unique,groups,n=np.unique(keys[tested],return_inverse=True,return_counts=True)
            if len(unique)<10:
                break
            gray=np.bincount(groups,weights=rgb[tested].mean(1))/n
            template=maps[fit['sprite'],fit['mip'],fit['turn']]
            metrics.append(float(_corr(template[unique],gray)))
        if len(metrics)==2:
            plans.append((min(metrics),level,ijk,texel[:,yi]))
            if not audit and len(plans)>2:
                plans.sort(key=lambda v:v[0],reverse=True)
                plans.pop()
    if not plans:
        return None
    plans.sort(key=lambda v:v[0],reverse=True)
    best,level,ijk,tex=plans[0]
    margin=best-plans[1][0] if len(plans)>1 else 0.
    accepted=best>=.77 and margin>=.12
    return dict(y=y,x=x,score=best,margin=margin,level=level,ijk=ijk,
                plans=plans,accepted=accepted,tex=tex,axis=axis)


def recover(image,parts,camera,prior,atlas,*,border=2):
    if not prior or camera.get('P') is None or not parts or atlas is None or atlas.provenance.get('atlas_sha256')!=ATLAS_SHA256:
        return prior,{}
    regions,ops=split_pillars(image,parts,camera)
    if not ops:
        return prior,{}
    p,_=orientation(camera,image.size)
    eye=-np.linalg.solve(p[:,:3],p[:,3])
    native=p.copy()
    native[:2]*=SAMPLE_STRIDE
    size=image.size[::-1]
    union=np.logical_or.reduce([native_mask(part,image.size) for part in parts])
    if border:
        union=np.asarray(Image.fromarray(union.astype('uint8')).filter(ImageFilter.MaxFilter(2*border+1)),bool)
    by_id={piece['id']:piece for piece in regions}
    supported={}
    maps=_mips(atlas)
    for op in ops:
        children=[by_id[op['ids'][0]+'-tex-'+s] for s in ('0','1')]
        plans=[predict(image,piece,camera,prior,atlas,maps) for piece in children]
        if any(q is None or q['score']<.65 or q['margin']<.18 for q in plans):
            continue
        if plans[0]['level']==plans[1]['level'] or plans[0]['axis']!=plans[1]['axis']:
            continue
        for region,q in zip(children,plans):
            mask=native_mask(region,image.size)
            values,n=np.unique(q['ijk'],axis=0,return_counts=True)
            for xyz in values[n>=24]:
                cell=tuple(map(int,xyz))
                if (floating.active() or cell[1]>=0) and cell not in prior:
                    supported[cell]=(mask,q['axis'])
    proposed=set(supported)
    baseline=_frontmost_cells(sorted(prior),native,eye,size) if proposed else None
    kept=set(proposed)
    while kept:
        sequence=sorted(prior|kept)
        owner,axes=_frontmost_cells(sequence,native,eye,size,face_axes=True)
        index={c:i for i,c in enumerate(sequence)}
        result=set()
        for c in kept:
            mask,axis=supported[c]
            visible=owner==index[c]
            face=visible & (axes==axis)
            if (face & mask).sum()<24 or (face & mask).sum()<.75*face.sum():
                continue
            if (visible & ~union & (baseline<0)).any():
                continue
            result.add(c)
        result=_keep_prior_visible(image.size,p,eye,prior,result)-prior if result else set()
        if kept==result:
            break
        kept=result
    return prior|kept,dict(proposed=len(proposed),added=len(kept))
