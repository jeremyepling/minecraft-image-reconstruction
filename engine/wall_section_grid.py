"""Choose among visible wall-section grid registrations measured in one photograph."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.surface_model import intrinsic
from engine.surface_perimeter import _native_alpha
from engine.thin_atlas_faces import StripEvidence, _match, _corr, model_probabilities
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _frontmost_cells, _sample_face

FEATURES=('native_pixels','new_fraction','axis_fraction','silhouette','same_plane_neighbors','normal_neighbors',
          'two_sided','material_split','texel_phase','depth_margin','material_color','material_match',
          'texel_count','option_cells','plane_offset','grid_offset','plane_support','best_cell_split')
MODEL=Path(__file__).with_name('wall_section_grid_weights.json')
ORIGINS=((0.,0.),(-.375,0.),(.375,0.),(0.,-.375),(0.,.375))


def cell_evidence(ctx,cell,axis):
    if cell[1]<0 or cell[axis]<=ctx.eye[axis]<=cell[axis]+1:
        return None
    sign=1 if ctx.eye[axis]>cell[axis]+1 else -1
    hit=_sample_face(ctx.native,cell,axis,sign,ctx.shape)
    if hit is None:
        return None
    box,mask,depth,*_=hit
    fresh=mask & (ctx.depth[box]==0) & ctx.masks[axis][box]
    n=int(fresh.sum())
    if n<12:
        return None
    visible=mask & (depth>ctx.depth[box]+1.e-8)
    if n<.25*visible.sum():
        return None
    inside=total=0
    for ax in range(3):
        if cell[ax]<=ctx.eye[ax]<=cell[ax]+1:
            continue
        proj=hit if ax==axis else _sample_face(ctx.native,cell,ax,1 if ctx.eye[ax]>cell[ax]+1 else -1,ctx.shape)
        if proj:
            area,m,z,*_=proj
            shown=m & (z>ctx.depth[area]+1.e-8)
            inside+=int((shown & ctx.union[area]).sum())
            total+=int(shown.sum())
    if inside<.88*total:
        return None
    y,x=np.nonzero(fresh)
    take=np.linspace(0,n-1,min(160,n),dtype=int)
    x,y=x[take]+box[1].start,y[take]+box[0].start
    colors=ctx.rgb[y,x]
    rays=ctx._rays(x,y)
    free=[i for i in range(3) if i!=axis]
    slope=rays[:,free]/rays[:,[axis]]
    plane=cell[axis]+int(sign>0)
    uv=ctx.eye[free]+(plane-ctx.eye[axis])*slope
    match=_match(np.floor(uv*16).astype(int),colors,ctx.atlas,ctx.maps,list(ctx.anchors))
    split=phase=margin=texels=0.
    color=24.
    if match:
        split,phase,color,texels=[match[k] for k in ('split','phase','color','texels')]
        sprite=ctx.maps[match['sprite'],match['mip'],match['turn']]
        gray=colors.mean(1)
        def corr(coords):
            bins=np.floor(16*coords).astype(int)%16
            return float(_corr(sprite[16*bins[:,1]+bins[:,0]],gray))
        margin=corr(uv)-max(corr(uv-slope),corr(uv+slope))
    neighbors=[]
    for ax in (*free,axis):
        pair=[]
        for direction in (-1,1):
            q=list(cell)
            q[ax]+=direction
            pair.append(tuple(q) in ctx.prior)
        neighbors.append(pair)
    return np.array([np.log1p(n),n/max(1,visible.sum()),(visible & ctx.masks[axis][box]).sum()/max(1,visible.sum()),
                     inside/max(1,total),sum(map(sum,neighbors[:2])),sum(neighbors[2]),any(all(pair) for pair in neighbors[:2]),
                     split,phase,margin,color,bool(match),np.log1p(texels)],float)


def candidates(image,parts,camera,prior,atlas_path=None):
    if not prior or not parts or camera.get('P') is None:
        return None,[]
    atlas=_atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance.get('atlas_sha256')!=ATLAS_SHA256:
        return None,[]
    ctx=StripEvidence(image,parts,camera,prior,atlas)
    P=ctx.P
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rotation=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    cache={}
    sections=[]
    for part in parts:
        if not part.get('support') or part.get('normal_camera') is None:
            continue
        axis=int(np.argmax(abs(rotation.T@part['normal_camera'])))
        if axis==1:
            continue
        region=_native_alpha(part['support'],image.size)
        if region is None:
            continue
        mask,(left,top,_,_)=region
        sy,sx=np.nonzero(np.asarray(mask)[1::3,1::3])
        x,y=left+3*sx+1,top+3*sy+1
        if len(x)<8:
            continue
        depth=ctx.depth[y,x]
        supported=(depth>0)&(ctx.axes[y,x]==axis)
        empty=depth==0
        if empty.sum()<3 or supported.sum()<3:
            continue
        rays=ctx._rays(x,y)
        hit_levels=np.rint(ctx.eye[axis]+rays[supported,axis]/depth[supported]).astype(int)
        levels=Counter(hit_levels.tolist())
        base=levels.most_common(1)[0][0]
        if np.min(abs(rays[:,axis]))<1.e-6 or abs(base-ctx.eye[axis])<.1:
            continue
        # Anchor 2-by-2-block sections at the dominant photographed wall plane.
        free=[i for i in range(3) if i!=axis]
        slopes=rays[:,free]/rays[:,[axis]]
        base_uv=ctx.eye[free]+(base-ctx.eye[axis])*slopes
        bins=np.floor(base_uv[empty]/2).astype(int)
        blocks,group=np.unique(bins,axis=0,return_inverse=True)
        options=[{} for _ in blocks]
        valid=np.flatnonzero(empty)
        surface_levels=sorted({base+d for d in (-2,-1,0,1,2)}|{h for h,n in levels.most_common(3) if n>=4 and abs(h-base)<=6})
        for h in surface_levels:
            if (h-ctx.eye[axis])*(base-ctx.eye[axis])<=0:
                continue
            layer=h-int(ctx.eye[axis]>h)
            uv=ctx.eye[free]+(h-ctx.eye[axis])*slopes[valid]
            for origin in ORIGINS:
                ij=np.floor(uv+origin).astype(int)
                grouped,n=np.unique(np.column_stack((group,ij)),axis=0,return_counts=True)
                cells_by_group={}
                for (g,u,v),votes in zip(grouped,n):
                    if votes<2:
                        continue
                    xyz=[0,0,0]
                    xyz[axis],xyz[free[0]],xyz[free[1]]=layer,int(u),int(v)
                    cell=tuple(xyz)
                    if cell in prior:
                        continue
                    key=(cell,axis)
                    if key not in cache:
                        cache[key]=cell_evidence(ctx,cell,axis)
                    if cache[key] is not None:
                        cells_by_group.setdefault(int(g),[]).append(cell)
                for g,cells in cells_by_group.items():
                    key=tuple(sorted(cells))
                    if key in options[g]:
                        continue
                    values=np.stack([cache[c,axis] for c in cells])
                    feat=[*values.mean(0),len(cells),abs(h-base),sum(map(abs,origin)),levels[h]/supported.sum(),values[:,7].max()]
                    options[g][key]=dict(cells=key,features=np.round(feat,5).tolist(),axis=axis,level=h,origin=origin)
        sections.extend(list(group.values()) for group in options if group)
    return ctx,sections


def disjoint(sections):
    # Fix the owner of each candidate before any labels are opened. Each section retains a keep-baseline action.
    claimed=set()
    groups=[]
    for group in sections:
        owned={tuple(c) for opt in group for c in opt['cells']}-claimed
        if not owned:
            continue
        # Do not modify option geometry or its features when another section owns some of its cells.
        choices=[opt for opt in group if all(tuple(c) in owned for c in opt['cells'])]
        if choices:
            groups.append(choices)
            claimed.update(tuple(c) for opt in choices for c in opt['cells'])
    return groups


def materialize(ctx,prior,choices):
    offered={}
    for opt in choices:
        for cell in map(tuple,opt['cells']):
            offered.setdefault(cell,set()).add(opt['axis'])
    keep=set(offered)
    while keep:
        current=_keep_prior_visible(ctx.size,ctx.P,ctx.eye,prior,keep)-prior
        if not current:
            break
        owner,axes,ids=ctx.joint(current)
        checked=set()
        for cell in current:
            for axis in offered[cell]:
                hit=_sample_face(ctx.native,cell,axis,1 if ctx.eye[axis]>cell[axis]+1 else -1,ctx.shape)
                if hit:
                    box,mask,_,*_=hit
                    observed=mask & (owner[box]==ids[cell]) & (axes[box]==axis) & (ctx.depth[box]==0) & ctx.masks[axis][box]
                    if observed.sum()>=12:
                        checked.add(cell)
                        break
        if keep==checked:
            return prior|keep
        keep=checked
    return prior


def select(groups,model,threshold):
    flat=[o for g in groups for o in g]
    if not flat:
        return []
    probs=model_probabilities(np.array([o['features'] for o in flat]),model)
    selected=[]
    start=0
    for group in groups:
        scores=probs[start:start+len(group)]
        eligible=[(len(o['cells'])*(p-threshold+.015),p,j) for j,(o,p) in enumerate(zip(group,scores)) if p>=threshold]
        if eligible:
            selected.append(group[max(eligible)[2]])
        start+=len(group)
    return selected


@lru_cache(maxsize=2)
def load_model(raw):
    model=json.loads(raw)
    if model['features']!=list(FEATURES):
        raise ValueError('Unexpected wall-section model input schema')
    return model


def recover(image,parts,counts,camera,prior,*,atlas_path=None,mode='strict'):
    try:
        model=load_model(MODEL.read_bytes())
        threshold=model['thresholds'][mode]
    except (OSError,ValueError,KeyError,TypeError):
        return prior,dict(method='unavailable')
    ctx,raw=candidates(image,parts,camera,prior,atlas_path)
    groups=disjoint(raw)
    choices=select(groups,model,threshold)
    after=materialize(ctx,prior,choices) if choices else prior
    return after,dict(sections=len(groups),alternatives=sum(map(len,groups)),accepted_sections=len(choices),
                      alternate_origin=sum(any(o['origin']) for o in choices),added=len(after-prior))


def balanced(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,mode='balanced')


def operating(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,mode='operating')


def transferred(image,parts,counts,camera,prior,*,atlas_path=None,anchored=False):
    """Score each entire alternative with the existing vertical-face decision model."""
    from engine import vertical_atlas_faces as faces
    try:
        model=faces.weights(faces.WEIGHTS,faces.WEIGHTS.stat().st_mtime_ns)
    except (OSError,ValueError,TypeError,KeyError):
        return prior,dict(method='unavailable')
    ctx,sections=candidates(image,parts,camera,prior,atlas_path)
    groups=disjoint(sections)
    offered={}
    for group in groups:
        for opt in group:
            for cell in opt['cells']:
                key=cell,opt['axis']
                if key not in offered:
                    adjacent=sum(tuple(v+(s if i==axis else 0) for i,v in enumerate(cell)) in prior
                                 for axis in range(3) if axis!=key[1] for s in (-1,1))
                    offered[key]=faces.measure(ctx,cell,key[1],adjacent)
    def passes(key,joint=None):
        row=offered[key]
        if joint is not None and row:
            row=faces.measure(ctx,*key,row['neighboring'],joint)
        return faces.accepted(row,(model,.95)) and (not anchored or row['donor_dist']<=3)
    choices=[]
    for group in groups:
        valid=[o for o in group if all(passes((c,o['axis'])) for c in o['cells'])]
        if valid:
            choices.append(max(valid,key=lambda o:(len(o['cells']),min(offered[c,o['axis']]['split'] for c in o['cells']))))
    if choices:
        joint=ctx.joint({c for o in choices for c in o['cells']})
        choices=[o for o in choices if all(passes((c,o['axis']),joint) for c in o['cells'])]
    after=materialize(ctx,prior,choices) if choices else prior
    return after,dict(sections=len(groups),choices=len(choices),added=len(after-prior))


def anchored(image,parts,counts,camera,prior):
    return transferred(image,parts,counts,camera,prior,anchored=True)
