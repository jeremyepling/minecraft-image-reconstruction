"""Register photographed sections to integer planes using atlas art seen on the build."""
from engine import floating_grid_policy as floating
from collections import Counter
from functools import lru_cache

import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.surface_model import intrinsic
from engine.thin_atlas_faces import _match, _mips
from engine.visible_horizontal_faces import _keep_prior_visible, _surface_masks
from engine.visible_placement import _frontmost_cells


def _rays(x, y, inv):
    return np.column_stack((x, y, np.ones(len(x)))) @ inv.T


def _library(rgb, P, eye, inverse, points, owner, axes, atlas, maps):
    y, x = np.nonzero(owner >= 0)
    groups = owner[y, x] * 3 + axes[y, x]
    order = np.argsort(groups)
    chunks = np.split(order, np.flatnonzero(np.diff(groups[order])) + 1)
    chunks = [chunk for chunk in chunks if len(chunk) >= 30]
    if len(chunks) > 100:
        chunks = [chunks[i] for i in np.linspace(0, len(chunks)-1, 100, dtype=int)]
    choices = [Counter() for _ in range(3)]
    for chunk in chunks:
        idx, axis = divmod(int(groups[chunk[0]]), 3)
        take = chunk[np.linspace(0, len(chunk)-1, min(100, len(chunk)), dtype=int)]
        # _frontmost_cells uses 4-pixel samples centered at 4i+2. Use the
        # nearest actual pixel center when reading colors and building texel IDs.
        u, v = 4*x[take]+2, 4*y[take]+2
        rays = _rays(u+.5, v+.5, inverse)
        free = [a for a in range(3) if a != axis]
        level = points[idx,axis] + int(eye[axis] > points[idx,axis]+1)
        uv = eye[free] + (level-eye[axis])*rays[:,free]/rays[:,[axis]]
        match = _match(np.floor(16*uv).astype(int),rgb[v,u],atlas,maps)
        if match and match['split']>.66 and match['phase']>.12 and match['color']<11:
            choices[axis][match['sprite'],match['mip'],match['turn']] += 1
    return [keys.most_common(9) for keys in choices]


def _ncc(templates, image):
    diff = templates - templates.mean(axis=-1,keepdims=True)
    gray = image - image.mean()
    return (diff @ gray)/np.maximum(.1,np.linalg.norm(diff,axis=-1)*np.linalg.norm(gray))


def _bins(rays, axis, levels, eye):
    free = [a for a in range(3) if a!=axis]
    uv = eye[free] + (levels-eye[axis])[:,None,None]*(rays[:,free]/rays[:,[axis]])
    bins = np.floor(uv*16).astype(int)%16
    return bins[:,:,1]*16+bins[:,:,0]


def sections(image, parts, camera, prior, *, atlas_path=None):
    """Return plane matches and native sample owners, without accessing reference data."""
    if not prior or not parts or camera.get('P') is None:
        return None
    atlas = _atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance['atlas_sha256'] != ATLAS_SHA256:
        return None
    maps = _mips(atlas)
    P = np.asarray(camera['P'],float)
    P = P/np.linalg.norm(P[2,:3])
    inverse = np.linalg.inv(P[:,:3])
    eye = -inverse@P[:,3]
    focal = np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rotation = np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    rgb = np.asarray(image.convert('RGB'))
    points = np.array(sorted(prior))
    xs,ys=np.arange(2,image.width,4),np.arange(2,image.height,4)
    mask, oriented = _surface_masks(parts, image.size, xs, ys, rotation)
    owner, axes = _frontmost_cells(list(map(tuple,points)),P,eye,mask.shape,face_axes=True)
    library = _library(rgb,P,eye,inverse,points,owner,axes,atlas,maps)
    models = []
    for axis, entries in enumerate(library):
        if not entries:
            continue
        lo, hi = np.percentile(points[:,axis], [1,99])
        levels=np.arange(max(0 if axis==1 and not floating.active() else -9999,int(lo)-4),int(hi)+6)
        if len(levels)>85:
            continue
        levels=levels[abs(levels-eye[axis])>.5]
        refs=np.stack([maps[s,m,t] for (s,m,t),votes in entries])
        models.append((axis,levels,refs))
    if not models:
        return None
    # Each independent tile searches complete candidate planes over the
    # measured scene range. Samples alternate across the image for validation.
    y,x=np.nonzero(mask)
    width=(mask.shape[1]+6)//7
    key=(y//7)*width+x//7
    order=np.argsort(key,kind='stable')
    chunks=np.split(order,np.flatnonzero(np.diff(key[order]))+1)
    votes = []
    for chunk in chunks:
        if len(chunk)<12:
            continue
        u,v=xs[x[chunk]],ys[y[chunk]]
        rays=_rays(u+.5,v+.5,inverse)
        gray=rgb[v,u].mean(1).astype(np.float32)
        if gray.std()<.7:
            continue
        left=((u//4 + v//4)%2)==0
        if min(left.sum(),(~left).sum())<5:
            continue
        all_scores=[]
        for axis,levels,refs in models:
            if np.min(abs(rays[:,axis]))<1e-5:
                continue
            good = (levels-eye[axis])/np.median(rays[:,axis])>1
            levels=levels[good]
            if not len(levels):
                continue
            ids=_bins(rays,axis,levels,eye)
            patterns=refs[:,ids]
            select=_ncc(patterns[...,left],gray[left])
            idx=np.argmax(select,axis=0)
            predicted=patterns[idx,np.arange(len(levels))]
            verify=_ncc(predicted[:,~left],gray[~left])
            quality=np.minimum(select.max(0),verify)
            top=np.argsort(quality)[-2:]
            for i in top:
                all_scores.append((float(quality[i]),axis,levels[i],predicted[i]))
        if not all_scores:
            continue
        all_scores.sort(key=lambda r:r[0],reverse=True)
        quality,axis,level,pattern=all_scores[0]
        gap=quality-all_scores[1][0] if len(all_scores)>1 else 0.
        if quality<.55 or gap<.02:
            continue
        normal=(gray-gray.mean())/max(gray.std(),.1)
        ref=(pattern-pattern.mean())/max(pattern.std(),.1)
        valid=abs(normal-ref)<1.25
        xyz=eye+rays*((level-eye[axis])/rays[:,axis])[:,None]
        cells=np.floor(xyz).astype(int)
        cells[:,axis]=level-int(eye[axis]>level)
        supported=oriented[axis,y[chunk],x[chunk]]
        for i in np.flatnonzero(valid & (floating.active() | (cells[:,1]>=0))):
            votes.append((int(v[i]),int(u[i]),int(owner[y[chunk[i]],x[chunk[i]]]),axis,*cells[i],
                          quality,gap,int(supported[i])))
    return points, owner, P, eye, votes


def decide(image, prior, evidence, threshold=.87, gap=.20, min_votes=5, fraction=.75):
    if evidence is None or not evidence[-1]:
        return prior
    points,owner,P,eye,rows=evidence
    pts = list(map(tuple, points.tolist()))
    hits=np.array(rows)
    hits=hits[(hits[:,7]>=threshold)&(hits[:,8]>=gap)]
    if not len(hits):
        return prior
    old = hits[:,2].astype(int)
    target = [tuple(pos) for pos in hits[:,4:7].astype(int).tolist()]
    counts=Counter(target)
    own=np.bincount(owner[owner>=0],minlength=len(points))
    wrong=Counter(i for i,t in zip(old,target) if i>=0 and pts[i]!=t and counts[t]>=min_votes)
    retired={pts[i] for i,n in wrong.items() if n>=min_votes and n>=fraction*own[i]}
    added={t for t,n in counts.items() if n>=min_votes and t not in prior}
    # A newly placed cell must still own its measured pixels on the claimed
    # face after all competing geometry is rasterized together.
    if added:
        trial=sorted((prior-retired)|added)
        new_owner,new_axes=_frontmost_cells(trial,P,eye,owner.shape,face_axes=True)
        iy,ix=hits[:,0].astype(int)//4,hits[:,1].astype(int)//4
        matched=Counter(t for t,i,a,b in zip(target,new_owner[iy,ix],new_axes[iy,ix],hits[:,3])
                        if i>=0 and trial[i]==t and a==b)
        added={t for t in added if matched[t]>=min_votes}
    # Removed cells require their photographically preferred alternative to
    # survive the visibility check.
    remain=prior-retired
    scene=_keep_prior_visible(image.size,P,eye,remain,added) if added else remain
    needed=Counter(i for i,t in zip(old,target) if i>=0 and pts[i]!=t and t in scene and counts[t]>=min_votes)
    proven={pts[i] for i,n in needed.items() if n>=min_votes and n>=fraction*own[i]}
    if proven!=retired:
        # Restoring sources can hide further added cells. Require a final ray
        # check at each claimed screen location with the restored set present.
        remain=prior-proven
        added=scene-prior
        joint=sorted(remain | added)
        if added:
            visible,axes=_frontmost_cells(joint,P,eye,owner.shape,face_axes=True)
            matches=Counter(t for t,i,a,b in zip(target,visible[iy,ix],axes[iy,ix],hits[:,3])
                            if i>=0 and joint[i]==t and a==b)
            added={t for t in added if matches[t]>=min_votes}
        scene=remain|added
    return scene


def recover(image, parts, camera, prior, *, atlas_path=None):
    evidence=sections(image,parts,camera,prior,atlas_path=atlas_path)
    return decide(image,prior,evidence)

