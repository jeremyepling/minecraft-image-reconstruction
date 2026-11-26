"""Try an observed roof band behind a conflicting old layer, using the same pixels at both depths."""
from collections import defaultdict

import numpy as np

from engine.camera_atlas_floor import _atlas
from engine.thin_atlas_faces import _match, _mips
from engine.visible_horizontal_faces import _surface_masks
from engine.visible_placement import _frontmost_cells


def exchange_candidates(image, points, proposals, native, eye, rot, P, atlas_path):
    candidates = set.union(*(cells for _,_,cells in proposals))
    src, src_axis = _frontmost_cells(points,native,eye,image.size[::-1],face_axes=True)
    ordered = sorted(candidates)
    dst, dst_axis = _frontmost_cells(ordered,native,eye,image.size[::-1],face_axes=True)
    old_area = np.bincount(src[src>=0],minlength=len(points))
    xs,ys = np.arange(image.width)+.5,np.arange(image.height)+.5
    witness = np.zeros_like(src,dtype=bool)
    for axis,part,cells in proposals:
        mask = _surface_masks([part],image.size,xs,ys,rot)[0]
        ids = [i for i,p in enumerate(ordered) if p in cells]
        witness |= mask & (dst_axis==axis) & np.isin(dst,ids)
    v,u = np.nonzero(witness & (src>=0))
    if len(u)==0:
        return {}, {}
    old_ids = src[v,u]
    supported = np.bincount(old_ids,minlength=len(points))
    qualified = np.flatnonzero((supported>=22) & (supported>=.70*old_area))
    audit = {'overlapped_prior':int(np.count_nonzero(supported)), 'mostly_covered':len(qualified)}
    if not len(qualified):
        return {}, audit
    atlas = _atlas('26.3',atlas_path)
    if atlas is None:
        return {}, audit
    maps = _mips(atlas)
    rgb = np.asarray(image.convert('RGB'))
    inverse = np.linalg.inv(P[:,:3])
    threats = {}
    strong = 0
    for old_idx in qualified:
        replace = defaultdict(int)
        pixels = np.flatnonzero(old_ids==old_idx)
        packed = dst[v[pixels],u[pixels]]*9+3*dst_axis[v[pixels],u[pixels]]+src_axis[v[pixels],u[pixels]]
        for key in np.unique(packed):
            slots = pixels[packed==key]
            if len(slots)<22:
                continue
            take = slots[np.linspace(0,len(slots)-1,min(220,len(slots)),dtype=int)]
            y,x = v[take],u[take]
            dst_idx,pair = divmod(int(key),9)
            a_new,a_old = divmod(pair,3)
            rays = np.column_stack((x+.5,y+.5,np.ones(len(x))))@inverse.T
            colors = rgb[y,x]
            def match(cell,axis):
                face = cell[axis]+int(eye[axis]>cell[axis]+1)
                free = [i for i in range(3) if i!=axis]
                distance = (face-eye[axis])/rays[:,axis]
                uv = eye[free]+distance[:,None]*rays[:,free]
                m = _match(np.floor(uv*16).astype(int),colors,atlas,maps)
                return m, np.median(distance)
            new,dn = match(ordered[dst_idx],a_new)
            old,do = match(points[old_idx],a_old)
            if dn<=do+.5 or not new or new['texels']<10:
                continue
            score = new['split']-new['color']/60
            old_score = old['split']-old['color']/60 if old else 0.
            if score>=.60 and new['phase']>=.12 and score-old_score>=.18:
                replace[ordered[dst_idx]]+=len(slots)
                strong+=1
        if sum(replace.values())>=.70*old_area[old_idx]:
            threats[points[old_idx]] = (dict(replace),int(old_area[old_idx]))
    audit['stronger_atlas_pairs'] = strong
    audit['eligible_removals'] = len(threats)
    return threats,audit
