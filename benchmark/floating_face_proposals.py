"""Discover a native complete wall tile from the screenshot and the observed parent mask."""
from collections import Counter

import cv2
import numpy as np

from benchmark.full_block_lattice import atlas, line_camera, verify_sprite
from benchmark.camera_structure_scaffold import outline_lines
from engine.atlas_count import _shortlist
from engine.geometry import UNIT, project
from engine.surface_model import intrinsic


def image_axes(image, planes):
    lines, _ = outline_lines(image, planes, [])
    return line_camera(lines, image.size) if len(lines) >= 3 else None


def rectifications(rgb, mask, normal, linear):
    h, w = mask.shape
    y, x = np.nonzero(mask[::3, ::3])
    if len(x) < 20:
        return
    xy = np.column_stack((3*x+.5, 3*y+.5))
    focal = h if linear is None else np.linalg.norm(linear[0]-w/2*linear[2])
    k = intrinsic((w, h), focal)
    center = np.linalg.solve(k, np.r_[np.median(xy, axis=0), 1.])
    up = np.array([0., -1., 0.]) if linear is None else np.linalg.solve(k, linear[:, 1])
    up /= np.linalg.norm(up)
    normals = [('normal', np.asarray(normal, float))]
    if linear is not None:
        dirs = np.linalg.solve(k, linear[:, [0, 2]])
        normals.append(('straight-edges', dirs[:, np.argmax(abs(dirs.T @ normal))]))
    for label, n in normals:
        across = np.cross(up, n)
        across /= max(.001, np.linalg.norm(across))
        down = -up
        if (k @ across)[0] - w/2*across[2] < 0:
            across = -across
        H = k @ np.column_stack((across, down, center))
        if abs(np.linalg.det(H)) < .1:
            continue
        uv = project(np.linalg.inv(H), xy)
        low, high = uv.min(0), uv.max(0)
        if not np.isfinite(uv).all() or np.any(high-low < .002) or np.max(high-low)>7:
            continue
        # One absolute scale per plane. Rectified pixels remain square in wall units.
        factor = min(focal, 150 / max(high-low))
        shape = np.ceil((high-low)*factor+4).astype(int)
        if min(shape)<9:
            continue
        origin = low-2/factor
        R = H @ np.array([[1/factor,0,origin[0]],[0,1/factor,origin[1]],[0,0,1.]])
        yy, xx = np.indices(tuple(shape[::-1]))
        coords = (project(R, np.stack((xx+.5,yy+.5),-1))-.5).astype('f')
        patch = cv2.remap(rgb,coords,None,cv2.INTER_LINEAR)
        inside = cv2.remap(mask.astype('B'),coords,None,cv2.INTER_NEAREST)
        yield label, patch, inside, R


def palette(patch, support):
    cat = atlas()
    keys = Counter()
    for rgb, mask in [(patch, support)] + [(r, m) for rows, masks in zip(np.array_split(patch,2),np.array_split(support,2))
                                         for r,m in zip(np.array_split(rows,2,axis=1),np.array_split(masks,2,axis=1))]:
        if mask.sum() < 36:
            continue
        ids, loss = _shortlist(rgb,mask.astype(bool),cat)
        for i, idx in enumerate(ids[:6]):
            if loss[idx] <= 20 and np.mean(cat.spreads[idx]) >= 2.5:
                keys[int(idx)] += 1/(i+1)
    return [k for k,_ in keys.most_common(6)]


def candidates(patch, mask, R):
    gray = patch.astype('f').mean(2)
    iv = cv2.integral(mask)
    # Four independently measured sides prevent a single strong stripe from dominating the boundary score.
    dx = np.abs(np.diff(gray,axis=1,prepend=gray[:,:1]))
    dy = np.abs(np.diff(gray,axis=0,prepend=gray[:1]))
    ix = np.pad(np.cumsum(dx,axis=0), ((1,0),(0,0)))
    iy = np.pad(np.cumsum(dy,axis=1), ((0,0),(1,0)))
    keys = palette(patch, mask)
    cat = atlas()
    offers = []
    for size in (6,7,8,9,10,12,14,16,19,22,26,31,37,44,53,64):
        if size >= min(gray.shape)-1:
            continue
        occupied = iv[size:,size:]-iv[:-size,size:]-iv[size:,:-size]+iv[:-size,:-size]
        valid = occupied[:-1,:-1] >= .985*size*size
        if not valid.any():
            continue
        vx = (ix[size:]-ix[:-size])[:-1]/size
        vy = (iy[:,size:]-iy[:,:-size])[:,:-1]/size
        sides = np.stack([vx[:,:-size],vx[:,size:],vy[:-size,:],vy[size:,:]])
        edge = np.mean(np.minimum(sides,45),axis=0)*.5+np.min(sides,axis=0)*.5
        edge = np.minimum(1,edge/18)
        edge[~valid] = -1
        _, pv, _, pos = cv2.minMaxLoc(edge)
        offers.append((pv, size, pos, 'four-sides', float(pv)))
        peak = -1
        winner = None
        for key in keys:
            for orient in (0,2,4,6):
                tile = cv2.resize(cat.patterns[key,orient],(size,size),interpolation=cv2.INTER_LINEAR)
                corr = cv2.matchTemplate(gray,tile,cv2.TM_CCOEFF_NORMED)[:-1,:-1]
                corr += .07*edge
                corr[~valid] = -1
                _, val, _, (x,y) = cv2.minMaxLoc(corr)
                if val > peak:
                    peak, winner = val, (x,y)
        if peak >= .51:
            x,y = winner
            offers.append((peak,size,(x,y),'atlas',float(edge[y,x])))
    for prior,size,offset,kind,edge in offers:
        x,y = offset
        H = R @ np.array([[size,0,x],[0,size,y],[0,0,1.]])
        quad = project(H,UNIT)
        if np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1).min() >= 6:
            yield H, quad, kind, prior, edge


def source_offers(rgb, mask, normal, linear):
    options = []
    for axis,patch,inside,R in rectifications(rgb,mask,normal,linear):
        for H,quad,kind,prior,edge in candidates(patch,inside,R):
            hit = verify_sprite(rgb,quad)
            corr = 0. if hit is None else hit[2]
            neighbors = []
            if hit or edge >= .6:
                for delta in ((1,0),(-1,0),(0,1),(0,-1)):
                    coords = project(H,UNIT+delta)
                    grid = project(H, NEIGHBOR_SAMPLES + delta).astype(int)
                    xx, yy = grid.T
                    if min(xx.min(), yy.min()) < 0 or xx.max()>=mask.shape[1] or yy.max()>=mask.shape[0] or not mask[yy,xx].all():
                        continue
                    other = verify_sprite(rgb,coords)
                    if other:
                        neighbors.append(other[2])
            supported = sum(c >= .70 for c in neighbors)
            score = corr + .07*min(2,supported) + .075*edge + .04*(max(neighbors,default=0.))
            # The source vote must have a whole-sprite match and independent adjacent or boundary support.
            accept = corr >= .72 and score >= .88 and (supported >= 1 or edge >= .62)
            options.append(dict(axis=axis,method=kind,quad=quad.tolist(),atlas=hit[0] if hit else None,
                                correlation=corr,edge=edge,neighbors=neighbors,score=score,accept=bool(accept)))
    return options


NEIGHBOR_SAMPLES = np.array([(x,y) for y in (.04,.27,.5,.73,.96) for x in (.04,.27,.5,.73,.96)])
