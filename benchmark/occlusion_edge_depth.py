"""Test visible T junctions as near/far constraints within a merged wall carrier."""
from collections import Counter, defaultdict

import numpy as np

from benchmark.wall_color_partition import bounded_rgb_regions
from engine.atlas_plane_depth import _boundary
from engine.visible_raster import SAMPLE_STRIDE

RING = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])


def junctions(labels, rgb):
    """A semicircular cap persists on two rings; two separate colours meet its long edge."""
    h, w = labels.shape
    # Sample half-size centres between pixels to permit a T at a three-way pixel vertex.
    seeds = (labels[:-1, :-1] != labels[1:, 1:]) | (labels[:-1, 1:] != labels[1:, :-1])
    yy, xx = np.nonzero(seeds)
    keep = (yy >= 6) & (yy < h-7) & (xx >= 6) & (xx < w-7)
    yy, xx = yy[keep], xx[keep]
    arms = [labels[yy[:, None]+size*RING[:, 0], xx[:, None]+size*RING[:, 1]] for size in (2, 4)]
    found = defaultdict(list)
    for start in range(8):
        small, large = (np.roll(a, -start, axis=1) for a in arms)
        # Both rear wedges need coherent observed colour. One may be outside
        # this wall carrier (for example a sky gap), so its numeric label can be zero.
        valid = ((small[:, :4] == small[:, :1]).all(1) & (small[:, 4] == small[:, 5])
                 & (small[:, 6] == small[:, 7]) & (small[:, 0] > 0)
                 & (small[:, 0] != small[:, 4]) & (small[:, 0] != small[:, 6]) & (small[:, 4] != small[:, 6]))
        valid &= ((large[:, :4] == small[:, :1]).sum(1)>=3)
        valid &= (large[:, 4:6] == small[:, 4:5]).any(1) & (large[:, 6:8] == small[:, 6:7]).any(1)
        direction = np.roll(RING, -start, axis=0)
        for i in np.flatnonzero(valid):
            if small[i].min()==0:
                slots = [j for j in (4,5,6,7) if small[i,j]==0]
                patch = np.array([rgb[yy[i]+r*direction[j,0], xx[i]+r*direction[j,1]] for j in slots for r in (2,4)])
                if np.ptp(patch,axis=0).max()>32:
                    continue
            for back in (4, 6):
                if not small[i,back]:
                    continue
                key = int(small[i, 0]), int(small[i, back])
                p = (int(yy[i]), int(xx[i]))
                if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 5 for q in found[key]):
                    found[key].append(p)
    return found


def grid_fit(region, projection, eye, axis, levels):
    free = [i for i in range(3) if i != axis]
    boundary = _boundary(region, np.linalg.inv(projection[:, :3]), axis, free)
    boundary = boundary[np.isfinite(boundary).all(1)]
    fits = {}
    for level in levels:
        uv = eye[free] + (level-eye[axis])*boundary
        error = abs((uv+.5) % 1-.5).min(1)
        fits[int(level)] = float(np.maximum(0, 1-error/.18).mean()) if len(error) else 0.
    return fits


def face_support(region, union, rays, eye, axis, level):
    """Measure grid-cell coverage and projected spill into unassigned pixels near this component."""
    ys, xs = np.nonzero(region)
    pad = 12
    crop = (slice(max(0,ys.min()-pad),ys.max()+pad+1), slice(max(0,xs.min()-pad),xs.max()+pad+1))
    q = rays[crop]
    r = region[crop]
    with np.errstate(divide='ignore',invalid='ignore'):
        depth = (level-eye[axis])/q[:,:,axis]
    valid = (depth>1.5)&(depth<150)
    free = [i for i in range(3) if i!=axis]
    uv = np.floor(eye[free]+np.where(valid,depth,0)[:,:,None]*q[:,:,free]).astype(np.int64)
    ids = uv[:,:,0]*(1<<32)+uv[:,:,1]
    keys, inverse = np.unique(ids, return_inverse=True)
    inverse = inverse.reshape(r.shape)
    hits = np.bincount(inverse[r & valid],minlength=len(keys))
    totals = np.bincount(inverse[valid],minlength=len(keys))
    core = (hits>=4)&(hits>=.5*totals)
    rendered = valid & core[inverse]
    return float(np.count_nonzero(rendered&r)/len(ys)),float(np.count_nonzero(rendered&~union[crop])/max(1,rendered.sum()))


def propose(image, mask, union, P, eye, axis, initial):
    labels = bounded_rgb_regions(image,mask)
    h,w = mask.shape
    rgb = np.asarray(image.convert("RGB"))[:4*h,:4*w].reshape(h,4,w,4,3).mean(axis=(1,3))
    pairs = junctions(labels,rgb)
    yy,xx=np.indices(mask.shape)
    rays=np.stack(((xx+.5)*SAMPLE_STRIDE,(yy+.5)*SAMPLE_STRIDE,np.ones(mask.shape)),axis=-1)@np.linalg.inv(P[:,:3]).T
    levels = [l for l in range(initial-12, initial+13) if (l-eye[axis])*(initial-eye[axis])>0]
    fits = {}
    def grid(group):
        if group not in fits:
            fits[group] = grid_fit(labels==group,P,eye,axis,levels)
        return fits[group]

    offers = []
    for (front,back),points in pairs.items():
        # An independent floor intercept supplies absolute depth where the cap reaches observed floor.
        fg = labels==front
        foot = fg & (yy == np.where(union,yy,-1).max(0)) & (yy<mask.shape[0]-1) & (rays[:,:,1]<-.001)
        foot_values = eye[axis]-eye[1]*rays[foot,axis]/rays[foot,1]
        rank = sorted(grid(front),key=lambda l:(-grid(front)[l],abs(l-initial)))
        anchor = rank[0]
        reason = 'provisional contour'
        if len(foot_values)>=max(5,.15*fg.any(0).sum()):
            floor = int(np.rint(np.median(foot_values)))
            if np.quantile(abs(foot_values-floor),.8)<=.35:
                anchor, reason = floor,'floor'
        if reason=='provisional contour' and grid(front)[anchor]>=.58 and grid(front)[anchor]-grid(front)[rank[1]]>=.055:
            reason = 'distinct contour'
        region = labels==back
        candidates = []
        for level in levels:
            behind = abs(level-eye[axis])-abs(anchor-eye[axis]) >= .5
            coverage,spill = face_support(region,union,rays,eye,axis,level)
            merit = grid(back)[level]+.30*coverage-1.2*spill
            candidates.append(dict(level=level,behind=bool(behind),grid=grid(back)[level],coverage=coverage,
                                   spill=spill,merit=merit))
        different = [c for c in candidates if c['behind'] and c['level']!=initial and c['coverage']>0]
        if not different:
            continue
        best = max(different,key=lambda c:(c['merit'],-abs(c['level']-initial)))
        prior = next(c for c in candidates if c['level']==initial)
        gain = best['merit']-prior['merit']
        eligible = (reason!='provisional contour' and gain>=.08 and best['coverage']>=.65
                    and best['spill']<=.04 and best['spill']<=prior['spill']+.01)
        offers.append(dict(front=front,back=back,points=points,anchor=anchor,anchor_reason=reason,
                           samples=int(region.sum()),level=best['level'],gain=gain,eligible=bool(eligible),rows=candidates))
    ranked = sorted(offers,key=lambda o:(o['eligible'],o['anchor_reason']!='provisional contour',
                                            o['gain'],len(o['points']),o['samples']),reverse=True)
    return labels, dict(components=int(labels.max()),pairs=len(pairs),offers=ranked)
