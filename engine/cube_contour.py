"""Fit the visible silhouette to observed edges of an orthogonal world frame."""
import numpy as np
from PIL import Image, ImageFilter


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _sample(mask, points):
    h, w = mask.shape
    x, y = np.floor(points).astype(int).T
    return mask[y.clip(0,h-1), x.clip(0,w-1)]


def _edge_runs(mask, lines, axes, detail):
    h, w = mask.shape
    unit = max(h,w)/400
    candidates = []
    for line in lines:
        p, q = np.asarray([line['p'], line['q']], float)
        motion = q-p
        length = np.linalg.norm(motion)
        if length < max(12, 4*unit):
            continue
        mid = (p+q)/2
        rays = axes[:, :2] - mid * axes[:, 2, None]
        rays /= np.maximum(1.e-10,np.linalg.norm(rays,axis=1,keepdims=True))
        alignment = rays @ (motion/length)
        index = int(np.argmax(abs(alignment)))
        tangent = rays[index]*np.sign(alignment[index])
        # Limit the positional error of the projected line at either end.
        offset = abs(_cross(tangent, motion))*.5
        if offset > min(.045*length, 1.65*unit):
            continue
        normal = np.array([-tangent[1], tangent[0]])
        u = np.linspace(-.48*length,.48*length,max(9,round(length/(2*unit))))
        pts = mid + u[:,None]*tangent
        # A narrow pillar may have only a few pixels of same-side support.
        # Evaluate several distances, retaining the nearest reliable pair.
        widths = np.unique(np.minimum([1.15*unit,2.0*unit,3.25*unit], .23*length))
        best, best_quality = None, 0.
        for width in widths:
            a, b = pts+width*normal, pts-width*normal
            valid = ((a>=0)&(a<[w,h])&(b>=0)&(b<[w,h])).all(1)
            if np.mean(valid) < .6:
                continue
            signed = _sample(mask,a).astype(float)-_sample(mask,b)
            sign = np.sign(signed[valid].sum())
            agreement = np.mean(sign*signed[valid])
            quality = agreement - .02*width/unit
            if agreement >= .64 and quality > best_quality:
                best = (sign*normal, width, sign*signed)
                best_quality = quality
        # A pale cloud attached to a roof can hide the silhouette vote close
        # to the edge. Look farther out for verified background, while asking
        # for uninterrupted fine wall texture on the other side of the line.
        far = min(length*.20,9.*unit)
        smooth_side = False
        if length>12*unit:
            for sign in (-1.,1.):
                exterior = pts-sign*normal*far
                interior = pts+sign*normal*.9*far
                valid = ((exterior>=0)&(exterior<[w,h])&(interior>=0)&(interior<[w,h])).all(1)
                if valid.mean()<.8:
                    continue
                a,b = _sample(detail,exterior)[valid], _sample(detail,interior)[valid]
                if (np.mean(a<1.8)>.78 and np.mean(b>=2)>.57 and np.median(b)>=np.median(a)+1.8
                        and _sample(mask,interior)[valid].mean()>.8
                        and _sample(mask,exterior)[valid].mean()<.64):
                    best = (sign*normal,min(2.5*unit,far),np.ones(len(u)))
                    best_quality = max(.65,best_quality)
                    smooth_side = True
                    break
        if best is None:
            continue
        inward, width, signed = best
        # Split a long line around actual gaps with contradictory evidence.
        # This prevents a continuous floor or texture line filling a doorway.
        keep = np.convolve(np.pad(signed,(2,2),mode='edge'),np.ones(5)/5,'valid') >= .08
        for chunk in np.split(np.arange(len(u)),np.flatnonzero(keep[:-1]!=keep[1:])+1):
            if not keep[chunk[0]] or len(chunk)<3:
                continue
            lo = -length*.5 if chunk[0]==0 else (u[chunk[0]]+u[chunk[0]-1])*.5
            hi = length*.5 if chunk[-1]==len(u)-1 else (u[chunk[-1]]+u[chunk[-1]+1])*.5
            if hi-lo < max(10.,3*unit):
                continue
            candidates.append(dict(ends=np.array([mid+lo*tangent,mid+hi*tangent]),
                                   tangent=tangent, normal=inward, axis=index,
                                   width=width, weight=best_quality, smooth=smooth_side))
    return candidates


def _join(runs, limit):
    """Intersect neighboring ends only when they belong to different world axes."""
    if not runs:
        return []
    ends = np.array([r['ends'] for r in runs])
    moves = []
    for i,r in enumerate(runs):
        for j in range(i):
            s = runs[j]
            if r['axis']==s['axis']:
                continue
            a, b = r['tangent'],s['tangent']
            divider = _cross(a,b)
            if abs(divider)<.15:
                continue
            point = ends[i,0]+a*(_cross(ends[j,0]-ends[i,0],b)/divider)
            di = np.linalg.norm(ends[i]-point,axis=1)
            dj = np.linalg.norm(ends[j]-point,axis=1)
            k, l = int(di.argmin()), int(dj.argmin())
            gap = limit
            if r['smooth'] and s['smooth']:
                gap=max(limit,min(2.4*limit,.20*np.linalg.norm(ends[i,0]-ends[i,1]),
                                  .20*np.linalg.norm(ends[j,0]-ends[j,1])))
            if max(di[k],dj[l])<=gap:
                moves.append((di[k]+dj[l],i,k,j,l,point))
    occupied = set()
    corners = []
    for _,i,k,j,l,point in sorted(moves,key=lambda m:m[0]):
        if (i,k) in occupied or (j,l) in occupied:
            continue
        runs[i]['ends'][k] = point
        runs[j]['ends'][l] = point
        occupied.update(((i,k),(j,l)))
        corners.append(dict(point=point.round(2).tolist(), runs=[i,j]))
    return corners


def _clear_convex_tips(votes, detail, runs, turns, unit):
    """Cover the unseen wedge between two measured, smooth-background strips."""
    h,w = votes.shape
    for turn in turns:
        r,s = (runs[k] for k in turn['runs'])
        if not (r['smooth'] and s['smooth']):
            continue
        tip = np.array(turn['point'])
        directions = []
        for run in (r,s):
            endpoints = run['ends']-tip
            directions.append(endpoints[np.linalg.norm(endpoints,axis=1).argmax()])
        # For a convex corner each next edge runs into the previous edge's
        # foreground half-plane. A recessed step must keep its outside gap.
        if directions[0]@s['normal']<=0 or directions[1]@r['normal']<=0:
            continue
        reach = min(38*unit,.95*min(np.linalg.norm(d) for d in directions))
        lo = np.maximum(0,np.floor(tip-reach).astype(int))
        hi = np.minimum([w,h],np.ceil(tip+reach).astype(int))
        if np.any(hi<=lo):
            continue
        y,x = np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
        delta = np.stack((x+.5-tip[0],y+.5-tip[1]),axis=2)
        outer = (delta@r['normal']<0)&(delta@s['normal']<0)&(np.linalg.norm(delta,axis=2)<reach)
        ray = delta[outer]
        distance = np.maximum(1.,np.linalg.norm(ray,axis=1,keepdims=True))
        clear = np.ones(len(ray),bool)
        for t in np.linspace(.1,1.,13):
            points = tip + ray*np.maximum(2.8*unit/distance,t)
            clear &= _sample(detail,points)<=2
        outer[outer] = clear
        votes[lo[1]:hi[1],lo[0]:hi[0]][outer] -= 1.


def refine(image, mask, lines, axes):
    """Locally correct a binary outline using line and foreground-side evidence."""
    if image.size != mask.size:
        raise ValueError('Contour mask and image dimensions must match')
    evidence = dict(runs=[], turns=[], changed_pixels=0)
    if len(axes)!=3 or ('A' in image.getbands() and image.getchannel('A').getextrema()[0]<128):
        return mask,evidence
    base = np.asarray(mask.convert('L'))>=128
    h,w = base.shape
    unit = max(w,h)/400
    gray = image.convert('L')
    residual = abs(np.asarray(gray,dtype=np.int16) -
                   np.asarray(gray.filter(ImageFilter.GaussianBlur(max(1.,.46*unit))),dtype=np.int16))
    detail = np.asarray(Image.fromarray(residual.astype('uint8')).filter(ImageFilter.BoxBlur(max(1.,unit))))
    runs = _edge_runs(base,lines,np.asarray(axes,float),detail)
    evidence['turns'] = _join(runs,3.3*unit)
    votes = np.zeros((h,w),np.float32)
    for run in runs:
        p,q = run['ends']
        reach = run['width'] + .55*unit
        extent = min(26*unit,.66*np.linalg.norm(q-p)) if run['smooth'] else reach
        lo = np.maximum(0,np.floor(np.minimum(p,q)-extent).astype(int))
        hi = np.minimum([w,h],np.ceil(np.maximum(p,q)+extent).astype(int))
        if np.any(hi<=lo):
            continue
        y,x = np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
        delta = np.stack((x+.5-p[0], y+.5-p[1]),axis=2)
        normal_distance = delta@run['normal']
        along = delta@run['tangent']
        inside = (along>=0)&(along<=(q-p)@run['tangent'])&(abs(normal_distance)<reach)
        # A correction must join existing labels on *both* sides. This also
        # prevents a second parallel edge from punching slits in a full face.
        center = p + along[inside,None]*run['tangent']
        pair = _sample(base,center+reach*run['normal']) & ~_sample(base,center-reach*run['normal'])
        inside[inside] = pair
        weight = run['weight'] * (1-abs(normal_distance)/reach)
        region = votes[lo[1]:hi[1],lo[0]:hi[0]]
        region += np.where(inside,np.sign(normal_distance)*weight,0)
        if run['smooth']:
            outward = ((along>=0)&(along<=(q-p)@run['tangent'])&
                       (normal_distance<0)&(normal_distance>=-extent))
            # Stop the outward strip if it crosses another textured surface.
            origin = p + along[outward,None]*run['tangent']
            depth = -normal_distance[outward,None]
            clear = np.ones(len(depth),bool)
            for t in np.linspace(.1,1.,9):
                step = np.maximum(2.8*unit,t*depth)
                clear &= _sample(detail,origin-step*run['normal'])<=2
            outward[outward] = clear
            region[outward] -= .8
    _clear_convex_tips(votes,detail,runs,evidence['turns'],unit)
    updated = np.where(abs(votes)>.08,votes>0,base)
    evidence['runs'] = [dict(points=r['ends'].round(2).tolist(),axis=r['axis']) for r in runs]
    evidence['changed_pixels'] = int((updated!=base).sum())
    return Image.fromarray(updated.astype('uint8')*255),evidence
