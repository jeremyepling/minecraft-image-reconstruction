"""Estimate three orthogonal world directions from line segments in one image."""
import numpy as np


def _unit(v):
    return v / np.maximum(1e-12, np.linalg.norm(v, axis=-1, keepdims=True))


def _agreement(vps, mid, tangent, tolerance):
    """Compact angular votes in the image, also defined for points at infinity."""
    dx = vps[:, 0, None] - vps[:, 2, None] * mid[:, 0]
    dy = vps[:, 1, None] - vps[:, 2, None] * mid[:, 1]
    sine = abs(dx*tangent[:, 1] - dy*tangent[:, 0]) / np.maximum(1e-10, np.hypot(dx, dy))
    return np.maximum(0., 1 - (sine/tolerance)**2)


def _directions(mid, tangent, equations, weights):
    # Generate pencil centers from actual intersections plus parallel-family
    # hypotheses. Take the intersection in homogeneous coordinates so nearly
    # parallel rays remain well conditioned.
    n = min(len(mid), 65)
    a,b = np.triu_indices(n,1)
    cross = np.cross(equations[a],equations[b])
    proposals = _unit(np.vstack([cross[np.linalg.norm(cross,axis=1)>1e-5],
                                 np.c_[tangent[:n],np.zeros(n)]]))
    votes = _agreement(proposals, mid, tangent, .046)
    scores = votes @ weights
    chosen = []
    for i in np.argsort(-scores, kind='stable'):
        vp = proposals[i]
        if scores[i] < 1.5*weights[0]:
            break
        if any(abs(vp @ other) > .9985 for other in chosen):
            continue
        # Algebraic incidence alone favors vanishing points near the image.
        # Weight by the inverse ray length to minimize angular residuals.
        ray = vp[:2] - mid*vp[2]
        w = weights * votes[i]**2 / np.maximum(.04, np.sum(ray**2,axis=1))
        values, vectors = np.linalg.eigh((equations*w[:,None]).T @ equations)
        if values[1] < 1e-7:
            continue
        refined = vectors[:,0]
        if any(abs(refined @ other) > .9985 for other in chosen):
            continue
        chosen.append(refined)
        if len(chosen) == 24:
            break
    return np.array(chosen)


def fit_axes(lines, size):
    """Return homogeneous pixel VPs for a centered square-pixel camera, or [].

    The focal length is unknown. A camera is eligible only if independent
    segments support all three projected axes. The order is arbitrary.
    """
    if len(lines)<9 or min(size)<=0:
        return []
    center = .5*np.asarray(size, float)
    scale = max(size)
    endpoints = (np.array([[s['p'],s['q']] for s in lines], float)-center)/scale
    delta = endpoints[:,1]-endpoints[:,0]
    length = np.linalg.norm(delta,axis=1)
    # Cap the influence of single very long borders or terrain horizons.
    weights = np.minimum(length,.20)*np.array([min(1.,max(.15,float(s.get('support',1.)))) for s in lines])
    good = np.flatnonzero((length>.016) & np.isfinite(weights))
    order = good[np.argsort(-weights[good],kind='stable')[:180]]
    if len(order)<9:
        return []
    mid = endpoints[order].mean(axis=1)
    tangent = _unit(delta[order])
    normal = np.column_stack([-tangent[:,1], tangent[:,0]])
    equations = np.column_stack([normal,-(normal*mid).sum(1)])
    weights = weights[order]
    pencils = _directions(mid,tangent,equations,weights)
    if len(pencils)<3:
        return []
    # Two prominent families alone leave the third unobservable. Require a
    # three-axis explanation to cover more evidence than the best free pair;
    # otherwise a few crossing segments can invent a spurious third pencil.
    free = _agreement(pencils,mid,tangent,.043)
    pair_a,pair_b = np.triu_indices(len(pencils),1)
    two_family = (np.maximum(free[pair_a],free[pair_b]) @ weights).max()
    frames = []
    # Unproject under candidate focal lengths. Orthogonalize two axes and
    # construct the third by cross product. Thus every proposed frame has
    # three physical right angles, even when a VP is at infinity.
    fixed_focals = np.geomspace(.23,2.6,12)
    for i, first in enumerate(pencils):
        for second in pencils[i+1:]:
            product = first[2]*second[2]
            f2 = -(first[:2] @ second[:2])/product if abs(product)>1e-8 else -1.
            focals = np.r_[np.sqrt(f2),fixed_focals] if .23**2<=f2<=2.6**2 else fixed_focals
            optics = np.column_stack([np.ones((len(focals),2)),focals])
            u,v = _unit(first*optics),_unit(second*optics)
            cosine = (u*v).sum(1)
            keep = np.abs(cosine)<.09
            if not keep.any():
                continue
            u = u[keep]
            v = _unit(v[keep]-cosine[keep,None]*u)
            basis = np.stack([u,v,np.cross(u,v)],axis=1)
            frames.extend(basis/optics[keep,None,:])
    if not frames:
        return []
    best, result = 0., []
    for batch in np.array_split(np.array(frames), max(1, (len(frames)+79)//80)):
        votes = _agreement(batch.reshape(-1,3),mid,tangent,.043).reshape(len(batch),3,-1)
        assigned = votes.argmax(axis=1)
        distinct = assigned[:,None,:] == np.arange(3)[None,:,None]
        totals = (weights*votes*distinct).sum(2)
        counts = ((votes>.5)&distinct).sum(2)
        score = totals.sum(1) + .75*totals.min(1)
        valid = ((counts.min(1)>=3) & (totals.min(1)>.12) &
                 (totals.sum(1)>.30*weights.sum()) &
                 (totals.sum(1)>two_family+.02*weights.sum()))
        score = np.where(valid,score,0.)
        idx = score.argmax()
        if score[idx]>best:
            best = score[idx]
            vp = batch[idx]
            xy = vp[:,:2]*scale + center*vp[:,2,None]
            result = np.column_stack([xy,vp[:,2]]).tolist()
    return result
