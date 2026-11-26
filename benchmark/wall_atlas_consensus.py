"""Native-resolution atlas consensus on independently sampled wall cells."""
from functools import lru_cache

import numpy as np

from benchmark.wall_atlas_transition import _patterns
from benchmark.wall_texture_camera import _abstain, _ground, _line_data, _norm, _result


def _joint_optics(seg, horizon, size):
    w, h = size
    endpoints = (seg - np.array(size)/2)/w
    mid = endpoints.mean(1)
    diff = endpoints[:, 1]-endpoints[:, 0]
    lengths = np.linalg.norm(diff, axis=1)
    normals = _norm(np.stack([diff[:, 1], -diff[:, 0]], axis=1))
    bias = -(normals*mid).sum(1)
    horizontal = np.flatnonzero(abs(normals[:, 1])>.63)[:145]
    if len(horizontal)<6:
        return None
    eq = np.column_stack([normals[horizontal], bias[horizontal]])
    i,j = np.triu_indices(len(eq), 1)
    cross = np.cross(eq[i], eq[j])
    cross = cross[abs(cross[:, 2])>1.e-9]
    vp = cross[:, :2]/cross[:, 2:3]
    yh = (horizon-h/2)/w
    vp = vp[(vp[:, 1]>yh-.18) & (vp[:, 1]<yh+.055) & (abs(vp[:, 0])>.2) & (abs(vp[:, 0])<9)]
    banks = []
    for side in (-1, 1):
        points = vp[vp[:, 0]*side>0]
        if not len(points):
            return None
        totals = []
        for pts in np.array_split(points, 1+len(points)//300):
            dist = np.maximum(.04, np.linalg.norm(pts[:, None]-mid[horizontal], axis=2))
            angle = (pts @ eq[:, :2].T+eq[:, 2])/dist
            # Angular accuracy from a raster segment improves with its length.
            residual = angle*lengths[horizontal]*w
            totals.extend(np.maximum(0, 1-(residual/1.9)**2) @ np.minimum(lengths[horizontal], .23)**1.6)
        bank = []
        for k in np.argsort(totals)[::-1]:
            v = points[k]
            if all(np.linalg.norm(v-other)>.035 for other in bank):
                bank.append(v)
                if len(bank)==7:
                    break
        banks.append(bank)
    best = None
    for left in banks[0]:
        for right in banks[1]:
            if abs(left[1]-right[1])>.04:
                continue
            pars = np.array([left[0], right[0], (left[1]+right[1])*.5])
            for _ in range(5):
                points = np.stack([pars[:2], np.full(2, pars[2])], axis=1)
                distance = np.maximum(.04, np.linalg.norm(points[:, None]-mid[horizontal], axis=2))
                errors = abs(points @ eq[:, :2].T+eq[:, 2])/distance
                sides = errors.argmin(0)
                idx = np.arange(len(sides))
                err = errors[sides, idx]*lengths[horizontal]*w
                weights = np.maximum(0, 1-(err/1.9)**2)*lengths[horizontal]**1.6
                wt = np.sqrt(weights)*lengths[horizontal]/distance[sides, idx]
                coeff = np.stack([eq[:, 0]*(sides==0), eq[:, 0]*(sides==1), eq[:, 1]], axis=1)
                if min(np.count_nonzero((sides==s)&(wt>0)) for s in (0,1))<3:
                    break
                pars = np.linalg.lstsq(wt[:, None]*coeff, -wt*eq[:, 2], rcond=None)[0]
            else:
                a,b,hy = pars
                f2 = -(a*b+hy*hy)
                if not (.34*h/w)**2 < f2 < (2.8*h/w)**2:
                    continue
                f = np.sqrt(f2)
                pitch = np.arctan2(-hy, f)
                ey = np.array([0., -np.cos(pitch), -np.sin(pitch)])
                ex = _norm(np.array([b,hy,f]))
                ez = np.cross(ey,ex)
                K = np.array([[w*f,0,w/2],[0,w*f,h/2],[0,0,1.]])
                M = K @ np.stack([ex,ey,ez],axis=1)
                vlines = abs(normals[:, 0])>.93
                ray = np.stack([np.zeros(vlines.sum()),np.full(vlines.sum(),ey[1]*f)],axis=1)-mid[vlines]*ey[2]
                v_err = abs((_norm(ray)*normals[vlines]).sum(1))*lengths[vlines]*w
                strength = weights.sum()+.6*(np.maximum(0, 1-(v_err/2.1)**2)*lengths[vlines]**1.6).sum()
                if best is None or strength>best[0]:
                    best = strength,M
    return best[1] if best else None


def _floor_anchors(rgb, mask, seg, M):
    import cv2

    # A dark contact shadow preserves the floor's chromaticity and smoothness.
    # Learn both from visible image-border floor pixels.
    h,w = mask.shape
    patch = cv2.medianBlur(rgb,5).astype("float32")
    base = np.median(np.vstack([patch[-h//15:, :w//12].reshape(-1,3),
                                patch[-h//15:, -w//12:].reshape(-1,3)])[::12],axis=0)
    scale = (patch @ base)/max(1, base@base)
    chroma = np.max(abs(patch-scale[...,None]*base),axis=2)
    texture = cv2.blur(abs(rgb.astype("float32")-patch).mean(2),(5,5))
    foreground = ((chroma>2.2) | (texture>2.5) | (scale<.43)) & (mask>0)
    contours = _ground(rgb, foreground, seg, M)+_ground(rgb, mask, seg, M)
    anchors = []
    for a in sorted(contours, key=lambda c:-c[0]):
        if not any(a[1]==b[1] and np.linalg.norm(a[2]-b[2])<5 for b in anchors):
            anchors.append(a)
    return anchors[:6]


@lru_cache(maxsize=2)
def _atlas(version):
    data = _patterns(version)
    if data is None:
        return None
    at = data[0]
    gray = at.tiles.mean(3).astype("float32")
    tiles = np.stack([gray, gray[:,:,::-1]],axis=1).reshape(len(gray),2,-1)
    tiles -= tiles.mean(2,keepdims=True)
    tiles /= np.maximum(1.,np.linalg.norm(tiles,axis=2,keepdims=True))
    return at, tiles


def _grid(rgb, mask, M, anchor, heights, atlas):
    import cv2

    at, templates = atlas
    ix,iy = np.meshgrid(np.arange(-3,3),np.arange(5))
    u,v = np.meshgrid((np.arange(16)+.5)/16,1-(np.arange(16)+.5)/16)
    xyz = np.zeros((*ix.shape,16,16,3))
    xyz[...,anchor[1]] = ix[:,:,None,None]+u
    xyz[...,1] = iy[:,:,None,None]+v
    relative = xyz.reshape(-1,3) @ M.T
    offset = M @ anchor[3]
    out = []
    for height in heights:
        screen = relative+height*offset
        xy = (screen[:,:2]/screen[:,2:3]).astype("float32").reshape(-1,256,2)
        xx,yy = xy[:,:,0]-.5,xy[:,:,1]-.5
        valid = cv2.remap(mask,xx,yy,cv2.INTER_NEAREST).mean(1)>.97
        valid &= (np.linalg.norm(xy[:,15]-xy[:,0],axis=1)>12) & (np.linalg.norm(xy[:,240]-xy[:,0],axis=1)>12)
        valid &= screen[:,2].reshape(-1,256).min(1)>0
        if valid.sum()<3:
            continue
        crops = cv2.remap(rgb,xx[valid],yy[valid],cv2.INTER_LINEAR).astype("float32")
        means = crops.mean(1)
        gain = np.clip((means @ at.means.T)/np.maximum(1,(at.means**2).sum(1)),.33,1.12)
        color = abs(means[:,None]-gain[:,:,None]*at.means[None]).mean(2)
        short = np.argpartition(color,32,axis=1)[:,:32]
        gray = crops.mean(2)
        gray -= gray.mean(1,keepdims=True)
        gray /= np.maximum(5.,np.linalg.norm(gray,axis=1,keepdims=True))
        corr = np.einsum("nkod,nd->nko", templates[short],gray,optimize=False).max(2)
        q = corr-.017*np.take_along_axis(color,short,axis=1)
        winner = q.argmax(1)
        labels = np.full(ix.size,-1)
        values = np.full(ix.size,-1.)
        labels[valid] = short[np.arange(len(q)),winner]
        values[valid] = q[np.arange(len(q)),winner]
        ids = np.flatnonzero(values>.53)
        if len(ids)<3:
            continue
        best = []
        trans,rep = [],[]
        for i in ids:
            for j in ids[ids>i]:
                if abs(ix.flat[i]-ix.flat[j])+abs(iy.flat[i]-iy.flat[j])!=1:
                    continue
                pair = [int(i),int(j)]
                if labels[i]==labels[j]:
                    rep.append(pair)
                else:
                    mi,mj = means[np.searchsorted(np.flatnonzero(valid),pair)]
                    if abs(mi-mj).mean()>7:
                        trans.append(pair)
                # Verify the local fit on a separate nonadjacent cell.
                independent = [k for k in ids if min(abs(ix.flat[k]-ix.flat[m])+abs(iy.flat[k]-iy.flat[m])
                                                        for m in (i,j))>=2]
                if independent and pair in rep+trans:
                    k = max(independent,key=lambda n:values[n])
                    best.append((min(values[i],values[j],values[k]),i,j,k))
        if not best:
            continue
        strength,i,j,k = max(best)
        score = strength+.18*values[ids].mean()+.02*min(5,len(ids)-3)
        ev = dict(transition_pairs=len(trans),repeat_pairs=len(rep), consensus_cells=len(ids),
                  validation_cell=[int(ix.flat[k]),int(iy.flat[k])], weakest_score=float(strength),
                  cells=[dict(x=int(ix.flat[n]),y=int(iy.flat[n]),texture=at.labels[labels[n]],
                              score=round(float(values[n]),4)) for n in ids])
        out.append((score,height,anchor,ev))
    return out


def predict(image, version="26.3"):
    atlas = _atlas(version)
    if atlas is None:
        return _abstain("atlas_unavailable")
    data = _line_data(image)
    if data is None:
        return _abstain("missing_lines")
    rgb,mask,seg,horizon = data
    M = _joint_optics(seg,horizon,image.size)
    if M is None:
        return _abstain("no_joint_manhattan_fit")
    anchors = _floor_anchors(rgb,mask,seg,M)
    if not anchors:
        return _abstain("no_ground_contact")
    proposals = []
    for anchor in anchors:
        proposals.extend(_grid(rgb,mask,M,anchor,np.geomspace(.8,21,90),atlas))
    if not proposals:
        return _abstain("no_independent_wall_cells", anchors=len(anchors))
    quality,height,anchor,ev = max(proposals,key=lambda r:r[0])
    refined = _grid(rgb,mask,M,anchor,height*np.linspace(.972,1.028,23),atlas)
    if refined:
        quality,height,anchor,ev = max(refined,key=lambda r:r[0])
    if ev["weakest_score"]<.69 or quality<.86:
        return _abstain("weak_wall_consensus", **ev)
    pred = _result(M,anchor,height,dict(height=height,anchor=anchor[2].tolist(),axis=anchor[1],**ev))
    pred["reason"] = "independently_verified_wall_cells"
    return pred

