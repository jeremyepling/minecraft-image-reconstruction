"""Fit a repeating floor tile from two-dimensional translations in rectified RGB."""
from itertools import combinations

import numpy as np
from engine.camera_texel_floor import _ground
from engine.camera_atlas_floor import register, _atlas


def boundary(rgb):
    """Use a long border transition as an optional horizon proposal."""
    h, w = rgb.shape[:2]
    ys = []
    for x in (slice(0, w//8), slice(7*w//8, w)):
        patch = rgb[:int(.72*h), x].astype(float)
        diff = np.abs(patch[3:] - patch[:-3]).mean(axis=2)
        strength = np.quantile(diff, .30, axis=1)
        i = np.argmax(strength)
        if strength[i] < 4:
            return None
        ys.append(int(i+2))
    return float(np.mean(ys)) if abs(ys[0]-ys[1]) < .025*h else None


def sample(gray,x,y):
    x=np.clip(x,0,gray.shape[1]-1.001)
    y=np.clip(y,0,gray.shape[0]-1.001)
    ix=x.astype(int)
    iy=y.astype(int)
    a,b=x-ix,y-iy
    return (1-b)*((1-a)*gray[iy,ix]+a*gray[iy,ix+1])+b*((1-a)*gray[iy+1,ix]+a*gray[iy+1,ix+1])


def peaks(gray, horizon, ref):
    h,w=gray.shape
    d=ref-horizon
    # Uniform orthographic floor samples; avoid the bottom screen boundary.
    yy,xx=np.mgrid[-42:80,-384:384]
    ratio=d/(d+yy)
    data=sample(gray, w/2+xx*ratio-.5, horizon+d*ratio-.5)
    data-=np.mean(data)
    window=np.hanning(data.shape[0])[:,None]*np.hanning(data.shape[1])
    spec=np.fft.rfft2(data*window,s=(256,1024),axes=(0,1))
    corr=np.fft.irfft2(abs(spec)**2,s=(256,1024),axes=(0,1))
    # Normalize the taper overlap so that vertical translations compete fairly.
    filter_spec=np.fft.rfft2(window,s=(256,1024),axes=(0,1))
    overlap=np.fft.irfft2(abs(filter_spec)**2,s=(256,1024),axes=(0,1))
    ylim,xlim=55,120
    yi,xi=np.mgrid[0:ylim,-xlim:xlim+1]
    field=corr[yi,xi]/overlap[yi,xi].clip(.01)
    field/=max(field[0,xlim],1e-6)
    q=(yi**2+xi**2>7**2)&((yi>0)|(xi>0))
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        q &= field>=np.roll(field,(dy,dx),(0,1))
    iy,ix=np.nonzero(q&(field>.17))
    order=sorted(range(len(iy)),key=lambda i:-field[iy[i],ix[i]])[:22]
    result=[]
    for i in order:
        j,k=iy[i],ix[i]
        if not 0<j<ylim-1 or not 0<k<2*xlim:
            continue
        p=field[j,k]
        offsets=[.5*(a-b)/min(-.001,a+b-2*p) for a,b in
                 ((field[j,k-1],field[j,k+1]),(field[j-1,k],field[j+1,k]))]
        result.append((float(p),np.array([xi[j,k]+offsets[0],j+offsets[1]])))
    return result


def charts(peaks,horizon,ref,w,h):
    d=ref-horizon
    M=np.array([[d,0,-d*w/2],[0,-d,d*ref],[0,1,-horizon]])
    candidates=[]
    for (s,a),(t,b) in combinations(peaks,2):
        vectors=np.column_stack([a,b])
        if abs(np.linalg.det(vectors))<35:
            continue
        inv=M.copy()
        inv[:2]=16*np.linalg.solve(vectors,M[:2])
        # Orthogonal projected axes need a physical lift with equivalent unit lengths.
        if _ground(inv,w,h) is not None:
            candidates.append((s+t,inv,vectors))
    return sorted(candidates,key=lambda x:-x[0])


def fit_translation(gray, horizon, ref, vectors):
    # Fit both full-tile translations to native source pairs. A shared horizon ties
    # their displacements across far-apart patches and separated rows.
    h,w=gray.shape
    y,x=np.mgrid[int(.76*h):int(.973*h):5,12:w-12:9]
    src=gray[y,x]
    x=x+.5-w/2
    y=y+.5
    def terms(v,hor):
        d=ref-hor
        Y=d*d/(y-hor)-d+v[1]
        yp=hor+d*d/(Y+d)
        xp=(x*d/(y-hor)+v[0])*(yp-hor)/d+w/2
        valid=(yp>int(.70*h))&(yp<h-1)&(xp>1)&(xp<w-1)
        val=sample(gray,xp-.5,yp-.5)
        return (val-src),valid
    hor=horizon
    vec=vectors.copy()
    for _ in range(18):
        matrices=[]
        residuals=[]
        for i in range(2):
            v=vec[:,i]
            base,ok=terms(v,hor)
            derivatives=[]
            for delta in ((.1,0),(0,.1)):
                f,_=terms(v+delta,hor)
                derivatives.append((f-base)/.1)
            dh,_=terms(v,hor+.4)
            jac=np.zeros((base.size,5))
            jac[:,2*i:2*i+2]=np.column_stack([a.ravel() for a in derivatives])
            jac[:,4]=(dh-base).ravel()/.4
            valid=ok.ravel()
            matrices.append(jac[valid])
            residuals.append(base.ravel()[valid])
        J=np.vstack(matrices)
        error=np.concatenate(residuals)
        delta=np.linalg.lstsq(J,-error,rcond=None)[0]
        delta[:4]=np.clip(delta[:4],-.6,.6)
        delta[4]=np.clip(delta[4],-4,4)
        vec+=delta[:4].reshape(2,2).T
        hor+=delta[4]
        if max(abs(delta))<.008:
            break
    return hor,vec,float(np.mean(error**2))


def phase_origin(gray, inv):
    h,w=gray.shape
    y,x=np.mgrid[int(.76*h):int(.98*h),0:w-1]
    edge=gray[y,x]!=gray[y,x+1]
    X=x[edge]+1
    Y=y[edge]+.5
    vals=(inv@np.stack((X,Y,np.ones_like(X))))
    uv=vals[:2]/vals[2]
    return -np.angle(np.exp(2j*np.pi*uv).mean(1))/(2*np.pi)


def _solve(image, atlas, *, use_boundary=True):
    rgb=np.asarray(image.convert('RGB')).astype(np.int16)
    h,w=rgb.shape[:2]
    gray=rgb.mean(2)
    floor=gray[int(.82*h):]
    lo,hi=np.quantile(floor[::2,::3],[.08,.92])
    gray=np.clip(gray,lo,hi)
    ref=.91*h
    est=boundary(rgb) if use_boundary else None
    coarse=np.arange(-.35*h,.74*h,.045*h)
    if est is not None:
        coarse=np.unique(np.r_[coarse,est+np.array([-70,-40,-20,-5,0])])
    seeds=[]
    for hor in coarse:
        for s,inv,v in charts(peaks(gray,hor,ref),hor,ref,w,h)[:4]:
            seeds.append((s,hor,v))
    evidence=dict(boundary_px=est,seeds=len(seeds),tried=0)
    tried=[]
    # Take candidates with distinct lattice magnitude, direction or horizon.
    for s,hor,v in sorted(seeds,key=lambda x:-x[0]):
        if any(abs(hor-prev)<30 and np.allclose(v,other,atol=2) for prev,other in tried):
            continue
        tried.append((hor,v))
        fitted,vec,err=fit_translation(gray,hor,ref,v)
        d=ref-fitted
        M=np.array([[d,0,-d*w/2],[0,-d,d*ref],[0,1,-fitted]])
        inv=M.copy()
        inv[:2]=16*np.linalg.solve(vec,M[:2])
        if _ground(inv,w,h) is not None:
            phase=phase_origin(gray,inv)
            inv[:2]+=phase[:,None]*inv[2]
            origin,match=register(image,inv,atlas)
            evidence['tried']+=1
            if origin is not None:
                inv[:2]=(inv[:2]+origin[:,None]*inv[2])/16
                camera=_ground(inv,w,h)
                if camera is not None:
                    P,f,disagree=camera
                    evidence.update(horizon_px=fitted,repeat_residual=err,vector_patch_px=vec.tolist(),
                                    focal_px=f,axis_length_disagreement=disagree,**match)
                    return dict(P=P,scale_status='supported',camera_source='horizon-repeat-floor',
                                reason='two independent tile translations registered in both image halves',evidence=evidence)
        if len(tried)>=5:
            break
    return dict(P=None,scale_status='uncertain',reason='no bilaterally registered tile translation pair',evidence=evidence)


def _expanded_solve(image, atlas):
    """Try more boundary-seeded tile grids after the original repeat search fails."""
    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    ref = .91*h
    horizon = boundary(rgb.astype(np.int16))
    evidence = dict(seed_horizon_px=horizon, seed_focal_px=None, fits=0)
    rejected = dict(P=None, scale_status='uncertain', reason='no bilaterally verified floor proposal', evidence=evidence)
    gray = rgb.mean(2)
    limits = np.quantile(gray[int(.82*h)::2, ::3], [.08, .92])
    gray = np.clip(gray, *limits)
    offsets = (-.08, -.04, -.02, 0., .02, .04, .08)
    hypotheses = ([(trial*h, 0.) for trial in np.linspace(-.35, .73, len(offsets))]
                  if horizon is None else [(horizon + offset*h, offset) for offset in offsets])
    seeds = []
    for trial, offset in hypotheses:
        if not -.8*h < trial <= .73*h:
            continue
        for corr, inv, vectors in charts(peaks(gray, trial, ref), trial, ref, w, h):
            seeds.append((corr - .5*abs(offset), trial, vectors))
    tried = []
    for _, trial, vectors in sorted(seeds, key=lambda s: -s[0]):
        if any(abs(trial-other) < .02*h and np.allclose(vectors, vec, atol=2) for other, vec in tried):
            continue
        tried.append((trial, vectors))
        evidence['fits'] = len(tried)
        fitted, vec, residual = fit_translation(gray, trial, ref, vectors)
        d = ref-fitted
        inv = np.array([[d, 0, -d*w/2], [0, -d, d*ref], [0, 1, -fitted]])
        if abs(np.linalg.det(vec)) > 1e-6:
            inv[:2] = 16*np.linalg.solve(vec, inv[:2])
            if _ground(inv, w, h) is not None:
                inv[:2] += phase_origin(gray, inv)[:, None]*inv[2]
                origin, match = register(image, inv, atlas)
                if origin is not None:
                    inv[:2] = (inv[:2]+origin[:, None]*inv[2])/16
                    metric = _ground(inv, w, h)
                    if metric is not None:
                        P, focal_fit, disagreement = metric
                        evidence.update(horizon_px=fitted, focal_px=focal_fit, repeat_residual=residual,
                                        axis_length_disagreement=disagreement, **match)
                        return dict(P=P, scale_status='supported', camera_source='boundary-budget-repeat',
                                    reason='two tile translations registered independently in both halves', evidence=evidence)
        if len(tried) >= 12:
            break
    return rejected


def predict(image, version='26.3', *, atlas_path=None):
    """Recover block scale only after two image halves agree with a verified atlas."""
    def abstain(reason):
        return dict(P=None, scale_status='uncertain', reason=reason, evidence={})
    if version != '26.3':
        return abstain('unsupported atlas version')
    if min(image.size) < 256 or max(image.size) > 4096:
        return abstain('outside native texel sampling range')
    atlas = _atlas(version, atlas_path)
    if atlas is None:
        return abstain('calibrated atlas unavailable')
    try:
        original = _solve(image, atlas)
        return original if original['P'] is not None else _expanded_solve(image, atlas)
    except np.linalg.LinAlgError:
        return abstain('degenerate tile translation fit')
