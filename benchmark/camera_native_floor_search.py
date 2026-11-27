"""Recover the ground chart from one source image before native atlas verification."""
import numpy as np
import cv2
from engine.camera_atlas_floor import _atlas
from engine.camera_texel_floor import _ground
from benchmark.camera_horizon_repeat import peaks, charts
from benchmark.camera_horizon_floor import boundary
from benchmark.camera_native_floor import patches, atlas_scan, coords, templates, sample_uv, correlation


def chart(horizon,vectors,w,h):
    ref=.91*h
    d=ref-horizon
    result=np.array([[d,0,-d*w/2],[0,-d,d*ref],[0,1,-horizon]])
    result[:2]=np.linalg.solve(vectors,result[:2])
    return result


def fit_periods(gray, valid, hor, vec, w, h):
    """Subpixel registration of two tile translations with shared horizon and valid native RGB."""
    ref=.91*h
    y,x=np.mgrid[int(.78*h):int(.97*h):6,8:w-8:9]
    ok=valid[y,x]; x,y=x[ok],y[ok]
    xp=x+.5-w/2
    yp=y+.5
    base=gray[y,x]
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=1)*.5
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=1)*.5
    def mapping(v,z):
        d=ref-z
        ratio=(1+v[1]*(yp-z)/d**2)**-1
        return ((xp+v[0]*(yp-z)/d)*ratio+w/2-.5,(yp-z)*ratio+z-.5)
    def lookup(field, X,Y):
        return cv2.remap(field,X.astype('f'),Y.astype('f'),cv2.INTER_LINEAR).ravel()
    for _ in range(15):
        J=[]; residual=[]
        for i in range(2):
            v=vec[:,i]; X,Y=mapping(v,hor)
            xs=lookup(gx,X,Y); ys=lookup(gy,X,Y)
            measured=lookup(gray,X,Y)
            good=lookup(valid.astype('f'),X,Y)>.99
            if good.sum()<80:
                return hor,vec
            jac=np.zeros((len(X),5))
            for j in (0,1,2):
                vv=v.copy()
                if j<2: vv[j]+=.12
                xx,yy=mapping(vv,hor+(.3 if j==2 else 0))
                jac[:,4 if j==2 else 2*i+j]=((xx-X)*xs+(yy-Y)*ys)/(.3 if j==2 else .12)
            J.append(jac[good]); residual.append((base-measured)[good])
        update=np.linalg.lstsq(np.vstack(J),np.concatenate(residual),rcond=None)[0]
        vec+=np.clip(update[:4],-.7,.7).reshape(2,2).T
        hor+=np.clip(update[4],-3,3)
        if np.max(abs(update))<.015: break
    return hor,vec


def solve(image, seed):
    rgb=np.asarray(image.convert('RGB'))
    raw=rgb.mean(2).astype('f')
    h,w=raw.shape
    center=np.median(raw[int(.8*h)::4,::4])
    good=cv2.erode((abs(raw-center)<4).astype('uint8'),np.ones((9,9),'uint8')).astype(bool)
    a=np.clip(raw,center-4,center+4)
    gray=(a-cv2.GaussianBlur(a,(0,0),6))*good
    samples=patches(image,bounds=((.04,.34),(.61,.84),(.89,.99),(.40,.54)))
    evidence=dict(hypotheses=0,checked=0,best_min_correlation=0.)
    failed=dict(P=None,reason='no native atlas agreement across separated floor regions',scale_status='uncertain',evidence=evidence)
    if any(len(y)<400 or y.std()<.08 for _,y in samples):return failed
    atlas=_atlas('26.3')
    # Selected on development by the frozen 448-sprite diagnostic.
    ids=[atlas.labels.index('light_gray_concrete')]
    observed=np.median(rgb[int(.8*h)::5,::7].reshape(-1,3),axis=0)
    expected=atlas.means[ids[0]]
    gain=float(observed.mean()/expected.mean())
    if not .70<gain<1.25 or np.max(abs(observed-gain*expected))>7:
        return dict(failed,reason='floor RGB outside the calibrated light gray concrete range')
    candidates=[]
    if seed is None:
        edge=boundary(rgb.astype(np.int16))
        horizons=np.linspace(-.35,.72,17)*h if edge is None else edge+np.linspace(-.18,.04,17)*h
    else:
        horizons=seed['horizon_center_px']+np.arange(-.064,.065,.008)*h
    ref=.91*h
    for hor in horizons:
        if not -.8*h<hor<.74*h:continue
        for s,inv,v in charts(peaks(gray,hor,ref),hor,ref,w,h)[:15]:
            length=np.linalg.norm(v,axis=0)
            fit=_ground(inv,w,h)
            penalty=.28*abs(np.log(fit[1]/seed['focal_xy_px'][1])) if seed else 0.
            rank=s-penalty-.004*sum(length)
            candidates.append((rank,hor,v))
    evidence['hypotheses']=len(candidates)
    tried=[]
    for rank,hor,v in sorted(candidates,key=lambda c:-c[0]):
        if any(abs(hor-z)<.02*h and np.allclose(v,V,atol=2.5) for z,V in tried): continue
        if len(tried)>=20:break
        tried.append((hor,v.copy()))
        hor,v=fit_periods(gray,good,hor,v.copy(),w,h)
        if abs(np.linalg.det(v))<2:continue
        inv=chart(hor,v,w,h)
        if _ground(inv,w,h) is not None:
            result=atlas_scan(inv,samples[:2],atlas,ids)
            quality=min(result['fit'],result['hold'])
            evidence['checked']+=1
            evidence['best_min_correlation']=max(quality,evidence['best_min_correlation'])
            if result['fit']>.18:
                finished=refine(inv,result,samples,atlas,ids,w,h)
                if finished is not None:
                    final,P=finished
                    evidence.setdefault('refined_proposals',[]).append(dict(P=P,**final))
                    if not final['accepted']:continue
                    return dict(P=P,reason='native atlas fit on two separated floor regions',scale_status='supported',camera_source='native-floor-atlas',evidence=dict(evidence,**final))
        if len(tried)>=20:break
    return failed


def refine(inv,result,samples,atlas,ids,w,h):
    n=result['n']; tile=templates(atlas,ids,n)[0,result['orientation']]
    inv=inv.copy()
    inv[:2]+=np.array(result['phase'])[:,None]*inv[2]/n
    # Fit a five-parameter upright plane and two phase offsets on separated west and far-east training patches.
    ref=.90*h
    T=np.array([[w*.5,0,w*.5],[0,h*.14,ref],[0,0,1.]])
    local=inv@T
    local/=local[2,2]
    pts=np.concatenate([samples[j][0] for j in (0,2)],axis=1)
    signal=np.concatenate([samples[j][1] for j in (0,2)])
    X=np.linalg.solve(T,pts)
    def response(matrix,points):
        return sample_uv(tile,coords(matrix,points,n),'linear')
    for _ in range(100):
        raw=response(local,X)
        y=signal-signal.mean(); raw-=raw.mean()
        gain=(raw@y)/max(raw@raw,1e-9)
        residual=y-raw*gain
        pars=[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,1)]
        derivatives=[]
        for i,j in pars:
            d=local.copy(); eps=.00015
            d[i,j]+=eps
            amp=response(d,X);amp-=amp.mean()
            derivatives.append((amp-raw)*gain/eps)
        change=np.linalg.lstsq(np.column_stack(derivatives),residual,rcond=None)[0]
        if not np.isfinite(change).all():return None
        # Limit maximum native phase change per iteration.
        trial=local.copy()
        for (i,j),delta in zip(pars,change):trial[i,j]+=delta
        shift=np.quantile(abs(coords(trial,X,n)-coords(local,X,n)),.98)
        change/=max(1,shift/.65)
        for (i,j),delta in zip(pars,change):local[i,j]+=delta
        if max(abs(change))<.000015:break
    inv=local@np.linalg.inv(T)
    metric=_ground(inv,w,h)
    if metric is None:return None
    scores=[correlation(y,response(inv,X)) for X,y in samples]
    record=dict(native_correlation=scores,mip_size=n,filter='bilinear',atlas_sha256=atlas.provenance['atlas_sha256'])
    if min(scores)<.60:return dict(record,accepted=False),metric[0]
    # Physically distinct harmonic scales get their own west phase before held scoring.
    alternatives=[]
    for scale in (.5,.98,1.02,2.):
        alt=inv.copy();alt[:2]*=scale
        fit=atlas_scan(alt,samples[:2],atlas,ids)
        alternatives.append(fit['hold'])
    passed=min(scores)-max(alternatives)>=.26
    return dict(record,wrong_scale=alternatives,accepted=passed),metric[0]
