"""Estimate a return from local material fingerprints at competing physical depths."""
import numpy as np

from engine.geometry import bilinear

BINS=24


def _normalize_gray(rgb):
    return np.log(np.asarray(rgb,dtype=np.float32) @ np.array([.213,.715,.072],np.float32) + 5.)


def _bank(gray,lo,hi,p):
    # The origin is arbitrary. Traversing a full local block obtains a complete
    # texture cycle; comparing every cyclic translation later removes the phase.
    unit=(np.arange(BINS)+.5)/BINS
    uv=np.stack(np.meshgrid(unit,unit),axis=-1)
    h,w=gray.shape
    start=np.maximum(0,lo-3*p)
    end=np.minimum([w,h],hi+3*p)
    samples=[]
    for y in np.arange(start[1],end[1]-p[1]+1,p[1]):
        for x in np.arange(start[0],end[0]-p[0]+1,p[0]):
            origin=np.array([x,y])
            if np.all(origin<hi+.08*p) and np.all(origin+p>lo-.08*p):
                continue
            samples.append(bilinear(gray[:,:,None],origin+uv*p)[...,0])
    if not samples:return None
    bank=np.asarray(samples)
    bank-=bank.mean(axis=(1,2),keepdims=True)
    deviation=bank.std(axis=(1,2))
    bank=bank[deviation>.025]
    if not len(bank):return None
    return np.fft.rfft2(bank),np.fft.rfft2(bank**2)


def _fingerprint(bank,vals,points,period):
    # Fold all visible back samples onto a torus. Matched filtering on that torus
    # retains the within-cell pattern without selecting a phase or a texture ID.
    index=np.floor(points/period*BINS).astype(int)%BINS
    flat=index[:,1]*BINS+index[:,0]
    vals=vals-vals.mean()
    energy=vals@vals
    if energy<.0005*len(vals):return None
    shape=(BINS,BINS)
    mass=np.bincount(flat,minlength=BINS**2).reshape(shape)
    summed=np.bincount(flat,weights=vals,minlength=BINS**2).reshape(shape)
    fft=np.fft.rfft2
    inv=lambda f:np.fft.irfft2(f,s=shape,axes=(-2,-1))
    ref,ref2=bank
    cross=inv(ref*np.conj(fft(summed)))
    kernel=np.conj(fft(mass))
    variance=inv(ref2*kernel)-inv(ref*kernel)**2/len(vals)
    return float(np.max(cross/np.sqrt(np.maximum(1e-7,energy*variance))))


def predict(rect,q,vp,distance,pitch,spacing):
    """Measure 0..8 block returns, or abstain when no depth separates sufficiently."""
    scale=np.asarray(spacing,dtype=float)*pitch
    if not np.isfinite(scale).all() or scale.min()<3 or not distance>0:
        return dict(depth=None,reason='invalid local scale')
    q=np.asarray(q,float)
    vp=np.asarray(vp,float)
    lo,hi=q.min(0),q.max(0)
    gray=_normalize_gray(rect)
    bank=_bank(gray,lo,hi,scale)
    # Fine local details survive gains, smooth shade and broad geometry gradients.
    detail=gray-.25*(np.roll(gray,2,axis=1)+np.roll(gray,-2,axis=1)+
                     np.roll(gray,2,axis=0)+np.roll(gray,-2,axis=0))
    profile=[]
    for depth in range(9):
        factor=distance/(distance+depth)
        local=scale*factor
        lower=np.maximum(lo,vp+(lo-vp)*factor)+.075*scale
        upper=np.minimum(hi,vp+(hi-vp)*factor)-.075*scale
        span=(upper-lower)/local
        if min(span)<.38 or span.prod()<.7:
            break
        gx,gy=np.meshgrid(np.arange(lower[0],upper[0],1.35),np.arange(lower[1],upper[1],1.35))
        coords=np.stack((gx.ravel(),gy.ravel()),axis=-1)
        vals=bilinear(gray[:,:,None],coords).ravel()
        motif=_fingerprint(bank,vals,coords-lower,local) if bank is not None else None
        repeats=[]
        for axis in (0,1):
            step=np.eye(2)[axis]*local
            points=coords[(coords+step<upper).all(axis=1)]
            if len(points)<.23*np.prod(local)/(1.35**2):continue
            a,b=(bilinear(detail[:,:,None],points+s).ravel() for s in (0,step))
            a-=a.mean();b-=b.mean()
            norm=np.linalg.norm(a)*np.linalg.norm(b)
            if norm>.00025*len(a):repeats.append(float((a@b)/norm))
        # A back of another material may have no match on the facade. Repeated
        # motifs inside the opening can still supply independent scale evidence.
        self_match=np.mean(repeats) if repeats else None
        evidence=max(motif or 0,self_match if self_match is not None else 0)
        profile.append(dict(depth=depth,motif=motif,self_repeat=self_match,
                            texture_score=float(evidence),width_height=span.tolist()))
    ordered=sorted(profile,key=lambda r:-r['texture_score'])
    if not ordered:return dict(depth=None,reason='insufficient visible back',profile=profile)
    best=ordered[0]
    second=ordered[1]['texture_score'] if len(ordered)>1 else 1.
    confident=best['texture_score']>.63 and best['texture_score']>second+.055
    return dict(depth=best['depth'] if confident else None,score=best['texture_score'],
                margin=best['texture_score']-second,profile=profile)
