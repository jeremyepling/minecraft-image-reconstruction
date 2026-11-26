"""Calibrate a level Minecraft camera using two perspective families of floor texels."""
import numpy as np


def _tracks(edges, rows):
    # A pencil intersects row y at evenly spaced X values with frequency A / (y-horizon).
    nfft = 8192
    width = edges.shape[1]
    window = np.hanning(width)
    spectrum = abs(np.fft.rfft((edges - edges.mean(1, keepdims=True))*window, nfft))
    peaks = (spectrum[:,1:-1] > spectrum[:,:-2]) & (spectrum[:,1:-1] > spectrum[:,2:])
    peaks[:,:int(.025*nfft)] = False
    peaks[:,int(.47*nfft):] = False
    return [(i, np.array(sorted(np.flatnonzero(p)+1,key=lambda k:-s[k])[:8])/nfft)
            for i,p,s in zip(rows,peaks,spectrum)], spectrum


def _pencils(edges, rows, h):
    sparse = np.arange(0,len(rows),8)
    peaks, spec = _tracks(edges[sparse], rows[sparse])
    available = np.array([len(p)>0 for _,p in peaks])
    if np.count_nonzero(available)<8: return []
    peaks=[p for p,valid in zip(peaks,available) if valid]
    spec=spec[available]
    a, b = len(peaks)//5, 4*len(peaks)//5
    y1,k1=peaks[a]
    y2,k2=peaks[b]
    finalists=[]
    for f1 in k1:
        for f2 in k2:
            if f1 <= 1.02*f2: continue
            horizon=(f1*y1-f2*y2)/(f1-f2)
            if not -.4*h<horizon<.7*h: continue
            A=(y1-horizon)*f1
            for _ in range(3):
                r=np.array([y for y,p in peaks])
                f=np.array([p[np.argmin(abs(p-A/(y-horizon)))] for y,p in peaks])
                good=abs(f-A/(r-horizon)) < .0015
                if sum(good)<5: break
                slope, offset=np.polyfit(r[good],1/f[good],1)
                if slope<=0: break
                A,horizon=1/slope,-offset/slope
            if sum(good)<.55*len(r): continue
            idx=np.clip(np.rint(A/(r-horizon)*8192).astype(int),0,4096)
            score=np.median(spec[np.arange(len(r))[good],idx[good]])*good.mean()
            finalists.append((score,A,horizon))
    chosen=[]
    for score,A,hor in sorted(finalists,reverse=True):
        if all(abs(A/(rows[-1]-hor)-other[1]/(rows[-1]-other[2]))>.003 for other in chosen):
            chosen.append((score,A,hor))
        if len(chosen)==2: break
    return chosen


def _phase(edges, rows, A, horizon):
    x=np.arange(edges.shape[1])+1-(edges.shape[1]+1)/2
    freqs=A/(rows-horizon)
    weighted=edges*(.25+.75*np.hanning(len(x)))
    states=np.sum(weighted*np.exp(-2j*np.pi*freqs[:,None]*x),axis=1)
    ref=np.mean(rows)
    d=ref-horizon
    inv=d*d*(1/(rows-horizon)-1/d)
    trials=np.arange(-2,2.001,.002)
    def solve(trials):
        vals=np.einsum("ij,j->i",np.exp(-2j*np.pi*trials[:,None]*inv),states)
        i=np.argmax(abs(vals))
        return trials[i], np.angle(vals[i])/(2*np.pi), abs(vals[i])/max(np.sum(abs(states)),1)
    slope,_,_=solve(trials)
    slope,offset,score=solve(np.arange(slope-.002,slope+.0021,.00005))
    return np.array([A,slope*d*d,offset-slope*d]),float(score)


def _refine(edges, rows, model, horizon):
    y,x=np.nonzero(edges[::3])
    y=rows[::3][y]
    x=x+1-(edges.shape[1]+1)/2
    # Refine the camera from quantized edge crossings in native pixels.
    m=model.copy()
    for _ in range(6):
        design=np.column_stack([x/(y-horizon),1/(y-horizon),np.ones(len(y))])
        coord=design@m
        residual=np.round(coord)-coord
        good=abs(residual)<np.minimum(.16,.6*m[0]/(y-horizon))
        if sum(good)<200: break
        m+=np.linalg.lstsq(design[good],residual[good],rcond=None)[0]
    return m


def _ground(Pinv,w,h):
    try:
        H=np.linalg.inv(Pinv)
    except np.linalg.LinAlgError:
        return None
    if np.any(abs(H[2,:2])<1e-12): return None
    if H[2,2]<0: H=-H
    vps=H[:2,:2]/H[2,:2]
    c=np.array([w/2,h/2])
    f2=-(vps[:,0]-c)@(vps[:,1]-c)
    if not (.3*h)**2<f2<(3*h)**2: return None
    f=np.sqrt(f2)
    K=np.array([[f,0,c[0]],[0,f,c[1]],[0,0,1]])
    axes=np.linalg.solve(K,H)
    lengths=np.linalg.norm(axes[:,:2],axis=0)
    err=abs(lengths[1]-lengths[0])/max(lengths)
    if err>.045: return None
    axes/=np.mean(lengths)
    up=np.cross(axes[:,0],axes[:,1])
    up*=np.sign(-up[1])/np.linalg.norm(up)
    return (K@np.column_stack([axes[:,0],up,axes[:,1],axes[:,2]])).tolist(),float(f),float(err)


def lattice(image, lower=.74):
    rgb=np.asarray(image.convert("RGB"),np.int16)
    h,w=rgb.shape[:2]
    rows=np.arange(int(lower*h),int(.983*h))+.5
    floor=rgb[rows.astype(int)]
    center=np.median(floor[::6,::6],axis=(0,1))
    mask=np.max(abs(floor-center),axis=2)<14
    edges=(np.any(np.diff(floor,axis=1)!=0,axis=2)&mask[:,1:]&mask[:,:-1]).astype(float)
    peaks=_pencils(edges,rows,h)
    if len(peaks)<2: return None,{"reason":"missing texel-frequency tracks"}
    if abs(peaks[0][2]-peaks[1][2])>.015*h: return None,{"reason":"inconsistent texel horizons"}
    horizon=np.mean([p[2] for p in peaks])
    models=[]
    scores=[]
    for _,A,hor in peaks:
        A=A*(np.mean(rows)-horizon)/(np.mean(rows)-hor)
        model,score=_phase(edges,rows,A,horizon)
        scores.append(score)
        models.append(_refine(edges,rows,model,horizon))
    if min(scores)<.45: return None,{"reason":"incoherent texel phase","phase_strength":scores}
    def coeff(m):
        A,B,C=m
        return [A,C,B-A*w/2-C*horizon]
    inv=np.array([coeff(m) for m in models]+[[0,1,-horizon]])
    return inv,{"horizon_px":float(horizon),"phase_strength":scores,"frequencies":[float(m[0]) for m in models]}


def predict(image,version="26.3"):
    inv,evidence=lattice(image)
    uncertain=dict(P=None,scale_status="uncertain",reason=evidence.get("reason","inconsistent metric axes"),evidence=evidence)
    if inv is None: return uncertain
    inv[:2]/=16
    camera=_ground(inv,*image.size)
    if camera is None: return uncertain
    P,f,err=camera
    evidence.update(focal_px=f,axis_length_disagreement=err)
    return dict(P=P,scale_status="supported",reason="two texel pencils; arbitrary tile phase",evidence=evidence)

