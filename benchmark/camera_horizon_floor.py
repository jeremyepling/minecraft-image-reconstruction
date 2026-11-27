"""Source-only perspective floor-track proposals for abstaining upright cameras."""
import numpy as np

from engine.camera_atlas_floor import _atlas, register
from engine.camera_texel_floor import _ground


def boundary(rgb):
    """Long, near-horizontal transitions at opposite borders suggest a weak horizon."""
    h, w = rgb.shape[:2]
    ys = []
    for x in (slice(0, w//8), slice(7*w//8, w)):
        patch = rgb[:int(.72*h), x].astype(float)
        # A distant ground edge may sit below the vanishing line.
        diff = np.abs(patch[3:] - patch[:-3]).mean(axis=2)
        strength = np.quantile(diff, .30, axis=1)
        i = np.argmax(strength)
        if strength[i] < 4:
            return None
        ys.append(int(i+2))
    return float(np.mean(ys)) if abs(ys[0]-ys[1]) < .025*h else None


def row_peaks(edges, rows):
    nfft = 8192
    spec = abs(np.fft.rfft((edges-edges.mean(1, keepdims=True))*np.hanning(edges.shape[1]), nfft))
    maxima = (spec[:,1:-1]>spec[:,:-2]) & (spec[:,1:-1]>spec[:,2:])
    maxima[:,:int(.012*nfft)] = False
    peaks = np.where(maxima, spec[:,1:-1], 0.)
    inds = np.argsort(peaks, axis=1)[:,-24:]+1
    power = np.take_along_axis(spec, inds, axis=1)
    # Nearest-neighbour sampled edges also carry folded frequencies.
    f = inds / nfft
    return np.concatenate((f, 1-f), axis=1), np.tile(power, (1,2))


def tracks(freq, power, rows, h, prior):
    center = len(rows)//2
    horizons = np.arange(-.48*h, .7*h, .035*h) if prior is None else prior + h*np.array([-.14,-.09,-.045,-.018,0,.018])
    proposals = []
    for guess in horizons:
        A = (rows[center]-guess)*freq[center]
        hor = np.full_like(A, guess)
        for tolerance in (.009, .004, .0018):
            target = A[:,None]/(rows-hor[:,None])
            delta = abs(freq[None,:,:]-target[:,:,None])
            ix = delta.argmin(axis=2)
            chosen = freq[np.arange(len(rows)),ix]
            ok = abs(chosen-target)<tolerance
            # Linear fit of 1/f against row for each independently proposed track.
            weight = ok / np.maximum(1,ok.sum(1,keepdims=True))
            mx, my = weight@rows, (weight/chosen).sum(1)
            var = (weight*(rows-mx[:,None])**2).sum(1)
            slope = (weight*(rows-mx[:,None])*(1/chosen-my[:,None])).sum(1)/np.maximum(var,1e-8)
            A = 1/np.maximum(slope, 1e-7)
            hor = mx-my*A
        for i in np.flatnonzero((ok.mean(1)>.36)&(hor>-.55*h)&(hor<.72*h)):
            if prior is not None and not prior-.18*h < hor[i] < prior+.04*h:
                continue
            weights = power[np.arange(len(rows)),ix[i]] * ok[i]
            proposals.append((float(weights.mean()),A[i],hor[i]))
    unique = []
    for p in sorted(proposals, reverse=True):
        if all(abs(p[1]/(rows[center]-p[2])-q[1]/(rows[center]-q[2]))>.0014 or abs(p[2]-q[2])>10 for q in unique):
            unique.append(p)
        if len(unique)==14:
            break
    return unique


def phase_tracks(edges, rows, A, horizon):
    h, w = edges.shape
    x = np.arange(w)+1-(w+1)/2
    f = A/(rows-horizon)
    phase = np.exp((-2j*np.pi)*f[:,None]*x)
    states = (edges * (.25+.75*np.hanning(w))) * phase
    halves = np.array([states[:,:w//2].sum(1), states[:,w//2:].sum(1)])
    total = abs(halves).sum(1).clip(1)
    d = np.mean(rows)-horizon
    inv = d*d*(1/(rows-horizon)-1/d)
    slopes = np.arange(-2.5,2.501,.003)
    def strengths(trials):
        vals = np.exp(-2j*np.pi*trials[:,None]*inv) @ halves.T
        return vals, np.min(abs(vals)/total,axis=1)
    _, strength = strengths(slopes)
    candidates = np.flatnonzero((strength[1:-1]>strength[:-2]) & (strength[1:-1]>strength[2:]))+1
    results = []
    for i in sorted(candidates, key=lambda i:-strength[i])[:3]:
        fine = np.arange(slopes[i]-.003, slopes[i]+.0031,.00012)
        vals, scores = strengths(fine)
        j = np.argmax(scores)
        if scores[j] < .28:
            continue
        B = fine[j]*d*d
        C = np.angle(vals[j].sum())/(2*np.pi)-fine[j]*d
        m = np.array([A,B,C])
        # Refine the full two-dimensional edge phase across separated regions.
        iy, ix = np.nonzero(edges[::4,::3])
        y, xobs = rows[::4][iy]-horizon, x[ix*3]
        design = np.column_stack((xobs/y, 1/y, np.ones_like(y)))
        for _ in range(5):
            residual = design@m
            residual = np.rint(residual)-residual
            good = abs(residual)<np.minimum(.17,.6*m[0]/y)
            if good.sum()<160:
                break
            m += np.linalg.lstsq(design[good],residual[good],rcond=None)[0]
        results.append((float(scores[j]),m))
    return results


def solve(image, mode):
    rgb = np.array(image.convert('RGB'),dtype=np.int16)
    h,w = rgb.shape[:2]
    prior = boundary(rgb) if mode=='boundary' else None
    evidence = dict(mode=mode, boundary_px=prior, pairs_tried=0)
    def fail(reason):
        return dict(P=None,scale_status='uncertain',reason=reason,evidence=evidence)
    if mode=='boundary' and prior is None:
        return fail('no bilateral horizon boundary')
    lower=.78
    rows=np.arange(int(lower*h),int(.983*h))+.5
    floor=rgb[rows.astype(int)]
    mask=np.max(abs(floor-np.median(floor[::6,::6],axis=(0,1))),axis=2)<14
    edges=(np.any(np.diff(floor,axis=1)!=0,axis=2)&mask[:,1:]&mask[:,:-1]).astype(float)
    peaks,power=row_peaks(edges[::5],rows[::5])
    paths=tracks(peaks,power,rows[::5],h,prior)
    evidence['tracks']=len(paths)
    # Same-direction harmonics agree on horizon too. Fit once at each consensus,
    # then discriminate directions using both components of the phase gradient.
    groups=[]
    for _, _, hor in paths:
        group=[p for p in paths if abs(p[2]-hor)<8]
        if len(group)>=2:
            groups.append((sum(p[0] for p in group),sum(p[0]*p[2] for p in group)/sum(p[0] for p in group),group))
    if not groups:
        return fail('no common track horizon')
    used=[]
    for _,horizon,group in sorted(groups,key=lambda g:-g[0]):
        if any(abs(horizon-old)<3 for old in used):
            continue
        used.append(horizon)
        models=[]
        ref=rows.mean()
        for _,amplitude,hor in group[:8]:
            A=amplitude*(ref-horizon)/(ref-hor)
            if any(abs(A-m[1][0])/(ref-horizon)<.001 for m in models):
                continue
            models.extend(phase_tracks(edges,rows,A,horizon))
        pairs=[]
        for i,(sa,ma) in enumerate(models):
            for sb,mb in models[:i]:
                grad=np.array([[m[0]/(ref-horizon),-m[1]/(ref-horizon)**2] for m in (ma,mb)])
                sine=abs(np.linalg.det(grad))/np.prod(np.linalg.norm(grad,axis=1))
                if sine < .12:
                    continue
                inv=np.array([[m[0],m[2],m[1]-m[0]*w/2-m[2]*horizon] for m in (ma,mb)]+[[0,1,-horizon]])
                if _ground(inv,w,h) is not None:
                    pairs.append((sa*sb,inv,sine))
        for _,inv,sine in sorted(pairs,key=lambda p:-p[0])[:8]:
            evidence['pairs_tried']+=1
            origin,match=register(image,inv,_atlas('26.3'))
            if origin is None:
                continue
            inv[:2]=(inv[:2]+origin[:,None]*inv[2])/16
            camera=_ground(inv,w,h)
            if camera is not None:
                P,f,err=camera
                evidence.update(horizon_px=horizon,gradient_sine=sine,focal_px=f,axis_length_disagreement=err,**match)
                return dict(P=P,scale_status='supported',reason='independent floor gradients and bilateral atlas match',
                            camera_source='horizon-floor',evidence=evidence)
        if len(used)>=3:
            break
    return fail('no metric pair with bilateral sprite registration')
