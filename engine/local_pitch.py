"""Measure projected block spacing from repeated texture in local image windows."""
import numpy as np


def _peaks(patch, inside, axis):
    """Autocorrelation stays local so convergence and setbacks can vary across the image."""
    patch = patch if axis == 0 else patch.T
    inside = inside if axis == 0 else inside.T
    h, w = patch.shape
    # Remove illumination gradients and directional edges, retaining fine texture.
    detail = np.zeros_like(patch)
    detail[2:-2,2:-2] = patch[2:-2,2:-2] - (
        patch[:-4,2:-2] + patch[4:,2:-2] + patch[2:-2,:-4] + patch[2:-2,4:])/4
    detail = np.clip(detail,-32,32)
    window = np.outer(np.hanning(h),np.hanning(w))**.5 * inside
    detail *= window
    fft = np.fft.rfft2(detail)
    ac = np.fft.irfft2(fft*fft.conj(),s=patch.shape,axes=(0,1)).real
    if ac[0,0] < 1e-4:
        return []
    ac /= ac[0,0]
    max_lag = min(w//2-1,128)
    lag = np.arange(6,max_lag+1)
    if len(lag) < 3:
        return []
    offset = np.arange(-min(h//3,max_lag//2),min(h//3,max_lag//2)+1)
    band = ac[offset[:,None],lag]
    band = np.where(abs(offset[:,None])<=.48*lag,band,-1)
    # For each displacement retain the strongest supported transverse offset.
    profile = band.max(0)
    peaks = np.nonzero((profile[1:-1]>profile[:-2]) & (profile[1:-1]>=profile[2:]))[0]+1
    peaks = sorted(peaks,key=lambda i: -profile[i])[:4]
    if not peaks:
        return []
    best = profile[peaks[0]]
    return [(float(lag[i]),float(profile[i])) for i in peaks if profile[i]>.14 and profile[i]>.65*best]


def _texel_vote(patch, inside, axis):
    """Periodic jumps sample the native 16-texel art even where the motif does not repeat."""
    if axis:
        patch,inside=patch.T,inside.T
    signal=np.abs(np.diff(patch[::3],axis=1))
    mask=inside[::3,:-1] & inside[::3,1:]
    keep=mask.sum(1)>24
    if not np.any(keep):
        return []
    jumps,valid=signal[keep],mask[keep]
    jumps=np.minimum(jumps,24)*valid
    centered=(jumps-jumps.sum(1,keepdims=True)/valid.sum(1,keepdims=True))*valid
    taper=np.hanning(centered.shape[1])
    size = 1024
    spec=np.abs(np.fft.rfft(centered*taper,n=size,axis=1)).sum(axis=0)
    freqs=np.fft.rfftfreq(size)
    bins=np.flatnonzero((freqs>.105)&(freqs<.49))
    i=bins[np.argmax(spec[bins])]
    # Reject diffuse texture spectra and completely smooth patches.
    noise=np.median(spec[bins])
    signal=spec[i]/max(1,noise)
    if signal < 2.5:
        return []
    return [(float(16/freqs[i]),float(min(1,(signal-1)/6)))]


def _votes(rgb, mask):
    y,x=np.nonzero(mask)
    if not len(x):
        return [[], []], [[], []]
    gray=np.asarray(rgb,dtype=np.float32) @ np.array([.213,.715,.072],dtype=np.float32)
    x0,x1=x.min(),x.max()+1
    y0,y1=y.min(),y.max()+1
    votes = [[], []]
    native = [[], []]
    patch_size=min(192,max(48,int(max(x1-x0,y1-y0)*.55)))
    for cy in np.linspace(y0,y1-1,6):
        for cx in np.linspace(x0,x1-1,7):
            ystart=max(0,min(gray.shape[0]-patch_size,int(cy-patch_size//2)))
            xstart=max(0,min(gray.shape[1]-patch_size,int(cx-patch_size//2)))
            region=np.s_[ystart:ystart+patch_size,xstart:xstart+patch_size]
            valid=mask[region]
            if valid.mean() < .21:
                continue
            for axis in (0,1):
                votes[axis].extend(_peaks(gray[region],valid,axis))
                native[axis].extend(_texel_vote(gray[region],valid,axis))
    return votes, native


def _support(votes, grid):
    if not votes:
        return np.zeros_like(grid)
    pitch,weight=np.array(votes).T
    kernel=np.exp(-.5*(np.log(grid[:,None]/pitch)/.075)**2)
    density=kernel @ weight**3
    return density / max(density.max(),1e-8)


def estimate_pitch(rgb, mask):
    """Return horizontal and vertical block spacing in source pixels, or None."""
    repeated,native=_votes(rgb,np.asarray(mask,dtype=bool))
    if any(not a and not b for a,b in zip(repeated,native)):
        return None
    grid = np.geomspace(6, 160, 200)
    priors = []
    for axis in (0,1):
        a=_support(repeated[axis],grid)
        b=_support(native[axis],grid)
        priors.append(.60*a+.40*b+.40*np.sqrt(a*b))
    ratio=np.log(grid[:,None]/grid)
    # Both axes on the visible front face project the same physical cube.
    # Permit foreshortening; a very thin projected axis needs strong direct support.
    joint=priors[0][:,None]*priors[1][None,:]*np.exp(-.5*(ratio/.45)**2)
    x,y=np.unravel_index(joint.argmax(),joint.shape)
    return float(grid[x]),float(grid[y])
