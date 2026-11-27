"""Native-pixel floor atlas fit. All inference inputs come from the one RGB and release atlas."""
import numpy as np
import cv2
from engine.camera_atlas_floor import _atlas


def patches(image, bounds=((.04,.39),(.61,.84))):
    """Separate source regions; remove slow illumination before pooling by texel."""
    a = np.asarray(image.convert('RGB')).mean(2).astype(np.float32)
    h, w = a.shape
    low = int(.76*h)
    region = a[low:]
    center = np.median(region[::3, ::5])
    lo, hi = center-4, center+4
    good = cv2.erode(((region >= lo) & (region <= hi)).astype(np.uint8), np.ones((11,11),np.uint8))
    region = np.clip(region, lo, hi)
    signal = region-cv2.GaussianBlur(region, (0,0), 6)
    result = []
    for x0, x1 in bounds:
        y,x=np.mgrid[int(.79*h):int(.98*h):3,int(x0*w):int(x1*w):3]
        keep = good[y-low,x].astype(bool)
        x,y=x[keep],y[keep]
        screen=np.array([x.ravel()+.5,y.ravel()+.5,np.ones(x.size)])
        result.append((screen, signal[y-low,x].ravel()))
    return result


def templates(atlas, ids, n):
    # Gamma-aware box mip approximation; native nearest or bilinear sampled below.
    tiles = atlas.tiles[ids].astype(float)
    mip = (tiles**2.2).reshape(len(ids),n,16//n,n,16//n,3).mean((2,4))**(1/2.2)
    gray=mip.mean(-1)
    return np.stack([np.rot90(a,k,axes=(-2,-1)) for a in (gray,gray[:,:,::-1]) for k in range(4)],axis=1)


def coords(inv, points, n):
    uv=inv@points
    return n*uv[:2]/uv[2]


def sample_uv(tile, uv, mode):
    n=tile.shape[0]
    if mode=='nearest':
        i,j=np.floor(uv).astype(int)%n
        return tile[j,i]
    u,v=uv-.5
    i,j=np.floor([u,v]).astype(int)
    a,b=u-i,v-j
    return ((1-b)*((1-a)*tile[j%n,i%n]+a*tile[j%n,(i+1)%n])+
            b*((1-a)*tile[(j+1)%n,i%n]+a*tile[(j+1)%n,(i+1)%n]))


def correlation(a,b):
    a,b=a-a.mean(), b-b.mean()
    return float((a@b)/max(1.e-9,np.linalg.norm(a)*np.linalg.norm(b)))


def atlas_scan(inv, patches_, atlas, ids):
    """Search models on the west patch. Score its unmodified prediction on the east."""
    best=None
    for n in (16,8,4,2):
        tiles=templates(atlas,ids,n)
        uv=coords(inv,patches_[0][0],n)
        signal=patches_[0][1]
        # FFT over integer phase is exactly a native nearest-texel dot product.
        i,j=np.floor(uv).astype(int)%n
        index=j*n+i
        observed=np.bincount(index,signal-signal.mean(),minlength=n*n).reshape(n,n)
        counts=np.bincount(index,minlength=n*n).reshape(n,n)
        def dots(a,b):
            return np.fft.irfft2(np.conj(np.fft.rfft2(a))*np.fft.rfft2(b),s=(n,n),axes=(-2,-1))
        sums=dots(counts,tiles)
        variances=dots(counts,tiles**2)-sums**2/len(signal)
        corr=dots(observed,tiles)/np.maximum(1e-6,np.sqrt(np.maximum(0,variances))*np.linalg.norm(signal-signal.mean()))
        idx=np.unravel_index(corr.argmax(),corr.shape)
        ti,orient,dy,dx=map(int,idx)
        tile=tiles[ti,orient]
        for mode in ('nearest','linear'):
            phase=np.array([[dx],[dy]])
            scores=[correlation(target,sample_uv(tile,coords(inv,screen,n)+phase,mode))
                    for screen,target in patches_]
            if best is None or scores[0]>best['fit']:
                best=dict(fit=scores[0],hold=scores[1],n=n,mode=mode,orientation=orient,
                          phase=[dx,dy],tile=atlas.labels[ids[ti]])
    return best
