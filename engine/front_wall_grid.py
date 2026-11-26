"""Recover a local grid from a frame-filling, front-facing vanilla texture."""
import numpy as np

from engine.camera_atlas_floor import _atlas


def texel_axis(rgb, axis):
    """Estimate texel-edge spacing and phase from native-pixel colour changes."""
    changes = np.any(np.diff(rgb, axis=axis) != 0, axis=2).mean(axis=1-axis)
    # Suppress isolated object boundaries. Regular low-contrast texel steps still vote.
    weights = np.minimum(changes, .40)
    if np.count_nonzero(weights > .03) < 20:
        return None
    nfft = 1 << (len(weights) * 24).bit_length()
    spectrum = np.abs(np.fft.rfft(weights, n=nfft))
    lo, hi = int(nfft / min(40., len(weights)/32)), int(nfft / 3.5)
    if hi <= lo:
        return None
    i = lo + int(np.argmax(spectrum[lo:hi]))
    scale = nfft/i
    pos = np.arange(1, len(weights)+1)
    wave = np.sum(weights*np.exp(2j*np.pi*pos/scale))
    if abs(wave)/weights.sum() < .32:
        return None
    # Fit observed edge locations to integer texel indexes, allowing for pixel rounding.
    origin = np.angle(wave)*scale/(2*np.pi)
    index = np.rint((pos-origin)/scale)
    keep = (weights > .025) & (np.abs(pos-origin-index*scale) < 1.25)
    if keep.sum() < 20:
        return None
    scale, origin = np.polyfit(index[keep], pos[keep], 1, w=np.sqrt(weights[keep]))
    return float(scale), float(origin)


def _fold(rgb, xs, ys, xi, yi):
    """Remove broad lighting variation by subtracting each complete tile's mean."""
    values = rgb[np.ix_(ys, xs)].mean(2)
    tiles = []
    for row in range(0,len(ys)-15,16):
        for col in range(0,len(xs)-15,16):
            tile = values[row:row+16,col:col+16]
            tiles.append(tile-tile.mean())
    if len(tiles) < 3:
        return None
    patch = np.median(tiles,axis=0)
    patch = np.roll(patch,(yi[0]%16, xi[0]%16),axis=(0,1))
    return patch


def _match(patch, atlas):
    if patch is None or patch.std() < .11:
        return None
    spectra = np.conj(np.fft.rfft2(patch - patch.mean()))*atlas.spectra
    corr = np.fft.irfft2(spectra,s=(16,16),axes=(-2,-1))/(256*patch.std())
    corr /= np.maximum(.01,atlas.patterns.std(axis=(2,3)))[:,:,None,None]
    # Compete on phase. Multiple block IDs may have the same correctly aligned art.
    offset = corr.max(axis=(0,1))
    y, x = np.unravel_index(offset.argmax(),offset.shape)
    s, o = np.unravel_index(corr[:,:,y,x].argmax(),corr.shape[:2])
    if offset[y,x] < .85 or offset[y,x]-np.partition(offset.ravel(),-2)[-2] < .035:
        return None
    return (int(s),int(o),int(x),int(y)),float(offset[y,x])


def _centres(length, fit):
    pitch, origin = fit
    ids = np.arange(-2,int(length/pitch)+3)
    coords = np.floor(origin+(ids+.5)*pitch).astype(int)
    valid = (coords>=0)&(coords<length)
    return coords[valid],ids[valid]


def local_grid(image, version='26.3', atlas_path=None):
    rgb = np.asarray(image.convert('RGB'))
    h,w = rgb.shape[:2]
    xfit,yfit = texel_axis(rgb,1),texel_axis(rgb,0)
    if xfit is None or yfit is None or abs(xfit[0]/yfit[0]-1) > .008:
        return set(), {}
    atlas = _atlas(version, atlas_path)
    if atlas is None:
        return set(), {}
    xs,xi = _centres(w,xfit)
    ys,yi = _centres(h,yfit)
    matches = []
    for half in (slice(0,len(xs)//2),slice(len(xs)//2,None)):
        match = _match(_fold(rgb,xs[half],ys,xi[half],yi),atlas)
        if match is None:
            return set(), {}
        matches.append(match)
    if matches[0][0] != matches[1][0]:
        return set(), {}
    (sid,orientation,dx,dy),_ = matches[0]
    template = atlas.patterns[sid,orientation]
    # Pixel-center witnesses also cover clipped first/last texels at image borders.
    def samples(length, fit, delta):
        tex = np.floor((np.arange(length)+.5-fit[1])/fit[0]).astype(int)+delta
        ids,start,counts = np.unique(tex,return_index=True,return_counts=True)
        return ids,start+counts//2
    tx,px = samples(w,xfit,dx)
    ty,py = samples(h,yfit,dy)
    observed = rgb[np.ix_(py,px)].mean(2)
    cells = set()
    for by in np.unique(ty//16):
        iy = ty//16==by
        for bx in np.unique(tx//16):
            ix = tx//16==bx
            actual = observed[np.ix_(iy,ix)].ravel()
            expected = template[np.ix_(ty[iy]%16,tx[ix]%16)].ravel()
            if len(actual)<4 or actual.std()<.12 or expected.std()<.12:
                continue
            score = np.mean((actual-actual.mean())*(expected-expected.mean()))/(actual.std()*expected.std())
            if score >= .78:
                cells.add((int(bx),int(-by),0))
    return cells,dict(sprite=atlas.labels[sid],orientation=orientation,phase=[dx,dy],
                      texel_fits=[xfit,yfit],half_correlations=[m[1] for m in matches])

