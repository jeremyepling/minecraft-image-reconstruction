"""Check whether rays through a proposed block still show the observed floor."""
import numpy as np

from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE

FEATURES = ('floor_color_match','floor_phase_correlation','floor_evidence_fraction','floor_model_agreement')


def _tile(codes, color):
    counts=np.bincount(codes,minlength=256)
    if np.min(counts)<3:
        return None
    # A per-phase median ignores sparse foreground and shadow pixels in the floor strip.
    out=[]
    for channel in color.T:
        hist=np.bincount(256*codes+channel,minlength=256*256).reshape(256,256)
        out.append(np.argmax(hist.cumsum(1)>=counts[:,None]*.5,axis=1))
    return np.column_stack(out)


def floor_features(image,camera,cells):
    xyz=np.asarray(sorted(map(tuple,cells)),dtype=int).reshape(-1,3)
    zeros=np.zeros((len(xyz),len(FEATURES)))
    if not len(xyz):
        return zeros
    P=np.array(camera,dtype=float)
    P/=np.linalg.norm(P[2,:3])
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    if eye[1]<=0:
        return zeros
    w,h=image.size
    shape=tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in (h,w))
    yy,xx=np.indices(shape)
    px=xx*SAMPLE_STRIDE+SAMPLE_STRIDE//2
    py=yy*SAMPLE_STRIDE+SAMPLE_STRIDE//2
    valid=(px<w)&(py<h)
    rays=np.stack((px+.5,py+.5,np.ones(shape)),axis=2) @ np.linalg.inv(P[:,:3]).T
    t=np.divide(-eye[1],rays[:,:,1],out=np.zeros(shape),where=np.abs(rays[:,:,1])>1.e-7)
    valid &= (t>0) & (t<150)
    ground=eye+t[:,:,None]*rays
    phase=np.floor(ground[:,:,[0,2]]*16).astype(np.int64)%16
    code=phase[:,:,0]*16+phase[:,:,1]
    colors=np.asarray(image.convert('RGB'))[py.clip(0,h-1),px.clip(0,w-1)].astype(int)
    strip=valid & (py>h*.78) & (py<h*.97)
    tiles=[]
    for mask in (strip & (px<w*.40),strip & (px>w*.60)):
        tiles.append(_tile(code[mask],colors[mask]))
    if any(t is None for t in tiles):
        return zeros
    agreement=np.mean(np.max(np.abs(tiles[0]-tiles[1]),axis=1)<=2)
    if agreement<.75:
        return zeros
    tile=(tiles[0]+tiles[1])/2
    lookup=tile[code]
    owner=_frontmost_cells(list(map(tuple,xyz)),P,eye,shape)
    ys,xs=np.where((owner>=0)&valid)
    idx=owner[ys,xs]
    n=np.bincount(idx,minlength=len(xyz))
    if not len(idx):
        return zeros
    def grouped(values):
        return np.bincount(idx,weights=values,minlength=len(xyz))/np.maximum(1,n)
    seen=colors[ys,xs]
    reference=lookup[ys,xs]
    matches=np.max(np.abs(seen-reference),axis=1)<=2
    gray,expected=seen.mean(1),reference.mean(1)
    cov=grouped(gray*expected)-grouped(gray)*grouped(expected)
    var_a=np.maximum(0,grouped(gray**2)-grouped(gray)**2)
    var_b=np.maximum(0,grouped(expected**2)-grouped(expected)**2)
    corr=np.divide(cov,np.maximum(.1,np.sqrt(var_a*var_b)))
    cov_frac=n/np.maximum(1,np.bincount(owner[owner>=0],minlength=len(xyz)))
    return np.column_stack((grouped(matches),corr,cov_frac,np.full(len(xyz),agreement)))
