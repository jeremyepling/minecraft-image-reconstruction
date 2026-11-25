"""Find dark entrance candidates on a rectified facade using local shading and rim support."""
import numpy as np
from PIL import Image


def _components(mask):
    remaining=mask.copy()
    for row,col in zip(*remaining.nonzero()):
        if not remaining[row,col]:continue
        remaining[row,col]=False
        stack=[(row,col)];coords=[]
        while stack:
            y,x=stack.pop()
            coords.append((y,x))
            for j,i in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=j<mask.shape[0] and 0<=i<mask.shape[1] and remaining[j,i]:
                    remaining[j,i]=False
                    stack.append((j,i))
        yield np.asarray(coords)


def _rim_score(facade,lo,hi,sub):
    x0,y0=lo*sub
    x1,y1=hi*sub
    # Pixels near the nominal block line may be misregistered by a fraction of
    # a block. Look across this narrow corridor along the full side of the box.
    strengths=[]
    for im,start,end,axis_start,axis_end in (
            (facade,y0,y1,x0,x1),(facade.T,x0,x1,y0,y1)):
        for coordinate,sign in ((start,1),(end,-1)):
            if coordinate<2 or coordinate>len(im)-3:
                continue
            column=np.arange(axis_start+1,axis_end-1)
            options=[]
            for shift in (-1,0,1):
                center=coordinate+shift
                inner=im[np.clip(center+sign,0,len(im)-1),column]
                outer=im[np.clip(center-sign,0,len(im)-1),column]
                # One long bright texel near a corner cannot carry a side.
                change=outer-inner
                options.append(np.quantile(change,.35))
            strengths.append(max(options))
    if len(strengths)<2:return 0.
    strengths.sort(reverse=True)
    return max(0,float(strengths[1]))


def _lintels(lum,contrast,sub):
    ny,nx=lum.shape[0]//sub,lum.shape[1]//sub
    def centers(image):
        return image.reshape(ny,sub,nx,sub)[:,1:-1,:,1:-1].mean((1,3))
    block=centers(lum)
    shade=centers(contrast)
    guesses=[]
    for y in range(1,ny-1):
        # Each uninterrupted dark transition below a bright lintel starts a
        # separate aperture, even when its sill connects to its neighbor.
        start=(block[y-1]-block[y]>.085)&(shade[y]>.05)
        breaks=np.diff(np.r_[False,start,False].astype(int))
        for left,right in zip(np.flatnonzero(breaks==1),np.flatnonzero(breaks==-1)):
            if right-left<2:continue
            bottom=y+1
            while bottom<ny and np.quantile(shade[bottom,left:right],.63)>.045:
                bottom+=1
            if bottom-y<2:continue
            lo=np.array([left,y]);hi=np.array([right,bottom])
            guesses.append((lo,hi))
    return guesses


def boxes(rect,pitch):
    """Return disjoint whole-cell candidate rectangles as (low_xy,high_xy) arrays.

    They propose likely regions for geometric verification. Brightness alone
    cannot decide whether an entrance has any depth.
    """
    p=float(pitch)
    if p<=0:raise ValueError('Positive pixel pitch required')
    ny,nx=np.rint(np.array(rect.shape[:2])/p).astype(int)
    if min(nx,ny)<2:return []
    sub=4
    grid=np.asarray(Image.fromarray(np.uint8(rect[:,:,:3])).resize(
        (nx*sub,ny*sub),Image.Resampling.BOX),float)
    lum=np.log(grid@np.array([.213,.715,.072])+4)
    # A local row reference prevents a building-wide exposure slope from
    # turning half the facade into one proposal. Multiple levels separate
    # neighboring apertures joined by a shaded band.
    radius=min(6*sub,(nx*sub-1)//2)
    padded=np.pad(lum,((0,0),(radius,radius)),mode='edge')
    local=np.quantile(np.lib.stride_tricks.sliding_window_view(padded,2*radius+1,axis=1),.79,axis=-1)
    reference=.75*local+.25*np.quantile(lum,.78,axis=1,keepdims=True)
    contrast=reference-lum
    scores={}
    # Lintels have better extent evidence than an irregular patch of dark
    # texture. Preserve their full span when a threshold isolates only a jamb.
    for low,high in _lintels(lum,contrast,sub):
        scores[(*low,*high)]=.56+min(.30,_rim_score(lum,low,high,sub))
    for threshold in (.055,.09,.14,.21,.30):
        for group in _components(contrast>=threshold):
            if len(group)<sub**2:continue
            lo=group.min(axis=0)[::-1]
            hi=group.max(axis=0)[::-1]+1
            extent=hi-lo
            if min(extent)<.75*sub or len(group)<.42*extent.prod():continue
            low=np.clip(np.rint(lo/sub).astype(int),0,[nx,ny])
            high=np.clip(np.rint(hi/sub).astype(int),0,[nx,ny])
            if min(high-low)<1 or max(high-low)<2:continue
            # Bright sills or floors can detach the last row from a doorway.
            # Test one-cell lower and higher bounds before adjudicating the rim.
            for dy in (-1,0,1):
                end=np.array([high[0],min(ny,high[1]+dy)])
                if min(end-low)<1 or np.prod(end-low)<3:continue
                rim=_rim_score(lum,low,end,sub)
                if rim<.035:continue
                inside=contrast[low[1]*sub:end[1]*sub,low[0]*sub:end[0]*sub]
                fill=float(np.mean(inside>.065))
                if fill<.48:continue
                score=rim+.20*fill+.025*(dy==0)
                key=(*low,*end)
                scores[key]=max(score,scores.get(key,0.))
    selected=[]
    for key in sorted(scores,key=lambda k:-scores[k]):
        lo,hi=np.array(key[:2]),np.array(key[2:])
        area=np.prod(hi-lo)
        if any(np.prod(np.maximum(0,np.minimum(hi,b)-np.maximum(lo,a)))>.39*min(area,np.prod(b-a))
               for a,b in selected):
            continue
        selected.append((lo,hi))
    return sorted(selected,key=lambda pair:(pair[0][1],pair[0][0]))
