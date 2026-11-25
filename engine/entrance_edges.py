"""Propose rectangular entrance perimeters on a fronto-parallel facade image."""
import numpy as np
from PIL import Image, ImageFilter


def _axis_field(image,pitch):
    """Response to a sustained contrast step normal to the x coordinate."""
    r=max(1,round(.06*pitch))
    pad=np.pad(image,((0,0),(r,r),(0,0)),mode='edge')
    jump=pad[:,2*r:]-pad[:,:-2*r]
    half=max(2,round(.22*pitch))
    raw=np.pad(jump,((half,half),(0,0),(0,0)),mode='edge')
    acc=np.concatenate([np.zeros_like(raw[:1]),np.cumsum(raw,axis=0,dtype=np.float64)])
    coherent=(acc[2*half+1:]-acc[:-(2*half+1)])/(2*half+1)
    return np.sqrt((coherent**2).mean(axis=2))


def _peaks(field,pitch):
    # A short isolated highlight has little support along the tangent.
    n=max(4,min(len(field),round(.8*pitch)))
    scores=np.partition(field,-n,axis=0)[-n:].mean(0)
    threshold=max(3.,np.quantile(scores,.45)*1.18)
    order=np.argsort(-scores,kind='stable')
    chosen=[]
    radius=max(3.,.13*pitch)
    for i in order:
        if scores[i]<threshold:
            break
        if i<2 or i>=field.shape[1]-2 or any(abs(j-i)<radius for j in chosen):
            continue
        chosen.append(int(i))
    return np.array(sorted({0,field.shape[1],*chosen}),dtype=int)


def _runs(mask,bridge):
    pos=np.flatnonzero(mask)
    if not len(pos):
        return []
    parts=np.split(pos,np.flatnonzero(np.diff(pos)>bridge+1)+1)
    return [(int(g[0]),int(g[-1]+1)) for g in parts]


def _jamb(vertical,x,u,pitch,threshold):
    """Snap to a vertical line that starts at the lintel and trace its endpoint."""
    h,w=vertical.shape
    if x<.15*pitch or w-x<.15*pitch:
        # A cropped edge cannot locate a jamb; the opposite jamb must supply
        # the height. Keep the extent explicitly clipped to the rectangle.
        return (0 if x<w/2 else w,h,None)
    xs=np.arange(max(1,round(x-.42*pitch)),min(w-1,round(x+.42*pitch)+1))
    ys=np.arange(int(u+.15*pitch),min(h,int(u+.72*pitch)))
    if not len(xs) or not len(ys):
        return None
    line=np.median(vertical[ys[:,None],xs],axis=0)
    scores=line-.36*abs(xs-x)
    k=np.argmax(scores)
    if line[k]<threshold:
        return None
    x=xs[k]
    # Permit mild drift from rectification error while following the same edge.
    profile=vertical[:,max(0,x-1):min(w,x+2)].max(1)
    # A texture that continues uninterrupted through the supposed upper
    # corner is weak evidence for a jamb beginning there.
    previous=profile[max(0,int(u-.6*pitch)):max(0,int(u-.18*pitch))]
    if len(previous) and np.median(previous)>.76*line[k]:
        return None
    cutoff=max(threshold*.72,line[k]*.22)
    runs=_runs(profile[u:]>=cutoff,max(2,round(.18*pitch)))
    if not runs or runs[0][0]>.38*pitch:
        return None
    end=u+runs[0][1]
    if end-u<.83*pitch:
        return None
    # Coherent tangent filters blur the end of a line. The half-response
    # gives a less biased location for an abrupt line termination.
    if end<h-.15*pitch:
        radius=max(2,round(.38*pitch))
        candidates=np.arange(max(u+1,end-radius),min(h-1,end+radius)+1)
        step=max(1,round(.15*pitch))
        jump=profile[(candidates-step).clip(0,h-1)]-profile[(candidates+step).clip(0,h-1)]
        end=int(candidates[np.argmax(jump-.18*abs(candidates-end))])
    return int(x),int(end),float(line[k])


def boxes(rect,pitch):
    """Trace lintels and jambs in a rectified image; return corners in block units."""
    rgb=np.asarray(rect)
    if pitch<4 or rgb.ndim!=3 or min(rgb.shape[:2])<pitch:
        return []
    smooth=np.array(Image.fromarray(rgb.astype('uint8')).convert('RGB').filter(
        ImageFilter.GaussianBlur(float(max(.7,pitch*.043)))),dtype=float)
    vertical=_axis_field(smooth,pitch)
    horizontal=_axis_field(smooth.transpose(1,0,2),pitch)
    ys=_peaks(horizontal,pitch)
    h,w=vertical.shape
    t_h=max(8.,np.median(horizontal)*3.6)
    t_v=max(7.,np.median(vertical)*3.)
    candidates=[]
    for top in ys:
        if not 0<top<h-pitch:
            continue
        field=horizontal[:,top]
        for a,b in _runs(field>=t_h,max(1,round(.14*pitch))):
            if b-a<.82*pitch or b-a>min(.94*w,24*pitch):
                continue
            jambs=[_jamb(vertical,x,top,pitch,t_v) for x in (a,b)]
            if any(j is None for j in jambs):continue
            actual=[j for j in jambs if j[2] is not None]
            if not actual:continue
            x0,_,_=jambs[0];x1,_,_=jambs[1]
            if x1-x0<.8*pitch:continue
            ends=[j[1] for j in actual]
            # A reveal or the floor can hide contrast on just one jamb.
            bottom=max(ends)
            if min(ends)-top<.5*(bottom-top):
                continue
            head=float(np.median(field[a:b]))
            score=np.sqrt(head*min(j[2] for j in actual))
            if len(actual)==1:score*=.8
            candidates.append((score,np.array([x0,top,x1,bottom],float)))
    # Suppress intersecting proposals for the same entrance/hard line.
    accepted=[]
    for _,q in sorted(candidates,key=lambda elem:-elem[0]):
        area=np.prod(q[2:]-q[:2])
        for other in accepted:
            overlap=np.prod(np.maximum(0,np.minimum(q[2:],other[2:])-np.maximum(q[:2],other[:2])))
            if overlap>.28*min(area,np.prod(other[2:]-other[:2])):break
        else:
            accepted.append(q)
    return [(q[:2]/pitch,q[2:]/pitch) for q in accepted]
