"""Trace the visible corner between an entrance and the back of a rectangular opening."""
import numpy as np
from PIL import Image, ImageFilter
from engine.geometry import bilinear


def _gray(rect):
    raw=np.asarray(rect,dtype=np.uint8)
    return np.asarray(Image.fromarray(raw).convert('L').filter(ImageFilter.GaussianBlur(.9)),float)[:,:,None]


def _probe(field,p,r,radius=1.65):
    """Signed edge contrast along a segment; invalidate samples outside the frame."""
    v=r-p
    length=np.linalg.norm(v)
    if length<3:
        return np.zeros(1)
    normal=np.array([-v[1],v[0]])/length
    pts=p+np.linspace(.10,.90,max(7,min(65,round(length))))[:,None]*v
    h,w=field.shape[:2]
    inside=((pts>=radius+1)&(pts<[w-radius-1,h-radius-1])).all(1)
    if inside.sum()<4:
        return np.zeros(1)
    pts=pts[inside]
    return (bilinear(field,pts+normal*radius)-bilinear(field,pts-normal*radius)).ravel()


def _strength(values):
    # Winsorize texture outliers; require a consistent transition along the line.
    cap=max(4.,float(np.quantile(abs(values),.75)))
    return abs(np.clip(values,-cap,cap).mean())


def _snap(field,q,pitch,spacing):
    # Solve opposite rim edges jointly. A bright rear sill shifted inward
    # must not shorten a window when the top edge fixes its front-grid phase.
    bounds=np.r_[q.min(0),q.max(0)].astype(float)
    shifts=np.arange(-.4*pitch,.4*pitch+1)
    costs=[]
    for k,(i,j) in enumerate(((3,0),(0,1),(1,2),(2,3))):
        direction=np.eye(2)[k%2]
        p,r=q[i],q[j]
        d=r-p
        center=p+.13*d; end=r-.13*d
        values=[]
        for t in shifts:
            v=_probe(field,center+t*direction,end+t*direction)
            # Cap the pull from a bright far rim or floor edge.
            values.append(min(14.,max(0,-np.median(v)))-.32*abs(t))
        costs.append(np.array(values))
    for k in (0,1):
        target=(spacing[k]-1)*(bounds[k+2]-bounds[k])
        mismatch=(shifts[None,:]-shifts[:,None])-target
        both=costs[k][:,None]+costs[k+2][None,:] - 1.8*abs(mismatch)
        a,b=np.unravel_index(np.argmax(both),both.shape)
        bounds[k]+=shifts[a]; bounds[k+2]+=shifts[b]
    return bounds

def _segment(field,p,r,bounds,normal_slack):
    """Clip an edge to the part actually visible inside the aperture."""
    v=r-p; length=np.linalg.norm(v)
    if length<4:return 0.
    low=0.;high=1.
    for k in range(2):
        if abs(v[k])<.001:
            if not bounds[k]<=p[k]<=bounds[k+2]:return 0.
        else:
            a,b=sorted([(bounds[k]-p[k])/v[k],(bounds[k+2]-p[k])/v[k]])
            low=max(low,a);high=min(high,b)
    if (high-low)*length<5:return 0.
    r=p+high*v; p=p+low*v
    normal=np.array([-v[1],v[0]])/length
    vals=[]
    for delta in np.arange(-normal_slack,normal_slack+.1,1):
        vals.append(_strength(_probe(field,p+normal*delta,r+normal*delta))-.8*abs(delta))
    return max(0,max(vals))


def predict(rect,q,vp,distance,pitch,spacing):
    """Score corner/return agreement for integer recess depths in rectified pixels."""
    field=_gray(rect)
    vp=np.asarray(vp,dtype=float)
    q=np.asarray(q,dtype=float)
    bounds=_snap(field,q,pitch,spacing)
    if np.min(bounds[2:]-bounds[:2])<pitch*.65:
        return dict(depth=None,reason='rim too small')
    proposals=[bounds,np.r_[q.min(0),q.max(0)]]
    if 0<rect.shape[0]-bounds[3]<1.3*pitch:
        # Thresholded floor pixels can shorten a ground-level door. Consider
        # the facade baseline as a second entrance edge, and use only visible
        # segments as evidence for either hypothesis.
        base=bounds.copy()
        base[3]=_facade_base(field,bounds,pitch)
        proposals.append(base)
    hypotheses=[]
    for outer in proposals:
        left,top,right,bottom=outer
        corners=np.array([[left,top],[right,top],[right,bottom],[left,bottom]])
        # Only one corner ray enters the rectangle through both axes.
        corner=np.where(((corners[:,0]==left)==(vp[0]>(left+right)/2))&
                        ((corners[:,1]==top)==(vp[1]>(top+bottom)/2)))[0][0]
        prev=(corner-1)%4; nxt=(corner+1)%4
        near=corners[corner]
        for depth in range(1,9):
            back=vp+(corners-vp)/(1+depth/distance)
            far=back[corner]
            delta=near-far
            visible_size=np.minimum(outer[2:],back.max(0))-np.maximum(outer[:2],back.min(0))
            if min(visible_size)<.55*pitch:
                continue
            if np.min(abs(delta))<2:
                continue
            interior=outer+np.array([2,2,-2,-2])
            cross=[_segment(field,far,back[k],interior,1.) for k in (prev,nxt)]
            diagonal=_segment(field,near,far,outer,1.)
            strongest=max(cross); weaker=min(cross)
            if diagonal<5 or strongest<6:
                continue
            out=far+(far-near)*min(.5,8/np.linalg.norm(delta))
            after=_segment(field,far,out,interior,0.)
            if after>max(10.,.55*diagonal):
                continue
            # A real ray ends at the back plane. A continuing diagonal (for
            # instance a vein in a flat sprite) carries less junction evidence.
            score=(strongest+.45*weaker)*min(1,diagonal/12)*min(1,(diagonal+6)/(after+6))
            hypotheses.append((score,depth,outer.copy(),far.copy(),near.copy(),cross,diagonal,after))
    hypotheses.sort(key=lambda v:-v[0])
    if not hypotheses:return dict(depth=None,reason='no supported return',snapped_bounds=bounds.tolist())
    s,d,box,far,near,cross,diag,after=hypotheses[0]
    runner=next((a[0] for a in hypotheses if a[1]!=d),0.)
    evidence=dict(snapped_bounds=box.tolist(),corner=near.tolist(),back_corner=far.tolist(),
                  rim_strength=np.round(cross,2).tolist(),diagonal_strength=round(diag,2),
                  continuation_strength=round(after,2),score=round(s,2),margin=round(s-runner,2))
    if s<8 or s-runner<max(1.1,.08*s):
        return dict(depth=None,reason='ambiguous junction',**evidence)
    return dict(depth=int(d),**evidence)

def _facade_base(field,bounds,pitch):
    """Look for the wall/floor seam in flanking wall, outside the doorway."""
    h,w=field.shape[:2]
    x0,y0,x1,y1=bounds
    flanks=[]
    for a,b in ((x0-.75*pitch,x0-.16*pitch),(x1+.16*pitch,x1+.75*pitch)):
        positions=np.arange(max(2,a),min(w-2,b))
        if len(positions)>4:
            flanks.append(positions.astype(int))
    if not flanks:
        return h
    xs=np.concatenate(flanks)
    ys=np.arange(max(round(h-1.3*pitch),2),h-2)
    if not len(ys):
        return h
    grad=field[ys+2][:,xs,0]-field[ys-2][:,xs,0]
    score=abs(np.median(grad,axis=1))
    peak=score.argmax()
    return float(ys[peak]) if score[peak]>14 else h
