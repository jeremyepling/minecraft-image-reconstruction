"""Fit four sampling corners from observed pixels of a single surface."""
import numpy as np
from PIL import Image, ImageDraw

from engine.geometry import UNIT, project, transform
from engine.surface_perimeter import _native_alpha


SETTINGS=dict(bins=16, fit_span=(.12,.88), minimum_bins=8)


def envelope(axis, transverse):
    low,high=np.quantile(axis,[.01,.99])
    if high<=low:
        return None
    edges=low+(high-low)*np.linspace(*SETTINGS['fit_span'],SETTINGS['bins']+1)
    centers,lower,upper=[],[],[]
    for a,b in zip(edges,edges[1:]):
        band=(axis>=a)&(axis<b)
        if band.sum()<4:
            continue
        centers.append(float(np.mean(axis[band])))
        tail=np.quantile(transverse[band],[.005,.995])
        lower.append(tail[0])
        upper.append(tail[1])
    if len(centers)<SETTINGS['minimum_bins']:
        return None
    def line(v):
        # Median slope on separated pairs mitigates isolated occluded edge sections.
        x=np.asarray(centers)
        v=np.asarray(v)
        i,j=np.triu_indices(len(x),k=max(2,len(x)//4))
        m=np.median((v[j]-v[i])/(x[j]-x[i]))
        return m,np.median(v-m*x)
    return line(lower),line(upper)


def fit_quad(photo,part,quad):
    if not part.get('support'):
        return None,0.
    native=_native_alpha(part['support'],photo.size)
    if native is None:
        return None,0.
    alpha,(l,t,r,b)=native
    mask=np.asarray(alpha,bool)
    y,x=np.nonzero(mask)
    if len(x)<100:
        return None,0.
    try:
        h=transform(quad)
        uv=project(np.linalg.inv(h),np.column_stack([x+l+.5,y+t+.5]))
        # Bound extrapolation if the supplied chart is unstable.
        if not np.isfinite(uv).all() or np.max(np.abs(uv))>5:
            return None,0.
        rows,cols=envelope(uv[:,0],uv[:,1]),envelope(uv[:,1],uv[:,0])
        if rows is None or cols is None:
            return None,0.
        corners=[]
        for u,v in UNIT.astype(int):
            mx,bx=cols[u]
            my,by=rows[v]
            if abs(1-mx*my)<.2:
                return None,0.
            a=(mx*by+bx)/(1-mx*my)
            corners.append((a,my*a+by))
        result=project(h,corners)
    except np.linalg.LinAlgError:
        return None,0.
    if not np.isfinite(result).all() or (result<0).any() or (result>=photo.size).any():
        return None,0.
    canvas=Image.new('1',photo.size)
    ImageDraw.Draw(canvas).polygon(list(map(tuple,result)),fill=1)
    rect_pixels=int(np.asarray(canvas).sum())
    inside=np.asarray(canvas.crop((l,t,r,b)),bool)
    common=int((inside&mask).sum())
    quality=common/(rect_pixels+len(y)-common)
    return result.tolist(),quality
