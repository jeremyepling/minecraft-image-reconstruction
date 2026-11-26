"""Fit a quadrilateral to near-rectangular visible support, then test repeat consistency."""
import numpy as np
from PIL import Image, ImageDraw

from engine.geometry import UNIT, project, transform
from engine.masked_count import count, _options
from engine.surface_perimeter import _native_alpha
from engine.surface_sample import sample_surface


SETTINGS=dict(bins=16, fit_span=(.12,.88), minimum_bins=8,
              minimum_fit_iou=.82, minimum_valid_fraction=.48,
              cross_band_relative_tolerance=.08, minimum_band_agreement=.48)


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


def verify_bands(sample,axis,proposal):
    # Split only across the direction perpendicular to a repeat. Width/height and
    # rectification stay unchanged, so estimates from both halves share units.
    mask=sample['mask']
    extent=mask.shape[axis]  # across -> split rows, down -> split columns
    reliability=[]
    for half in (0,1):
        active=mask.copy()
        selection=slice(extent//2,None) if half==0 else slice(None,extent//2)
        if axis==0:
            active[selection,:]=False
        else:
            active[:,selection]=False
        options,_,_=_options(dict(sample,mask=active),axis)
        score=max((p['agreement'] for p in options
                   if abs(np.log(p['count']/proposal))<=SETTINGS['cross_band_relative_tolerance']),default=0.)
        reliability.append(float(score))
    return reliability


def predict_surface(photo,part,sample,baseline):
    original=baseline['counts']
    result=dict(method='mask-envelope-repeat',counts=original,phase=baseline['phase'],evidence={})
    log=result['evidence']
    if sample is None or all(k is not None for k in original):
        log['reason']='missing crop or complete original size'
        return result
    quad,overlap=fit_quad(photo,part,sample['quad'])
    log['fit_iou']=float(overlap)
    if quad is None or overlap<SETTINGS['minimum_fit_iou']:
        log['reason']='weak agreement between predicted mask and fitted outline'
        return result
    new=sample_surface(photo,dict(part,quad=quad))
    if new is None or new['mask'].mean()<SETTINGS['minimum_valid_fraction']:
        log['reason']='too little valid texture after straightening'
        return result
    candidate=count(new)
    log.update(fitted_quad=quad,valid_fraction=float(new['mask'].mean()),fitted_counts=candidate['counts'],
               chart_wh=list(new['mask'].shape[::-1]))
    if None in candidate['counts']:
        log['reason']='counter could not resolve both axes on fitted chart'
        return result
    votes={}
    for axis,p in enumerate(candidate['evidence']['axes']):
        if p.get('source')=='repeated texture':
            votes[str(axis)]=verify_bands(new,axis,candidate['counts'][axis])
    log['band_scores']=votes
    if not votes or any(min(v)<SETTINGS['minimum_band_agreement'] for v in votes.values()):
        log['reason']='independent perpendicular bands did not confirm an observed repeat'
        return result
    log['reason']='accepted fitted chart and repeated texture'
    result.update(quad=quad,counts=candidate['counts'],phase=candidate['phase'])
    return result
