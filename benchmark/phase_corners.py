"""Challenge the texture-guided crop's pitch against a versioned atlas."""
import numpy as np

from benchmark import portico_atlas
from engine import atlas_count, texture_guided_count
from engine.geometry import transform


SETTINGS = dict(texture_guided_count.SETTINGS, minimum_atlas_train=.55, minimum_atlas_hold=.45,
                minimum_atlas_margin=.07, atlas_shortlist=20, atlas_max_observations=2000)
METHOD = 'held-strip-texture-phase-corners'


def tile_check(sample,bands,counts):
    source=portico_atlas.atlas()
    rgb=sample['rgb']
    train=bands[0]|bands[1]
    ids=atlas_count._shortlist(rgb,train,source)[0][:SETTINGS['atlas_shortlist']]
    templates=source.spectra[ids],np.fft.rfft2(source.patterns[ids]**2)
    size=np.array(train.shape[::-1])
    def scores(observed,scale,subset,patterns):
        y,x=np.nonzero(observed)
        sel=np.linspace(0,len(y)-1,min(len(y),SETTINGS['atlas_max_observations'])).astype(int)
        y,x=y[sel],x[sel]
        if len(x)<SETTINGS['minimum_pairs'] or min(scale)<4:
            return None
        return portico_atlas._correlations(rgb[y,x].mean(-1),x,y,scale,subset,patterns)
    period=size/np.asarray(counts)
    fit=scores(train,period,ids,templates)
    if fit is None:
        return {}
    k,rotation,dy,dx=np.unravel_index(fit.argmax(),fit.shape)
    selected=float(fit[k,rotation,dy,dx])
    alt=[]
    for factors in ((.5,1.),(1.,.5),(.5,.5),(2.,1.),(1.,2.),(2.,2.)):
        other=scores(train,period*factors,ids,templates)
        if other is not None:
            alt.append(float(other.max()))
    held=scores(bands[2],period,ids[k:k+1],tuple(a[k:k+1] for a in templates))
    check=float(held[0,rotation,dy,dx]) if held is not None else -1.
    margin=selected-max(alt,default=1.)
    return dict(tile=source.labels[ids[k]],train=selected,held=check,margin=margin,
                accepted=selected>=SETTINGS['minimum_atlas_train'] and check>=SETTINGS['minimum_atlas_hold']
                         and margin>=SETTINGS['minimum_atlas_margin'])


def predict_surface(photo, part, sample, baseline):
    fit,chart=texture_guided_count.count_surface(photo,part,sample,baseline)
    log=dict(fit['evidence'].get('texture_check',{}))
    result=dict(method=METHOD,counts=baseline['counts'],phase=baseline['phase'],evidence=log)
    if fit['method']!=texture_guided_count.METHOD:
        return result
    log['self_repeat']=dict(quad=chart['quad'], counts=fit['counts'], phase=fit['phase'])
    axis=int(sample['mask'].shape[0]>sample['mask'].shape[1])
    bands=texture_guided_count.fixed_bands(chart,np.linalg.inv(transform(sample['quad'])),axis)
    tile=tile_check(chart,bands,fit['counts'])
    log['atlas']=tile
    if tile.get('accepted'):
        result.update(log['self_repeat'])
    return result
