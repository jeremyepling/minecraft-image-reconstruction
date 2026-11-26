"""Fit a quadrilateral to near-rectangular visible support, then test repeat consistency."""
import numpy as np

from engine.chart_envelope import SETTINGS as envelope_settings, fit_quad
from engine.masked_count import count, _options
from engine.surface_sample import sample_surface


SETTINGS=dict(**envelope_settings,
              minimum_fit_iou=.82, minimum_valid_fraction=.48,
              cross_band_relative_tolerance=.08, minimum_band_agreement=.48)


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
