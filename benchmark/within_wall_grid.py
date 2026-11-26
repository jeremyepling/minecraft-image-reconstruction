"""Measure a local tile spacing and extrapolate it within a single visible surface."""
import numpy as np

from engine.texture_guided_count import shift_score
from engine.geometry import transform
from engine.masked_count import _repeat_peaks
from engine.surface_pitch import _aspect, _phase


SETTINGS = dict(min_pairs_score=.58, held_repeat=.49, integer_residual=.24,
                flat_edge_phase_error=.16, minimum_cycles=2.1, minimum_coverage=.45,
                minimum_donor_contrast=2.4, flat_contrast_ratio=.65, partial_tolerance=.45)
METHOD = 'within-surface-patch-grid'


def box(axis,start,stop,side_start,side_stop):
    return [start,side_start,stop,side_stop] if axis==0 else [side_start,start,side_stop,stop]


def best_patch(sample,axis,settings):
    rgb,valid=sample['rgb'],sample['mask']
    if axis:
        rgb,valid=rgb.transpose(1,0,2),valid.T
    height,width=valid.shape
    if width<36 or height<6:
        return None
    cut=width//2
    # Select an end and a transverse band using local valid-pixel contrast alone.
    # No period or template is fitted on the opposite end.
    areas=[]
    for x0,x1 in ((0,cut),(cut,width)):
        for y0,y1 in ((0,height//2),(height//2,height),(0,height)):
            colors=rgb[y0:y1,x0:x1].mean(2)
            mask=valid[y0:y1,x0:x1]
            pairs=mask[:,1:] & mask[:,:-1]
            if mask.mean()<settings['minimum_coverage'] or pairs.sum()<32:
                continue
            contrast=float(np.abs(np.diff(colors,axis=1))[pairs].mean())
            areas.append((contrast,(x0,x1,y0,y1)))
    if not areas:
        return None
    contrast,(x0,x1,y0,y1)=max(areas)
    if contrast<settings['minimum_donor_contrast']:
        return None
    pixels=rgb[y0:y1,x0:x1].astype(np.float32)
    held0,held1=(cut,width) if x0==0 else (0,cut)
    mask=valid[y0:y1,x0:x1]
    peaks=[p for p in _repeat_peaks(pixels,mask) if p['agreement']>=settings['min_pairs_score']
           and (x1-x0)/p['lag']>=settings['minimum_cycles']]
    if not peaks:
        return None
    high=max(p['agreement'] for p in peaks)
    peak=min((p for p in peaks if p['agreement']>=.86*high),key=lambda p:p['lag'])
    lag=peak['lag']
    phase=_phase(pixels,mask,(x1-x0)/lag)
    if phase is not None:
        phase=float((phase+x0/lag)%1)
    far=rgb[y0:y1,held0:held1].mean(2).astype(np.float32)
    fv=valid[y0:y1,held0:held1]
    fp=fv[:,1:]&fv[:,:-1]
    far_contrast=float(abs(np.diff(far,axis=1))[fp].mean()) if fp.any() else None
    score=shift_score(far,fv,round(lag))
    # Last observed chart boundary opposite the donor, ignoring holes in each row.
    rows=fv.any(1)
    ends=None
    if rows.any():
        boundaries=(held1-np.argmax(fv[rows,::-1],axis=1) if x0==0 else held0+np.argmax(fv[rows],axis=1))
        ends=np.median(abs((boundaries/lag-phase+.5)%1-.5)) if phase is not None else None
    return dict(axis=axis,pitch_px=lag,chart_wh=list(sample['mask'].shape[::-1]),
                donor_box=box(axis,x0,x1,y0,y1),far_box=box(axis,held0,held1,y0,y1),
                donor_score=peak['agreement'],donor_contrast=contrast,phase=phase,
                far_score=score,far_contrast=far_contrast,far_coverage=float(fv.mean()),
                far_edge_residual=float(ends) if ends is not None else None)


def predict_surface(photo,part,sample,baseline,settings=SETTINGS):
    if sample is None or None not in baseline['counts']:
        return baseline
    aspect=sample.get('normal_aspect') or _aspect(transform(sample['quad']),sample['native_size'])
    if aspect is None or not np.isfinite(aspect) or aspect<=0:
        return baseline
    candidates=[p for axis in (0,1) if (p:=best_patch(sample,axis,settings)) is not None]
    if not candidates:
        return baseline
    patch=max(candidates,key=lambda p:p['donor_score'])
    i=patch['axis']
    n=patch['chart_wh'][i]/patch['pitch_px']
    guess=[n,n/aspect] if i==0 else [n*aspect,n]
    patch['raw_counts']=guess
    phase=[None,None]
    phase[i]=patch['phase']
    ok=False
    if max(abs(v-round(v)) for v in guess)>settings['integer_residual'] or min(guess)<.55:
        reason='noninteger extent'
    elif any(a is not None and abs(a-b)>settings['partial_tolerance'] for a,b in zip(baseline['counts'],guess)):
        reason='other measured axis disagrees'
    elif patch['far_coverage']<settings['minimum_coverage']:
        reason='insufficient far mask'
    elif patch['far_score']>=settings['held_repeat']:
        reason='repeat independently visible near far edge'
        ok=True
    elif (patch['far_edge_residual'] is not None and patch['far_edge_residual']<=settings['flat_edge_phase_error']
          and patch['far_contrast'] is not None and patch['far_contrast']<settings['flat_contrast_ratio']*patch['donor_contrast']):
        reason='donor seam phase reaches quiet far boundary'
        ok=True
    else:
        reason='far end does not confirm pitch or phase'
    patch.update(accepted=ok,reason=reason)
    return dict(method=METHOD,counts=guess if ok else baseline['counts'],phase=phase if ok else baseline['phase'],
                quad=sample['quad'],evidence=dict(patch=patch))
