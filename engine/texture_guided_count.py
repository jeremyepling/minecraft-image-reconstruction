"""Straighten a single supported surface using texture repeats and a reserved validation strip."""
import numpy as np

from engine.chart_envelope import SETTINGS as ENVELOPE_SETTINGS, fit_quad
from engine.geometry import UNIT, project, transform
from engine.masked_count import SETTINGS as MASKED_SETTINGS, count
from engine.surface_sample import sample_surface


SETTINGS = dict(envelope_min_iou=.72, envelope_weights=[.5, 1.], slope_changes=[-.075, .075],
                training_outer_fraction=.30, held_center_fraction=.30, minimum_pairs=32,
                minimum_repeat=.55, max_pitch_drift=.12)
METHOD = 'texture-guided-repeat'


def chart_options(photo, part, sample, axis):
    corners = np.asarray(sample['quad'])
    yield 'original', sample
    fitted, quality = fit_quad(photo, part, corners)
    seeds = [('original', corners)]
    if fitted is not None and quality >= SETTINGS['envelope_min_iou']:
        for weight in SETTINGS['envelope_weights']:
            quad = corners + weight*(np.asarray(fitted)-corners)
            seeds.append((f'envelope-{weight}', quad))
            yield seeds[-1][0], sample_surface(photo, dict(part, quad=quad))
    for name, seed in (seeds[0], seeds[-1]) if len(seeds)>1 else seeds:
        h = transform(seed)
        for slope in SETTINGS['slope_changes']:
            for shear in ('shear', 'taper'):
                uv=UNIT.copy()
                movement=slope*(2*UNIT[:,axis]-1)
                if shear=='taper':
                    movement*=2*UNIT[:,1-axis]-1
                uv[:,1-axis]+=movement
                yield f'{name}-{shear}-{slope}', sample_surface(photo, dict(part, quad=project(h,uv)))


def fixed_bands(sample, original_inverse, axis):
    # Hold the same source strip out for every candidate chart.
    height,width=sample['mask'].shape
    uv=np.stack(np.meshgrid((np.arange(width)+.5)/width, (np.arange(height)+.5)/height),axis=-1)
    coord=project(original_inverse@transform(sample['quad']),uv)[...,1-axis]
    outer=SETTINGS['training_outer_fraction']
    center=SETTINGS['held_center_fraction']/2
    return [sample['mask']&m for m in (coord<outer, coord>1-outer, abs(coord-.5)<center)]


def shift_score(gray, mask, lag):
    if lag<=0 or lag>=gray.shape[1]-1:
        return -1.
    # Compare local changes as well as brightness to suppress repeated large shadow gradients.
    valid=mask[:,1:]&mask[:,:-1]
    both=valid[:,lag:]&valid[:,:-lag]
    if both.sum()<SETTINGS['minimum_pairs']:
        return -1.
    d=np.diff(gray,axis=1)
    def corr(a):
        x,y=a[:,lag:][both],a[:,:-lag][both]
        x,y=x-x.mean(),y-y.mean()
        return float(np.mean(x*y)/max(1.,np.sqrt(np.mean(x*x)*np.mean(y*y))))
    return .65*corr(d)+.35*corr(gray[:,1:])


def train_chart(sample,bands,axis):
    fit=count(dict(sample,mask=bands[0]|bands[1]))
    if None in fit['counts']:
        return None
    axis_info=fit['evidence']['axes'][axis]
    if axis_info['source']!='repeated texture':
        return None
    lag=axis_info['lag']
    gray=sample['rgb'].mean(-1).astype(np.float32)
    if axis:
        gray=gray.T
    peaks=[]
    exact=[]
    delta=max(2,round(.10*lag))
    for valid in bands[:2]:
        mask=valid.T if axis else valid
        lags=range(max(3,round(lag)-delta),round(lag)+delta+1)
        peaks.append(max((shift_score(gray,mask,k),k) for k in lags)[1])
        exact.append(shift_score(gray,mask,round(lag)))
    drift=abs(peaks[0]-peaks[1])/lag
    if drift>SETTINGS['max_pitch_drift'] or min(exact)<SETTINGS['minimum_repeat']:
        return None
    return min(exact)-drift, fit, lag, gray, dict(train_scores=exact,band_pitch=peaks,drift=drift)


def count_surface(photo, part, sample, baseline=None, version='26.3'):
    """Return the reading and the corresponding full supported chart for later stages."""
    if baseline is None:
        baseline=count(sample,version=version)
    if sample is None or None not in baseline['counts']:
        return baseline,sample
    axis=int(sample['mask'].shape[0]>sample['mask'].shape[1])
    inverse=np.linalg.inv(transform(sample['quad']))
    best=None
    for name,chart in chart_options(photo,part,sample,axis):
        if chart is None:
            continue
        bands=fixed_bands(chart,inverse,axis)
        if min(int(b.sum()) for b in bands)<SETTINGS['minimum_pairs']:
            continue
        candidate=train_chart(chart,bands,axis)
        if candidate is not None and (best is None or candidate[0]>best[0]):
            best=(*candidate,name,chart,bands)
    if best is None:
        return baseline,sample
    score,fit,lag,gray,record,name,chart,bands=best
    held=shift_score(gray,bands[2].T if axis else bands[2],round(lag))
    check=dict(**record,chart=name,selected_score=score,held_repeat=held)
    if held<SETTINGS['minimum_repeat']:
        return dict(baseline,evidence=dict(baseline['evidence'],texture_check=check)),sample
    return dict(fit,method=METHOD,evidence=dict(fit['evidence'],texture_check=check)),chart
