"""Crop-only fallback: find a repeat vector that translates texture patches in two dimensions."""
import numpy as np

from engine.surface_pitch import _phase


METHOD = 'masked-patch-translations'
SETTINGS = dict(min_pitch=6, min_cycles=2.2, maximum_shift=4, max_slope=.13,
                minimum_pair_fraction=.20, bands=3, rows_per_band=12,
                minimum_band_correlation=.55, minimum_peak_prominence=.12,
                comparable_peak_fraction=.87, aspect_tolerance=.14,
                transfer_correlation=.75)


def axis_peaks(sample, axis):
    """Find repeated local shading/texel detail with an optional small cross-axis translation."""
    rgb, owned = sample['rgb'], sample['mask']
    if axis:
        rgb,owned=rgb.transpose(1,0,2),owned.T
    gray=rgb @ np.array([.299,.587,.114],np.float32)
    h,w=gray.shape
    limit=int(w/SETTINGS['min_cycles'])
    if limit<SETTINGS['min_pitch']+1 or h<6:
        return []
    valid=np.zeros_like(owned)
    valid[1:-1,1:-1]=owned[1:-1,1:-1]&owned[:-2,1:-1]&owned[2:,1:-1]&owned[1:-1,2:]&owned[1:-1,:-2]
    # Both derivatives describe the patch. Center each row so a constant mortar stripe cannot vote at every lag.
    feature=np.zeros_like(gray)
    feature[1:-1,1:-1]=gray[1:-1,2:]-gray[1:-1,:-2]+.55*(gray[2:,1:-1]-gray[:-2,1:-1])
    feature-=((valid*feature).sum(1)/np.maximum(1,valid.sum(1)))[:,None]
    feature=np.clip(feature,-32,32)*valid
    rows=np.flatnonzero((valid.sum(1)>=12)&((feature**2).sum(1)>4*valid.sum(1)))
    if len(rows)<6:
        return []
    bands=np.array_split(rows,SETTINGS['bands'])
    fft_size=1<<(2*w-1).bit_length()
    transforms=[np.fft.rfft(a,fft_size,axis=1) for a in (feature,feature**2,valid.astype(np.float32))]
    f,p,m=transforms
    lags=np.arange(limit+1)
    best=np.full(limit+1,-1.)
    shifts=np.zeros(limit+1,int)
    for dy in range(-SETTINGS['maximum_shift'],SETTINGS['maximum_shift']+1):
        votes=[]
        for group in bands:
            group=group[np.linspace(0,len(group)-1,min(SETTINGS['rows_per_band'],len(group))).round().astype(int)]
            group=group[(group+dy>=0)&(group+dy<h)]
            if len(group)<2:
                votes.append(np.full(limit+1,-1.))
                continue
            other=group+dy
            def correlation(a,b):
                return np.fft.irfft((a[other]*b[group].conj()).sum(0),fft_size)[:limit+1]
            n=correlation(m,m)
            power=correlation(p,m)*correlation(m,p)
            v=correlation(f,f)/np.sqrt(np.maximum(1.,power))
            v[n<max(22.,valid[group].sum()*SETTINGS['minimum_pair_fraction'])]=-1
            votes.append(v)
        score=np.median(votes,axis=0)
        better=(score>best)&(abs(dy)<=lags*SETTINGS['max_slope'])
        best[better]=score[better]
        shifts[better]=dy
    peaks=[]
    for i in range(SETTINGS['min_pitch'],limit):
        if best[i]<SETTINGS['minimum_band_correlation'] or best[i]<=best[i-1] or best[i]<best[i+1]:
            continue
        delta=max(2,round(i*.22))
        shoulder=max(np.min(best[max(1,i-delta):i]),np.min(best[i+1:min(limit+1,i+delta+1)]))
        if best[i]-shoulder < SETTINGS['minimum_peak_prominence']:
            continue
        pitch=i+.5*(best[i+1]-best[i-1])/max(1e-6,2*best[i]-best[i-1]-best[i+1])
        peaks.append(dict(pitch=float(pitch),count=float(w/pitch),correlation=float(best[i]),
                          cross_axis_shift=int(shifts[i]),prominence=float(best[i]-shoulder)))
    return peaks


def choose(peaks):
    if not peaks:
        return None
    best=max(p['correlation'] for p in peaks)
    candidates=[p for p in peaks if p['correlation']>=best*SETTINGS['comparable_peak_fraction']]
    # A true period can have several multiples. If the strong peaks are unrelated periods, abstain.
    first=min(candidates,key=lambda p:p['pitch'])
    if any(abs(p['pitch']/first['pitch']-round(p['pitch']/first['pitch']))>.14 for p in candidates):
        return None
    return first


def predict(sample, baseline):
    spans=list(baseline['counts'])
    phase=list(baseline['phase'])
    output=dict(method=METHOD,counts=spans,phase=phase,evidence={})
    if sample is None or all(n is not None for n in spans):
        output['evidence']['reason']='no sampling chart' if sample is None else 'preserve complete baseline count'
        return output
    alternatives=[axis_peaks(sample,axis) for axis in (0,1)]
    picks=[choose(options) for options in alternatives]
    aspect=baseline['evidence'].get('aspect')
    measured=[p['count'] if p else None for p in picks]
    evidence=dict(peaks=alternatives,selected=picks,aspect=aspect)
    output['evidence']=evidence
    for known, new in zip(spans,measured):
        if known is not None and new is not None and abs(np.log(known/new))>SETTINGS['aspect_tolerance']:
            evidence['reason']='patch period disagrees with previously measured axis'
            return output
    proposal=[old if old is not None else new for old,new in zip(spans,measured)]
    if aspect and all(n is not None for n in proposal):
        if abs(np.log(proposal[0]/proposal[1]/aspect))>SETTINGS['aspect_tolerance']:
            evidence['reason']='axes disagree with chart aspect'
            return output
    elif aspect and any(picks):
        axis=0 if picks[0] else 1
        if picks[axis]['correlation']>=SETTINGS['transfer_correlation'] and proposal[1-axis] is None:
            proposal[1-axis]=proposal[axis]/aspect if axis==0 else proposal[axis]*aspect
            evidence['transferred_axis']=1-axis
    for axis,new in enumerate(proposal):
        if spans[axis] is None and new is not None:
            spans[axis]=float(new)
            colors,mask=sample['rgb'],sample['mask']
            if axis:
                colors,mask=colors.transpose(1,0,2),mask.T
            phase[axis]=_phase(colors.astype(np.float32),mask,float(new))
    return output
