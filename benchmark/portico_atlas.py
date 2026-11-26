"""Test missing chart spans by fitting a periodically tiled vanilla face to visible pixels."""
from functools import lru_cache
from itertools import product

import numpy as np

from engine import atlas_count


SETTINGS = dict(texture_shortlist=20, axis_candidates=3, max_pairs=38, max_pixels=2400,
                minimum_samples=96, minimum_cell_coverage=64, min_period=5.,
                min_train_correlation=.65, min_check_correlation=.55, min_scale_margin=.10,
                relative_scale_separation=.10, old_axis_tolerance=.10)
METHOD = 'atlas-harmonic-check'


@lru_cache(maxsize=1)
def atlas():
    return atlas_count.load_atlas(version='26.3')


def _scales(sample, baseline):
    h,w = sample['mask'].shape
    lengths = (w,h)
    alternatives = baseline.get('evidence',{}).get('alternatives',[[],[]])
    aspect = baseline.get('evidence',{}).get('aspect') or sample.get('normal_aspect')
    axes = []
    for axis,options in enumerate(alternatives):
        candidates = sorted(options,key=lambda p:p.get('agreement',0)+p.get('native_support',0),reverse=True)
        seeds = [p['count'] for p in candidates[:SETTINGS['axis_candidates']]]
        if baseline['counts'][axis] is not None:
            seeds.insert(0,baseline['counts'][axis])
        axes.append(seeds)
    pairs = []
    def add(u,v):
        if min(u,v)<.55 or min(w/u,h/v)<SETTINGS['min_period']:
            return
        if not any(max(abs(np.log(u/old[0])),abs(np.log(v/old[1]))) < .025 for old in pairs):
            pairs.append((float(u),float(v)))
    # An observed spatial period can describe half a tile, one tile or a two-tile motif.
    seeds = list(product(axes[0],axes[1]))
    if aspect:
        seeds += [(u,u/aspect) for u in axes[0]] + [(v*aspect,v) for v in axes[1]]
    seeds += [(u,1.) for u in axes[0]]+[(1.,v) for v in axes[1]]
    for u,v in seeds:
        add(u,v)
    # Rectified windows often have a roughly integral number of cells. Include that
    # possibility without requiring the crop to start at a seam (phase is fitted).
    for u,v in seeds:
        add(round(u),round(v))
    for a,b in ((.5,.5),(2.,2.),(.5,1.),(1.,.5),(2.,1.),(1.,2.)):
        for u,v in seeds:
            add(u*a,v*b)
    return pairs[:SETTINGS['max_pairs']]


def _correlations(rgb,x,y,period,indices,templates):
    px,py=period
    field,norm,bins=atlas_count._fold(rgb,x,y,px,py)
    if norm<1. or np.count_nonzero(bins)<SETTINGS['minimum_cell_coverage']:
        return None
    spectra, squared = templates
    def cross(a,b):
        return np.fft.irfft2(a*b.conj(),s=(16,16),axes=(-2,-1))
    weights = np.fft.rfft2(bins)
    mean = cross(weights,spectra)
    var = cross(weights,squared)-mean*mean/len(rgb)
    return np.clip(cross(field,spectra)/(norm*np.sqrt(np.maximum(.001,var))),-1.,1.)


def predict(sample, baseline):
    result = dict(baseline, method=METHOD)
    evidence = dict(reason='ordinary counter already resolved both axes')
    result['evidence'] = {**baseline['evidence'],'atlas_check':evidence}
    if sample is None or None not in baseline['counts']:
        if sample is None:
            evidence['reason']='no usable crop'
        return result
    rgb,mask=sample['rgb'],sample['mask']
    y,x=np.nonzero(mask)
    if len(y)<SETTINGS['minimum_samples']*2:
        evidence['reason']='too few observed pixels'
        return result
    h,w=mask.shape
    scales=_scales(sample,baseline)
    if not scales:
        evidence['reason']='no repeat pitch hypothesis'
        return result
    source=atlas()
    ids=atlas_count._shortlist(rgb,mask,source)[0][:SETTINGS['texture_shortlist']]
    spectra=source.spectra[ids]
    templates=(spectra,np.fft.rfft2(source.patterns[ids]**2))
    keep=np.linspace(0,len(x)-1,min(len(x),SETTINGS['max_pixels'])).astype(int)
    y,x=y[keep],x[keep]
    gray=rgb[y,x].mean(axis=1)
    # Verify the same candidate orientation and phase on disjoint spatial patches.
    training=((4*x//w+4*y//h)%2)==0
    if min(training.sum(),(~training).sum())<SETTINGS['minimum_samples']:
        evidence['reason']='insufficient separate patch support'
        return result
    hypotheses=[]
    for u,v in scales:
        corr=_correlations(gray[training],x[training],y[training],(w/u,h/v),ids,templates)
        if corr is not None:
            loc=np.unravel_index(corr.argmax(),corr.shape)
            hypotheses.append((float(corr[loc]),(u,v),tuple(map(int,loc))))
    if not hypotheses:
        evidence['reason']='too few distinct observed texels'
        return result
    hypotheses.sort(reverse=True)
    score,span,loc=hypotheses[0]
    index,turn,dy,dx=loc
    other=max((q for q,uv,_ in hypotheses[1:] if max(abs(np.log(np.divide(uv,span))))>SETTINGS['relative_scale_separation']), default=None)
    # Explicitly challenge the winner at half and double its pitch even when a harmonic
    # was outside the initial proposal budget; any atlas texture may explain that scale.
    for a,b in ((.5,1.),(2.,1.),(1.,.5),(1.,2.),(.5,.5),(2.,2.)):
        periods=w/(span[0]*a),h/(span[1]*b)
        if min(periods)<SETTINGS['min_period']:
            continue
        corr=_correlations(gray[training],x[training],y[training],periods,ids,templates)
        if corr is not None:
            other=max(float(corr.max()),other if other is not None else -1.)
    margin=score-other if other is not None else 0.
    held=_correlations(gray[~training],x[~training],y[~training],(w/span[0],h/span[1]),ids[index:index+1],
                       tuple(arr[index:index+1] for arr in templates))
    check=float(held[0,turn,dy,dx]) if held is not None else -1.
    old_agrees=all(n is None or abs(np.log(n/p))<SETTINGS['old_axis_tolerance'] for n,p in zip(baseline['counts'],span))
    accepted=bool(score>=SETTINGS['min_train_correlation'] and check>=SETTINGS['min_check_correlation']
                  and margin>=SETTINGS['min_scale_margin'] and old_agrees)
    evidence.update(reason='validated whole-tile spacing' if accepted else 'ambiguous whole-tile spacing',accepted=accepted,
                    atlas_texture=source.labels[ids[index]],span=list(span),train=score,check=check,margin=margin,
                    best_distinct_scale_correlation=other,orientation=turn,phase=[dx/16,dy/16],candidate_grids=len(scales))
    if accepted:
        result['counts']=[p if n is None else n for p,n in zip(span,baseline['counts'])]
        result['phase']=[p if n is None else old for n,p,old in zip(baseline['counts'],evidence['phase'],baseline['phase'])]
    return result
