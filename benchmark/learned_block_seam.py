"""Use a trained seam likelihood on a rectified surface to measure its grid."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


MODEL = Path(__file__).with_name('data') / 'learned-block-seam-v1.json'
RADII = (1, 2, 4)
FEATURES = [f'{name}_{r}px' for r in RADII for name in
            ('color_step', 'coherent_step', 'dark_joint', 'bright_joint', 'strong_step_fraction')]


def profiles(rgb, mask):
    """Measure line evidence across supported scanlines. Row x is at pixel center x+.5."""
    h, w = mask.shape
    rows = np.flatnonzero(mask.sum(axis=1) >= 7)
    if w < 12 or len(rows) < 4:
        return None
    rows = rows[np.linspace(0, len(rows)-1, min(48,len(rows))).round().astype(int)]
    colors = rgb[rows].astype(np.float32)
    valid = mask[rows]
    luminance = colors @ np.array([.299,.587,.114], np.float32)
    columns = np.arange(w)
    result = []
    for r in RADII:
        low, high = np.maximum(0, columns-r), np.minimum(w-1, columns+r)
        supported = valid & valid[:,low] & valid[:,high] & (columns>=r) & (columns<w-r)
        count = supported.sum(axis=0)
        weight = supported / np.maximum(1,count)
        diff = colors[:,high] - colors[:,low]
        magnitude = np.abs(diff).mean(axis=2)
        dark = np.maximum(0,np.minimum(luminance[:,high],luminance[:,low])-luminance)
        bright = np.maximum(0,luminance-np.maximum(luminance[:,high],luminance[:,low]))
        result.extend([np.log1p((magnitude*weight).sum(axis=0)),
                       np.log1p(np.abs((diff*weight[:,:,None]).sum(axis=0)).mean(axis=1)),
                       np.log1p((dark*weight).sum(axis=0)), np.log1p((bright*weight).sum(axis=0)),
                       ((magnitude>9)*weight).sum(axis=0)])
    return np.stack(result,axis=1), count>=max(4,.12*len(rows))


@lru_cache(maxsize=1)
def model():
    return json.loads(MODEL.read_text(encoding='utf8'))


def probabilities(profile, weights):
    features, supported = profile
    logits = (features-weights['mean'])/weights['scale'] @ weights['coef'] + weights['intercept']
    return 1/(1+np.exp(-np.clip(logits,-30,30))), supported


def best_phase(prob, valid, count):
    """Compare repeated seam likelihood at candidate grid boundaries and between them."""
    n = len(prob)
    pitch = n/count
    if pitch < 6 or count < 1.35 or valid.sum() < 12:
        return None
    phases = np.arange(32)/32.
    origins = (np.arange(int(np.ceil(count))+1)[None,:]+phases[:,None])*pitch-.5
    # Averages over offsets below half a native pixel reduce sampling jitter.
    x = np.arange(n)
    near = np.interp(origins,x,prob)
    support = np.interp(origins,x,valid.astype(float),left=0.,right=0.) > .95
    votes = support.sum(axis=1)
    response = (near*support).sum(axis=1)/np.maximum(1,votes)
    enough = votes >= max(2,int(np.ceil(.42*count)))
    if not np.any(enough):
        return None
    best = int(np.argmax(np.where(enough,response,-1.)))
    distant = (abs((phases-phases[best]+.5)%1.-.5)>.16) & enough
    if not np.any(distant):
        return None
    runner = float(response[distant].max())
    mean = float(np.mean(prob[valid]))
    quality = float(response[best] - .65*runner - .35*mean)
    return quality, float(phases[best])


def pitch_guesses(prob, valid):
    """Search peaks in a zero-padded frequency spectrum of the seam probabilities."""
    n = len(prob)
    if n < 20 or valid.sum()<12:
        return []
    signal = (prob-prob[valid].mean())*valid
    amplitude = np.abs(np.fft.rfft(signal,n=n*8))
    peaks = np.flatnonzero((amplitude[1:-1]>amplitude[:-2]) & (amplitude[1:-1]>=amplitude[2:]))+1
    peaks = peaks[(peaks>=16)&(peaks<=8*n/6)]
    strong = sorted(peaks,key=lambda k:-amplitude[k])[:6]
    return sorted(set(float(k+offset)/8 for k in strong for offset in (-1,0,1)))


def best_new(prob, valid, expected):
    guesses = pitch_guesses(prob,valid)
    if expected is not None:
        guesses = [g for g in guesses if abs(np.log(g/expected))<.13]
        guesses.append(expected)
    options = []
    for n in guesses:
        peak = best_phase(prob,valid,n)
        if peak:
            quality, phase = peak
            options.append((quality,n,phase))
    return max(options,default=None)


def predict(sample, baseline):
    counts, phases = list(baseline['counts']), list(baseline['phase'])
    result = dict(counts=counts, phase=phases)
    if sample is None or all(v is not None for v in phases):
        return result
    weights = model()
    aspect = (baseline.get('evidence') or {}).get('aspect') or sample.get('normal_aspect')
    new = {}
    for axis in (0,1):
        if phases[axis] is not None:
            continue
        rgb, mask = sample['rgb'],sample['mask']
        if axis:
            rgb, mask = rgb.transpose(1,0,2),mask.T
        profile = profiles(rgb,mask)
        if profile is None:
            continue
        prob, valid = probabilities(profile,weights)
        if counts[axis] is not None:
            peak = best_phase(prob,valid,counts[axis])
            if peak and peak[0] >= weights['phase_threshold']:
                phases[axis] = peak[1]
        else:
            other = baseline['counts'][1-axis]
            expected = None if not aspect or other is None else other*(aspect if axis==0 else 1/aspect)
            proposal = best_new(prob,valid,expected)
            if proposal and proposal[0] >= weights['pitch_threshold']:
                new[axis] = proposal
    if all(n is None for n in counts):
        if len(new)!=2 or not aspect or abs(np.log(new[0][1]/new[1][1]/aspect))>.13:
            return result
    for axis,(_,n,phase) in new.items():
        counts[axis], phases[axis] = n, phase
    return result
