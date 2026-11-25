"""Measure texture repeats along each axis of a rectified visible surface."""
import numpy as np

from engine.surface_pitch import _aspect, _axis as native_texel_axis, _phase
from engine.geometry import transform


METHOD = 'masked-repeat'
SETTINGS = dict(scanlines=48, minimum_pitch_px=5., unverified_pitch_px=10., minimum_cycles=1.8, minimum_row_pairs=8,
                minimum_texture_contrast=1.6, minimum_correlation=.30,
                shorter_peak_ratio=.86, native_alignment=.13, aspect_log_tolerance=.16,
                independent_agreement=.50, aspect_transfer_agreement=.62)


def _repeat_peaks(colors, valid):
    """Cross-correlate pairs of *observed* pixels on independent parallel scanlines."""
    gray = colors @ np.array([.299,.587,.114],np.float32)
    # A small central difference removes smooth lighting changes. Mixing it
    # with color intensity retains a broad mortar joint as well as fine detail.
    n = gray.shape[1]
    if n < 12:
        return []
    weights = valid[:,1:-1] & valid[:,:-2] & valid[:,2:]
    available = weights.sum(1)
    centered = gray[:,1:-1] - ((gray[:,1:-1]*weights).sum(1)/np.maximum(1,available))[:,None]
    derivative = (gray[:,2:] - gray[:,:-2])*.5
    contrast = np.sqrt((centered**2*weights).sum(1)/np.maximum(1,available))
    keep = (available >= 14) & (contrast >= SETTINGS['minimum_texture_contrast'])
    if keep.sum() < 3:
        return []
    valid = weights[keep].astype(np.float32)
    nfft = 1 << (2*n-1).bit_length()
    maxlag = int(n/SETTINGS['minimum_cycles'])
    fft = lambda a: np.fft.rfft(a, nfft, axis=1)
    def corr(a,b):
        return np.fft.irfft(a*b.conj(), nfft, axis=1)[:,:maxlag+1]
    m = fft(valid)
    pairs = np.maximum(0., np.rint(corr(m,m)))
    enough = pairs >= np.maximum(SETTINGS['minimum_row_pairs'], available[keep,None]*.18)
    scores = np.zeros_like(pairs)
    # Staggered bricks and planks often repeat the same dark stroke twice or
    # four times per block. Subtract the part all rows share before correlating
    # the remaining individual texel pattern. This also removes broad shadows.
    derivative=derivative[keep]
    residual=derivative-(derivative*valid).sum(0)/np.maximum(1,valid.sum(0))
    for feature, significance in ((centered[keep],.24),(derivative,.16),(residual,.60)):
        weighted = feature*valid
        f = fft(weighted)
        f2 = fft(weighted**2)
        den = np.sqrt(np.maximum(0.,corr(f2,m)*corr(m,f2)))
        scores += significance * np.clip(corr(f,f)/np.maximum(den,.01),-1,1)
    # A mean over clipped per-row agreements rewards a consensus and gives
    # each measured row one vote, regardless of the darkness of its texture.
    lines = enough.sum(0)
    curve = (scores*enough).sum(0)/np.maximum(1,lines)
    curve[lines < max(3,.25*len(valid))] = 0
    peaks = np.flatnonzero((curve[1:-1] >= curve[:-2]) & (curve[1:-1] > curve[2:])) + 1
    options=[]
    for i in peaks:
        if i < SETTINGS['minimum_pitch_px'] or curve[i] < SETTINGS['minimum_correlation']:
            continue
        if i < SETTINGS['unverified_pitch_px'] and curve[i] < .68:
            continue
        neighbors=curve[i-1:i+2]
        curvature=2*neighbors[1]-neighbors[0]-neighbors[2]
        lag=float(i+(neighbors[2]-neighbors[0]) / max(1.e-6,2*curvature))
        options.append(dict(count=float(n/lag), lag=lag, agreement=float(curve[i]),
                            rows=int(lines[i]), source='repeated texture'))
    return options


def _options(sample, axis):
    rgb, mask = sample['rgb'],sample['mask']
    if axis:
        rgb,mask=rgb.transpose(1,0,2),mask.T
    rows = np.flatnonzero(mask.sum(1)>=12)
    if len(rows)>SETTINGS['scanlines']:
        rows=rows[np.linspace(0,len(rows)-1,SETTINGS['scanlines']).round().astype(int)]
    if len(rows)<3:
        return [],None,None
    colors,valid=rgb[rows].astype(np.float32),mask[rows]
    repeats=_repeat_peaks(colors,valid)
    # Sixteen resolved texel steps distinguish a full tile from a half-block
    # decorative motif. Keep this extra check confined to the observed pixels.
    native=[]
    if len(rows)>=8 and colors.shape[1]>=32:
        texture=native_texel_axis(colors,valid,float(colors.shape[1]))
        native=texture.get('candidates',[])
    for p in repeats:
        support=[q['strength'] for q in native if abs(np.log(q['count']/p['count']))<.08]
        p['native_support']=max(support,default=0.)
    if not repeats:
        for q in native:
            if q['strength']>=SETTINGS['native_alignment'] and q['effective_bins']>=10:
                repeats.append(dict(count=q['count'],lag=colors.shape[1]/q['count'],agreement=0.,
                                    rows=len(rows),native_support=q['strength'],source='16-texel spacing'))
    return repeats,colors,valid


def _best(peaks):
    if not peaks:
        return None
    supported=[p for p in peaks if p['native_support']>=SETTINGS['native_alignment']]
    if supported:
        return max(supported,key=lambda p:p['native_support']+.25*p['agreement'])
    best=max(p['agreement'] for p in peaks)
    # An exact tile also matches at two and three blocks. Prefer the first
    # strong peak that explains the pattern, including fractional sample lags.
    comparable=[p for p in peaks if p['agreement']>=best*SETTINGS['shorter_peak_ratio']]
    return min(comparable,key=lambda p:p['lag'])


def _select(options,aspect):
    selected=[_best(peaks) for peaks in options]
    if aspect and all(selected):
        # When the two strongest periods disagree with a measured rectangle's
        # shape, use aspect only to disambiguate convincing competing harmonics.
        if abs(np.log(selected[0]['count']/selected[1]['count']/aspect))<=SETTINGS['aspect_log_tolerance']:
            return selected
        pairs=[(a,b) for a in options[0] for b in options[1]
               if abs(np.log(a['count']/b['count']/aspect))<.10
               and a['agreement']>=.74*selected[0]['agreement'] and b['agreement']>=.74*selected[1]['agreement']]
        if pairs:
            return max(pairs,key=lambda pair:sum(p['agreement']+p['native_support'] for p in pair))
    selected=[p if p and (p['native_support']>=SETTINGS['native_alignment'] or
                           p['agreement']>=SETTINGS['independent_agreement']) else None for p in selected]
    if aspect and all(selected):
        # Inconsistent measured scales with no agreeing alternative often
        # arise from the narrow parallel boards within a single wood block.
        quality=[p['agreement']+p['native_support'] for p in selected]
        if max(quality)-min(quality)<.25:
            return [None,None]
        selected[int(np.argmin(quality))]=None
    if aspect and sum(p is not None for p in selected)==1:
        index=0 if selected[0] else 1
        p=selected[index]
        if p['native_support']>.20 or p['agreement']>SETTINGS['aspect_transfer_agreement']:
            amount=p['count'] / aspect if index==0 else p['count']*aspect
            selected[1-index]=dict(count=amount,source='measured axis and plane aspect')
    return selected


def count(sample, version='26.3'):
    """Measured edge spans in chart UV order. Each axis can independently abstain."""
    if version!='26.3':
        raise ValueError('Masked texture counting uses default Java 26.3 full-block textures.')
    if sample is None:
        return dict(method=METHOD,counts=[None,None],phase=[None,None],evidence={'reason':'no usable chart'})
    axes=[_options(sample,i) for i in (0,1)]
    normal_aspect=sample.get('normal_aspect')
    aspect=normal_aspect or _aspect(transform(sample['quad']),sample['native_size'])
    selected=_select([p for p,_,_ in axes],aspect)
    counts=[p['count'] if p else None for p in selected]
    phases=[_phase(colors,valid,p['count']) if p and colors is not None else None
            for p,(_,colors,valid) in zip(selected,axes)]
    return dict(method=METHOD,counts=counts,phase=phases,
                evidence=dict(axes=list(selected),alternatives=[p for p,_,_ in axes],aspect=aspect,
                              aspect_source=('camera-space normal and assumed lens' if normal_aspect else
                                             'chart vanishing directions' if aspect else None)))
