"""Measure a planar repeat against vanilla face textures and rank rectified block faces."""
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine.atlas import digest, discover, make_atlas
from engine.geometry import transform
from engine.reconstruction import opaque_sprites
from engine.surface_pitch import _aspect


SETTINGS = dict(color_shortlist=32, period_shortlist=12, analysis_side=224, tile_size=16,
                minimum_repeat_correlation=.48, maximum_axis_error=.34,
                min_brightness=.30, max_brightness=1.15, minimum_mask_pixels=32,
                minimum_axis_repeat=.16, maximum_phase_ambiguity_bins=2, phase_tie_correlation=.025)
METHOD = 'atlas-repeat'
ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_INDEX = ROOT/'benchmark/data/count-atlas-v1/placement-index-26.3.json.gz'
INDEX_SHA256 = '32f95873f5b10e0fa36023c149b4277a4b1f7a59fb77940ede120f9292f37ac6'
# Ordered texture IDs and pixel rectangles in the verified vanilla 26.3 sheet.
SPRITE_LAYOUT_SHA256 = 'e9168881e77cb0cd1287fdd7b52d15bf92e1b09c66271b070c4202384d95ad3c'


@dataclass
class Atlas:
    tiles: np.ndarray
    labels: list
    ids: list
    means: np.ndarray
    spreads: np.ndarray
    patterns: np.ndarray
    spectra: np.ndarray
    profiles: np.ndarray
    provenance: dict
    material_index: dict | None = None


def _prepare(tiles, labels, ids, provenance):
    linear = tiles.astype(np.float32)
    means = linear.mean((1, 2))
    # Within-tile contrast helps distinguish smooth and patterned materials of similar color.
    spreads = linear.std((1, 2))
    gray = linear.mean(-1)
    patterns = np.stack([np.rot90(g, k, axes=(1, 2)) for g in (gray, gray[:, :, ::-1])
                         for k in range(4)], axis=1)
    patterns -= patterns.mean((2, 3), keepdims=True)
    patterns /= np.maximum(1., patterns.std((2, 3), keepdims=True))
    spectra = np.fft.rfft2(patterns)
    profiles = []
    for axis in (2, 1):
        centered = gray-gray.mean(axis,keepdims=True)
        ac = np.stack([(centered*np.roll(centered,i,axis)).mean((1,2)) for i in range(16)],axis=1)
        profiles.append(ac / np.maximum(1e-6,ac[:,:1]))
    return Atlas(tiles, list(labels), ids, means, spreads, patterns, spectra,
                 np.stack(profiles,axis=1), provenance)


def prepared_atlas_paths(version):
    workspace_atlas = Path('workspace/builds/mcir-port-validation') / ('final-cli-' + version) / 'atlas'
    paths = [ROOT/'runs'/version/'atlas', ROOT/'runs'/'atlas',
             ROOT.parent/workspace_atlas, ROOT.parent.parent/workspace_atlas]
    return [p for p in paths if (p/'atlas.json').is_file()]


def _verify_material_index(local, calibrated):
    # Offline regeneration can omit stateful cubes. Every ID it does include
    # must be a calibrated choice for this texture and facing.
    if (any(local[k] != calibrated[k] for k in ('version', 'client_sha256', 'atlas_sha256'))
            or local['choices'].keys() != calibrated['choices'].keys()):
        raise ValueError('Material placement index does not match the calibrated atlas')
    for side, choices in local['choices'].items():
        for texture, ids in choices.items():
            if not ids or not set(ids).issubset(calibrated['choices'][side].get(texture, ())):
                raise ValueError(f'Unexpected block mapping for {texture} on the {side} face')
    if not any(local['choices'].values()):
        raise ValueError('Material placement index contains no block mappings')


def load_atlas(path=None, version='26.3', *, verify_material_index=False):
    """Load local vanilla art and the frozen full-block index for the measured game release."""
    if path is None:
        path = next(iter(prepared_atlas_paths(version)), None)
        if path is None and version=='26.3':
            path=ROOT/'runs'/version/'atlas'
            make_atlas(discover(version),path,version)
    if path is None or not (Path(path)/'atlas.json').is_file():
        raise FileNotFoundError(f'Prepare an atlas with python reconstruct.py atlas --version {version} --out runs/{version}/atlas')
    path = Path(path)
    # A catalog rebuilt without game-state defaults omits stateful full cubes. Keep
    # the recorded calibration index fixed while allowing local regeneration of art.
    if version=='26.3':
        index_bytes=gzip.decompress(CALIBRATED_INDEX.read_bytes())
        if hashlib.sha256(index_bytes).hexdigest()!=INDEX_SHA256:
            raise ValueError('The frozen 26.3 block placement index changed')
    else:
        index_bytes=(path/'placement-index.json').read_bytes()
    index = json.loads(index_bytes)
    meta = json.loads((path/'atlas.json').read_bytes())
    if (meta['version'] != version or index['version'] != version
            or index['atlas_sha256'] != meta['atlas_sha256']
            or index['client_sha256'] != meta['recipe']['client_sha256']
            or digest(path/meta['atlas']) != meta['atlas_sha256']):
        raise ValueError('Atlas art, target version and placement index must agree')
    local_index = None
    if version == '26.3':
        layout = [(t['id'], t['rect']) for t in meta['textures']]
        if hashlib.sha256(json.dumps(layout, separators=(',', ':')).encode()).hexdigest() != SPRITE_LAYOUT_SHA256:
            raise ValueError('Vanilla sprite names or coordinates do not match the calibrated atlas')
        if verify_material_index:
            local_index = json.loads((path/'placement-index.json').read_bytes())
            _verify_material_index(local_index, index)
    tiles, labels = opaque_sprites(path,index)
    ids = []
    for texture in labels:
        blocks = set().union(*(v.get(texture,[]) for v in index['choices'].values()))
        ids.append(['minecraft:'+b for b in sorted(blocks,key=lambda b:(b!=texture,b))])
    provenance = dict(version=version, atlas_sha256=meta['atlas_sha256'],
                      client_sha256=index['client_sha256'], placement_index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                      atlas_json_sha256=digest(path/'atlas.json'), texture_count=len(labels),
                      block_count=len(set().union(*map(set,ids))),
                      scope='all opaque untinted full-cube face sprites in the placement index, all six directions')
    atlas = _prepare(tiles, labels.tolist(), ids, provenance)
    atlas.material_index = local_index
    return atlas


def _shortlist(rgb, mask, atlas):
    pixels = rgb[mask]
    mean = np.median(pixels,axis=0)
    spread = .5*(np.percentile(pixels,84,axis=0)-np.percentile(pixels,16,axis=0))
    gain = np.clip((atlas.means*mean).sum(1)/np.maximum(1,(atlas.means**2).sum(1)),
                   SETTINGS['min_brightness'],SETTINGS['max_brightness'])
    color = np.mean(abs(atlas.means*gain[:,None]-mean),axis=1)
    contrast = np.mean(abs(np.log((spread+2)/(atlas.spreads*gain[:,None]+2))),axis=1)
    loss = color + 3*contrast + 3*abs(np.log(gain/.78))
    return np.argsort(loss,kind='stable')[:SETTINGS['color_shortlist']],loss


def _resample(rgb,mask,size):
    """Premultiply before resampling, so unobserved pixels cannot affect the match."""
    method = Image.Resampling.BOX if min(rgb.shape[:2])>=max(size) else Image.Resampling.BILINEAR
    m = np.asarray(Image.fromarray(mask.astype(np.float32)).resize(size,method))
    arr = np.stack([np.asarray(Image.fromarray(np.float32(rgb[...,k])*mask).resize(size,method))
                    for k in range(3)],axis=-1)
    return arr/np.maximum(m[...,None],1e-5), m>.90


def _labels(atlas,indices):
    ordered = list(dict.fromkeys(int(i) for i in indices))
    ids = list(dict.fromkeys(atlas.ids[i][0] for i in ordered))
    ids.extend(b for i in ordered for b in atlas.ids[i] if b not in ids)
    return dict(block_id=ids[0],ranked_ids=ids[:5],texture=atlas.labels[ordered[0]],
                same_texture_ids=atlas.ids[ordered[0]])


def _unknown():
    return dict(block_id=None,ranked_ids=[],texture=None,same_texture_ids=[])


def rank_tile(sample,atlas,version='26.3'):
    """Rank IDs for a single block with known corners, allowing face orientation and shade."""
    if atlas.provenance['version'] != version:
        raise ValueError('Material atlas is for a different game version')
    rgb = np.asarray(sample['rgb'])
    valid = np.asarray(sample.get('mask',np.ones(rgb.shape[:2],bool)),bool)
    if np.count_nonzero(valid)<4:
        return dict(**_unknown(),evidence=dict(reason='insufficient visible tile pixels'))
    observed,mask = _resample(rgb,valid,(16,16))
    if mask.sum()<16:
        return dict(**_unknown(),evidence=dict(reason='insufficient covered texels'))
    indices,prior = _shortlist(observed,mask,atlas)
    gray = observed.mean(2)[mask]
    gray -= gray.mean()
    templates = atlas.patterns[indices][:,:,mask]
    templates = templates-templates.mean(-1,keepdims=True)
    correlation = (templates*gray).sum(-1)/np.maximum(1e-5,np.linalg.norm(templates,axis=-1)*np.linalg.norm(gray))
    scores = correlation.max(1)-prior[indices]/45
    order = np.argsort(-scores,kind='stable')
    return dict(**_labels(atlas,indices[order]),evidence=dict(correlation=float(correlation[order[0]].max()),
                visible_texels=int(mask.sum()),material_score=float(scores[order[0]])))


def _observed_profile(gray,mask,axis):
    if axis==1:
        gray,mask=gray.T,mask.T
    rows=np.linspace(0,len(gray)-1,min(48,len(gray))).astype(int)
    g,m=gray[rows],mask[rows]
    window=np.maximum(1,m.sum(1,keepdims=True))
    signal=(g-(g*m).sum(1,keepdims=True)/window)*m
    size=2*g.shape[1]
    f=np.fft.rfft(signal,n=size)
    covariance=np.fft.irfft(f*np.conj(f),n=size).sum(0)
    fm=np.fft.rfft(m,n=size)
    power=np.fft.rfft(signal**2,n=size)
    norm=np.fft.irfft(power*fm.conj(),n=size).sum(0)
    overlap=np.fft.irfft(fm*fm.conj(),n=size).sum(0)
    lags=np.arange(1,min(85,max(1,int(g.shape[1]*.70))))
    lags=lags[overlap[lags]>=max(16,.14*overlap[0])]
    return lags,np.clip(covariance[lags]/np.maximum(1e-6,np.sqrt(np.maximum(0,norm[lags]*norm[-lags]))),-1,1)


def _periods(length,profile,templates):
    lag,obs = profile
    if len(lag)<4:
        return None
    # Compare measured autocorrelation with each periodically tiled atlas face.
    periods=np.geomspace(3.2,min(180,1.4*length),100)
    extrema=np.flatnonzero((obs[1:-1]>=obs[:-2]) & (obs[1:-1]>obs[2:]) & (obs[1:-1]>.18))+1
    maxima=[]
    for k in extrema:
        offset=.5*(obs[k-1]-obs[k+1])/min(-1e-6,obs[k-1]-2*obs[k]+obs[k+1])
        p=lag[k]+np.clip(offset,-.5,.5)
        maxima.extend([p,2*p,p/2])
    periods=np.unique(np.r_[periods,maxima])
    periods=periods[(periods>=3.)&(periods<=1.4*length)]
    coords=lag[None,:]*16/periods[:,None]
    base=np.floor(coords).astype(int)
    frac=coords-base
    curves=templates[:,base%16]*(1-frac)+templates[:,(base+1)%16]*frac
    # Smooth attenuation from minification and stochastic texture variants can weaken repetition.
    weights=1/np.sqrt(1+lag/10.)
    residual=curves-obs
    errors=np.average(np.minimum(abs(residual),.8),axis=2,weights=weights)
    return periods,errors


def _fold(gray,x,y,px,py):
    u=np.floor((x+.5)*16/px).astype(int)%16
    v=np.floor((y+.5)*16/py).astype(int)%16
    n=np.bincount(v*16+u,minlength=256).reshape(16,16)
    sums=np.bincount(v*16+u,weights=gray,minlength=256).reshape(16,16)
    # Fit every observed pixel. Collapsing inconsistent samples to a single mean per
    # phase bin can make an arbitrary period appear to match a faint atlas outline.
    center=gray.mean()
    return np.fft.rfft2(sums-n*center), np.linalg.norm(gray-center),n


def _texture_alignment(samples,cols,rows,px,py,atlas,index,rotations):
    field,norm,weights = _fold(samples,cols,rows,px,py)
    if norm<2 or np.count_nonzero(weights)<32:
        return None
    spectra = atlas.spectra[index,rotations].conj()
    def correlate(source,template):
        return np.fft.irfft2(source[None]*template,s=(16,16),axes=(-2,-1))
    cross = correlate(field,spectra)
    # Remove the mean for each candidate alignment using just observed pixels.
    visible = np.fft.rfft2(weights)
    squared = np.fft.rfft2(atlas.patterns[index,rotations]**2)
    mean = correlate(visible,spectra)
    variance = correlate(visible,squared.conj()) - mean**2/weights.sum()
    correlation = np.clip(cross/(norm*np.sqrt(np.maximum(1e-5,variance))),-1,1)
    rotation,dy,dx = np.unravel_index(correlation.argmax(),correlation.shape)
    peak = float(correlation[rotation,dy,dx])
    _,similar_y,similar_x = np.nonzero(correlation>=peak-SETTINGS['phase_tie_correlation'])
    phases = []
    for best,similar in ((dx,similar_x),(dy,similar_y)):
        distance = abs((similar-best+8)%16-8)
        phases.append(float(best/16) if distance.max()<=SETTINGS['maximum_phase_ambiguity_bins'] else None)
    return peak,phases


def _fits(gray,mask,atlas,ids,priors,aspect):
    rows,cols=np.nonzero(mask)
    if len(rows)>5000:
        select=np.linspace(0,len(rows)-1,5000).astype(int)
        rows,cols=rows[select],cols[select]
    samples=gray[rows,cols]
    results=[]
    observed=[_observed_profile(gray,mask,axis) for axis in (0,1)]
    fits=[_periods(gray.shape[1-axis],observed[axis],
                   atlas.profiles[ids].reshape(-1,16)) for axis in (0,1)]
    if any(fit is None for fit in fits):
        return []
    # Each texture provides its own candidate horizontal/vertical periods, in its two axis orders.
    search=[]
    for i,index in enumerate(ids):
        for swap in (0,1):
            errors=[fit[1][2*i+(axis^swap)] for axis,fit in enumerate(fits)]
            periods=[]
            for (values,_),err in zip(fits,errors):
                best=int(np.argmin(err))
                distinct=abs(np.log(values/values[best]))>.09
                other=int(np.argmin(np.where(distinct,err,np.inf)))
                periods.append([(values[k],err[k]) for k in (best,other)])
            min_error=periods[0][0][1]+periods[1][0][1]
            # One well-resolved axis can propose the partner pitch using the shape
            # of a square block restored by the geometry. The atlas still tests both.
            pairs=[(u,v) for u in periods[0] for v in periods[1]]
            if aspect:
                ratio=gray.shape[0]*aspect/gray.shape[1]
                for axis in (0,1):
                    values=fits[1-axis][0]
                    for anchor in periods[axis]:
                        p=anchor[0]*(ratio if axis==0 else 1/ratio)
                        if values[0]<=p<=values[-1]:
                            partner=(p,float(np.interp(p,values,errors[1-axis])))
                            pairs.append((anchor,partner) if axis==0 else (partner,anchor))
            search.append((min_error+priors[index]/55,index,swap,pairs))
    for _,index,swap,pairs in sorted(search,key=lambda s:s[0])[:SETTINGS['period_shortlist']]:
        rotations=np.array([swap,swap+2,swap+4,swap+6])
        for (px,ex),(py,ey) in pairs:
            alignment=_texture_alignment(samples,cols,rows,px,py,atlas,index,rotations)
            if alignment is None:
                continue
            corr,phase=alignment
            support=[]
            for (lags,ac),p in zip(observed,(px,py)):
                repeat=float(np.interp(p,lags,ac,right=-1))
                trough=float(np.min(ac[(lags>p*.22)&(lags<p*.86)],initial=1))
                support.append(max(0., min(repeat,repeat-trough)))
            aspect_error=abs(np.log(gray.shape[1]*py/(gray.shape[0]*px*aspect))) if aspect else 0.
            score=corr-.65*(ex+ey)-priors[index]/55+.18*sum(support)-.35*min(1.,aspect_error)
            results.append((score,index,px,py,phase,corr,ex,ey,support))
    return sorted(results,key=lambda s:-s[0])


def count(sample,atlas,version='26.3'):
    """Estimate chart spans by fitting periodic atlas faces. Monocular depth is unused."""
    if atlas.provenance['version']!=version:
        raise ValueError('Block Counter atlas is for a different game version')
    result=dict(method=METHOD,counts=[None,None],phase=[None,None],**_unknown())
    if sample is None:
        return dict(result,evidence=dict(reason='no sampling chart'))
    original=np.asarray(sample['rgb'])
    mask=np.asarray(sample['mask'],bool)
    ratio=min(1.,SETTINGS['analysis_side']/max(original.shape[:2]))
    size=tuple(max(1,round(n*ratio)) for n in original.shape[:2][::-1])
    rgb,mask=_resample(original,mask,size) if ratio<1 else (original.astype(np.float32),mask)
    if mask.sum()<SETTINGS['minimum_mask_pixels']:
        return dict(result,evidence=dict(reason='insufficient support'))
    indices,prior=_shortlist(rgb,mask,atlas)
    result.update(_labels(atlas,indices))
    gray=rgb.mean(2)
    if np.std(gray[mask])<1.8 or min(size)<6:
        return dict(result,evidence=dict(reason='color-only material; insufficient texture for pitch'))
    aspect=sample.get('normal_aspect')
    if not aspect and sample.get('native_size') and 'quad' in sample:
        aspect=_aspect(transform(sample['quad']),sample['native_size'])
    fits=_fits(gray,mask,atlas,indices,prior,aspect)
    if not fits:
        return dict(result,evidence=dict(reason='insufficient texture profile'))
    _,index,px,py,phases,corr,ex,ey,support=fits[0]
    result.update(_labels(atlas,[f[1] for f in fits]+list(indices)))
    supported=corr>=SETTINGS['minimum_repeat_correlation']
    aspect_agrees=bool(aspect and abs(np.log(size[0]*py/(size[1]*px*aspect)))<.06)
    for axis,(length,p,phase,err) in enumerate(zip(size,(px,py),phases,(ex,ey))):
        measured=support[axis]>=SETTINGS['minimum_axis_repeat']
        transferred=aspect_agrees and support[1-axis]>.36 and corr>.55
        if supported and err<=SETTINGS['maximum_axis_error'] and (measured or transferred):
            result['counts'][axis]=float(length/p)
            result['phase'][axis]=phase
    return dict(result,evidence=dict(correlation=corr,profile_error=[float(ex),float(ey)],repeat_support=support,
                reason='atlas repeat comparison',support_pixels=int(mask.sum())))

