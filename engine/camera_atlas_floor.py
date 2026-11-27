"""Register a visible ground texel lattice to verified vanilla 16 by 16 sprites."""
from functools import lru_cache
import json
from pathlib import Path
from zlib import error as CompressionError

import numpy as np

from engine.atlas_count import CALIBRATED_INDEX, _shortlist, load_atlas, prepared_atlas_paths
from engine.camera_texel_floor import _ground, lattice


@lru_cache(maxsize=2)
def _verified_atlas(path, version, fingerprint):
    """The cache key includes file timestamps and sizes so a repair can take effect."""
    return load_atlas(path, version)


def _file_stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _sheet_path(manifest, stamp):
    return manifest.parent / json.loads(manifest.read_bytes())['atlas']


def _atlas(version, atlas_path=None):
    paths = [Path(atlas_path)] if atlas_path is not None else prepared_atlas_paths(version)
    for path in paths:
        path = path.resolve()
        try:
            manifest = path / 'atlas.json'
            stamp = _file_stamp(manifest)
            sheet = _sheet_path(manifest, stamp)
            fingerprint = stamp, sheet, _file_stamp(sheet), _file_stamp(CALIBRATED_INDEX)
            return _verified_atlas(path, version, fingerprint)
        except (OSError, ValueError, TypeError, IndexError, KeyError, EOFError, CompressionError):
            continue
    return None


def _observations(image,inv):
    rgb=np.asarray(image.convert("RGB"))
    h,w=rgb.shape[:2]
    y,x=np.mgrid[int(.74*h):int(.984*h):2,0:w:2]
    screen=np.stack([x+.5,y+.5,np.ones_like(x)],axis=0).reshape(3,-1)
    uv=inv@screen
    uv=uv[:2]/uv[2]
    texel=np.floor(uv).astype(int)
    # Texel centers minimize one-pixel filtering and boundary roundoff.
    center=np.max(abs(uv-texel-.5),axis=0)<.36
    colors=rgb[y,x].reshape(-1,3).astype(float)
    median=np.median(colors[::11],axis=0)
    mask=center & (np.max(abs(colors-median),axis=1)<14)
    code=(texel[0]%16)+16*(texel[1]%16)
    gray=colors.mean(1)
    folds=[]
    for partition in (x.ravel()<w/2,x.ravel()>=w/2):
        sel=mask&partition
        counts=np.bincount(code[sel],minlength=256)
        if min(counts)<3: return None
        # Winsorizing around the median limits sparse object pixels on the floor.
        intensity=np.clip(gray[sel],*np.percentile(gray[sel],[2,98]))
        folds.append((np.bincount(code[sel],intensity,minlength=256)/counts).reshape(16,16))
    return folds,colors[mask]


def register(image,inv,atlas):
    data=_observations(image,inv)
    if data is None: return None,{"reason":"incomplete floor texel coverage"}
    folds,colors=data
    indices,_=_shortlist(colors[::5],np.ones(len(colors[::5]),bool),atlas)
    spectra=atlas.spectra[indices]
    std=atlas.patterns[indices].std((2,3)).clip(.01)[:,:,None,None]
    matches=[]
    for patch in folds:
        patch=patch-patch.mean()
        if patch.std()<.10: return None,{"reason":"flat rectified floor"}
        corr=np.fft.irfft2(np.conj(np.fft.rfft2(patch))*spectra,s=(16,16),axes=(-2,-1))/(256*std*patch.std())
        # First compete across sprites and orientations at each offset. The
        # runner-up below then measures ambiguity about the block origin itself.
        offsets=corr.max((0,1))
        y,x=np.unravel_index(np.argmax(offsets),offsets.shape)
        ti,orientation=np.unravel_index(np.argmax(corr[:,:,y,x]),corr.shape[:2])
        best = float(offsets[y,x])
        margin = float(offsets[y,x] - np.sort(offsets.ravel())[-2])
        matches.append((int(x), int(y), str(atlas.labels[indices[ti]]), int(orientation), best, margin))
    evidence={"atlas_matches":matches,"atlas_sha256":atlas.provenance["atlas_sha256"]}
    common_origin_and_sprite = matches[0][:4] == matches[1][:4]
    if not common_origin_and_sprite or min(m[4] for m in matches)<.75 or min(m[5] for m in matches)<.12:
        return None,dict(reason="ground patches lack a unique common atlas phase",**evidence)
    return np.array(matches[0][:2]),evidence


def _predict_band(image,version,lower,atlas_path):
    def abstain(reason,evidence):
        return dict(P=None,scale_status="uncertain",reason=reason,evidence=evidence)
    inv,evidence=lattice(image,lower)
    evidence["lower_bound_fraction"]=lower
    if inv is None: return abstain(evidence["reason"],evidence)
    atlas=_atlas(version,atlas_path)
    if atlas is None: return abstain("calibrated atlas unavailable",evidence)
    offset,registration=register(image,inv,atlas)
    evidence.update(registration)
    if offset is None: return abstain(registration["reason"],evidence)
    inv[:2]=(inv[:2]+offset[:,None]*inv[2])/16
    camera=_ground(inv,*image.size)
    if camera is None: return abstain("inconsistent metric axes",evidence)
    P,f,err=camera
    evidence.update(focal_px=f,axis_length_disagreement=err)
    return dict(P=P,scale_status="supported",reason="two floor texel pencils registered to a vanilla sprite in both image halves",evidence=evidence)


def predict(image,version="26.3",*,atlas_path=None):
    if version!="26.3":
        return dict(P=None,scale_status="uncertain",reason="unsupported atlas version",evidence={})
    if min(image.size)<256 or max(image.size)>4096:
        return dict(P=None,scale_status="uncertain",reason="outside native texel sampling range",evidence={})
    for lower in (.74,.83,.86):
        result=_predict_band(image,version,lower,atlas_path)
        if result["P"] is not None or result["reason"]=="calibrated atlas unavailable": break
    return result
