"""Extract single visible block faces after a counter establishes spacing and grid origin."""
import numpy as np

from engine.geometry import UNIT, face, project, transform
from engine.surface_model import canonical


def tile_sample(image, quad):
    quad = canonical(quad)
    rgb = face(np.asarray(image), quad, size=32)
    return dict(quad=quad, rgb=rgb, mask=np.ones((32,32),bool), native_size=image.size, source_image=image)


def complete_footprints(sample, result):
    if sample is None:
        return
    ns = result.get('counts')
    phase = result.get('phase')
    if ns is None or phase is None or any(x is None or not np.isfinite(x) for x in (*ns,*phase)):
        return
    mask = sample['mask']
    h,w = mask.shape
    ns = np.asarray(ns,float)
    if min(ns)<=0 or np.any(ns>np.array([w,h])):
        return
    phase = np.asarray(phase,float)%1.
    origins = [np.arange(int(np.floor(n-p+1e-7)) )+p for n,p in zip(ns,phase)]
    H = transform(sample['quad'])
    for v,y in enumerate(origins[1]):
        for u,x in enumerate(origins[0]):
            uv = (UNIT + [x,y]) / ns
            lo = np.floor(uv[0]*[w,h]+1e-7).astype(int).clip([0,0],[w,h])
            hi = np.ceil(uv[2]*[w,h]-1e-7).astype(int).clip([0,0],[w,h])
            if min(hi-lo)<1 or not mask[lo[1]:hi[1],lo[0]:hi[0]].all():
                continue
            q = project(H,uv)
            if (q<0).any() or (q>sample['source_image'].size).any():
                continue
            yield [u,v], q


def visible_tiles(sample, result):
    # Preserve the archived four-pixel edge rule for historical count replays.
    for cell, q in complete_footprints(sample, result):
        if np.min(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))>=4.:
            yield dict(cell=cell, **tile_sample(sample['source_image'],q))


def material_cells(sample, result, match):
    from engine.face_evidence import effective_width, MIN_FACE_WIDTH

    ranked, skipped = [], []
    for cell, quad in complete_footprints(sample, result):
        width = effective_width(quad)
        record = dict(cell=cell, quad=quad.tolist(), effective_width_px=width)
        if width < MIN_FACE_WIDTH:
            skipped.append(record)
        else:
            tile = tile_sample(sample['source_image'], quad)
            ranked.append(dict(record, quad=tile['quad'], material=match(tile)))
    return ranked, skipped
