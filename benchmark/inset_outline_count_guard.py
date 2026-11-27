"""Admit a learned outline only when its independent chart count fits its visible grid span."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageFilter

from engine import surface_model,texture_guided_count
from engine.pillar_seams import native_mask,orientation
from engine.plane_consensus import with_mask


def _span(size, camera, normal, level, mask):
    P,rotation=orientation(camera,size)
    axis=int(np.argmax(abs(rotation.T@normal)))
    if axis==1:
        return None
    inv=np.linalg.inv(P[:,:3])
    eye=-inv@P[:,3]
    y,x=np.nonzero(mask)
    rays=np.stack((x+.5,y+.5,np.ones(x.shape)),axis=-1)@inv.T
    xyz=eye+(level-eye[axis])*rays/rays[:,[axis]]
    cells,tally=np.unique(np.floor(xyz[:,[2-axis,1]]).astype(int),axis=0,return_counts=True)
    if not np.any(tally>=4):
        return None
    span=np.ptp(cells,axis=0)+1
    core=np.ptp(cells[tally>=4],axis=0)+1
    return span.tolist() if np.array_equal(span,core) else None


def _dims(image,parts,atlas):
    with TemporaryDirectory() as tmp:
        rows=surface_model._count_regions(image,parts,'26.3',texture_guided_count.METHOD,atlas,Path(tmp))['surfaces']
    return {r['id']:r['dimensions']['blocks'] for r in rows}


def apply(image,before,after,events,camera,atlas=None,*,collar=False):
    if not events:
        return before,[]
    parts={p['id']:p for p in after}
    groups=defaultdict(list)
    for e in events:
        groups[e['host']].append(e)
    result=[]
    logs=[]
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    for host in before:
        key=host['id']
        if key not in groups:
            result.append(host)
            continue
        candidates=[parts[e['output']] for e in groups[key]]
        if collar:
            candidates=[
                with_mask(kid,np.asarray(Image.fromarray(native_mask(kid,image.size)).filter(ImageFilter.MinFilter(5)),bool),
                          image.size,gray,np.ones(2)) for kid in candidates]
            parts.update((p['id'],p) for p in candidates)
        dims=_dims(image,[host,*candidates],atlas)
        remainder=native_mask(host,image.size)
        picked=[]
        for e in groups[key]:
            kid=parts[e['output']]
            mask=native_mask(kid,image.size)
            span=_span(image.size,camera,host['normal_camera'],e['photo_grid_level'],mask)
            if span is not None and dims[kid['id']]==span:
                picked.append(dict(e,photo_grid_span=span))
                remainder&=~mask
        if not picked:
            result.append(host)
            continue
        parent={k:v for k,v in host.items() if k not in ('source_crop','context_crop','corner_junctions','outline','sampling_chart')}
        residual=with_mask(parent,remainder,image.size,gray,np.ones(2))
        new=_dims(image,[residual],atlas)[key]
        changed_resolved=any(a is not None and a!=b for a,b in zip(dims[key],new))
        if changed_resolved or (collar and dims[key]!=new):
            result.append(host)
            continue
        result.extend(parts[e['output']] for e in picked)
        result.append(residual)
        logs.extend(picked)
    return result,logs
