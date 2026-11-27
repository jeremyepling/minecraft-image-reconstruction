"""Reuse an accepted photographed return fit only for that visible back-wall mask."""
from hashlib import sha256

import numpy as np


def projection_id(camera):
    p = np.asarray(camera.get('P'),dtype='<f8')
    return sha256(p.tobytes()).hexdigest() if p.shape==(3,4) else None


def measurement(camera, axis, fit):
    return dict(method='paired-visible-returns', projection=projection_id(camera), axis=int(axis),
                host=fit['host'], side=fit['side'], sill=fit['sill'], front=int(fit['level']),
                signed_depth=int(fit['depth']), rim=list(fit['rim']))


def levels(parts, camera):
    """Require surviving three-face support and the exact same estimated camera."""
    annotated = [p for p in parts if p.get('inset_back_plane')]
    if not annotated or camera.get('P') is None:
        return {}
    fingerprint=projection_id(camera)
    p=np.asarray(camera['P'],float)
    eye=np.linalg.solve(p[:,:3],-p[:,3])
    names={part['id'] for part in parts}
    measured={}
    for part in annotated:
        m=part['inset_back_plane']
        if m.get('method')!='paired-visible-returns' or m.get('projection')!=fingerprint:
            continue
        if {m.get('host'),m.get('side'),m.get('sill')} - names:
            continue
        axis,front,depth=m['axis'],m['front'],m['signed_depth']
        if axis not in (0,2) or abs(depth) not in (1,2,3,4) or (front-eye[axis])*depth<=0:
            continue
        measured[part['id']]=(axis,front+depth)
    return measured
