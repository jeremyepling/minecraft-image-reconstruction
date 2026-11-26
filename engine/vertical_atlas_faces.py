"""Confirm missing positions on vertical surfaces from their photographed tile phase."""
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.thin_atlas_faces import StripEvidence, _corr, _match
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _sample_face

FEATURES = ('split','corr','phase','depth','color','margin','texels','std','mip','neighboring','count',
            'fraction','anchored','plane_donors','donor_dist','empty','area_fraction','inside','aligned')
WEIGHTS = Path(__file__).with_name('vertical_face_weights.json')
MODEL_SHA256 = '26dc22c52206789bcda1335df8eac9ecb82c909d06dff4d7d1c0579b0a8d9599'


def vector(row):
    scaled = {'texels','std','count','plane_donors','donor_dist'}
    return [np.log1p(row[k]) if k in scaled else row[k] for k in FEATURES]


@lru_cache(maxsize=2)
def weights(path,mtime):
    raw = path.read_bytes()
    if path == WEIGHTS and sha256(raw.replace(b'\r\n',b'\n')).hexdigest() != MODEL_SHA256:
        raise ValueError('Vertical atlas weights do not match the selected model')
    model=json.loads(raw)
    if model['features'] != list(FEATURES):
        raise ValueError('Incompatible vertical atlas model')
    for k in ('weights','scale','mean'):
        model[k]=np.asarray(model[k],float)
        if model[k].shape != (len(FEATURES),) or not np.isfinite(model[k]).all():
            raise ValueError('Invalid vertical atlas model vector')
    model['bias']=float(model['bias'])
    if not np.isfinite(model['bias']) or not (model['scale']>0).all():
        raise ValueError('Invalid vertical atlas model scale or bias')
    return model


def candidate_faces(prior):
    # An adjacent cell on the same plane nominates a face. The photo decides occupancy.
    candidates = Counter()
    for cell in prior:
        for axis in (0, 2):
            for tangent in (1, 2-axis):
                for step in (-1, 1):
                    neighbor = list(cell)
                    neighbor[tangent] += step
                    neighbor = tuple(neighbor)
                    if neighbor not in prior:
                        candidates[neighbor, axis] += 1
    return candidates


def measure(ctx, cell, axis, adjacent, joint=None):
    if cell[axis] <= ctx.eye[axis] <= cell[axis]+1:
        return None
    sign = 1 if ctx.eye[axis] > cell[axis]+1 else -1
    plane = cell[axis] + int(sign > 0)
    hit = _sample_face(ctx.native, cell, axis, sign, ctx.shape)
    if hit is None:
        return None
    box, face, depth, _, _ = hit
    front = face & (depth > ctx.depth[box]+1e-8)
    if joint is not None:
        owner, axes, ids = joint
        front &= (owner[box] == ids[cell]) & (axes[box] == axis)
    witness = front & ctx.masks[axis][box] & (ctx.axes[box] != axis)
    count = int(witness.sum())
    if count < 16 or count < .55*front.sum():
        return None
    y,x = np.nonzero(witness)
    indices = np.linspace(0,len(x)-1,min(220,len(x)),dtype=int)
    x,y = x[indices]+box[1].start, y[indices]+box[0].start
    colors = ctx.rgb[y,x]
    rays = ctx._rays(x,y)
    free = [i for i in range(3) if i != axis]
    slope = rays[:,free]/rays[:,[axis]]
    uv = ctx.eye[free] + (plane-ctx.eye[axis])*slope
    bins = np.floor(uv*16).astype(int)
    match = _match(bins, colors, ctx.atlas, ctx.maps)
    if not match or match['split'] < .40 or match['color'] > 14:
        return None
    template = ctx.maps[match['sprite'], match['mip'], match['turn']]
    gray = colors.mean(1)
    def correlation(b):
        return float(_corr(template[16*(b[:,1]%16)+b[:,0]%16],gray))
    rivals = [correlation(np.floor((uv+shift*slope)*16).astype(int)) for shift in (-1,1)]
    same_plane = [point for point,a,_ in ctx.anchors.get(match['sprite'],[])
                  if a == axis and point[axis] == cell[axis]]
    near = min((sum(abs(int(x)-int(y)) for x,y in zip(cell,point)) for point in same_plane), default=32)
    empty = float((ctx.depth[box][witness] == 0).mean())
    total = int(front.sum())
    inside = int(np.count_nonzero(front & ctx.union[box]))
    aligned = int(np.count_nonzero(front & ctx.masks[axis][box]))
    for rival_axis in free:
        if cell[rival_axis] <= ctx.eye[rival_axis] <= cell[rival_axis]+1:
            continue
        rival = _sample_face(ctx.native,cell,rival_axis,1 if ctx.eye[rival_axis]>cell[rival_axis]+1 else -1,ctx.shape)
        if rival is None:
            continue
        area,mask,inv_z,*_ = rival
        exposed = mask & (inv_z>ctx.depth[area]+1e-8)
        if joint is not None:
            exposed &= (owner[area]==ids[cell]) & (axes[area]==rival_axis)
        total += int(exposed.sum())
        inside += np.count_nonzero(exposed & ctx.union[area])
        aligned += np.count_nonzero(exposed & ctx.masks[rival_axis][area])
    return dict(match, neighboring=adjacent, count=count, fraction=count/int(front.sum()),
                anchored=match['sprite'] in ctx.anchors, depth=correlation(bins)-max(rivals),
                plane_donors=len(same_plane),donor_dist=near,empty=empty,
                area_fraction=count/int(face.sum()),inside=inside/total,aligned=aligned/total)


def accepted(row, mode):
    if not row or row['texels'] < 10 or row['phase'] < .04 or row['depth'] < .015:
        return False
    if isinstance(mode,tuple):
        model,threshold=mode
        linear=((vector(row)-model['mean'])/model['scale'])@model['weights']+model['bias']
        return linear>=np.log(threshold/(1-threshold))
    if mode == 'strict':
        return row['split'] >= .80 and row['color'] <= 8 and row['phase'] >= .12 and row['anchored']
    if mode == 'balanced':
        return row['split'] >= .68 and row['color'] <= 9 and (row['anchored'] or row['neighboring'] >= 2)
    if mode == 'wide':
        return row['split'] >= .55 and row['color'] <= 11
    raise ValueError(mode)


def observations(image, parts, camera, prior, atlas):
    source = getattr(atlas,'provenance',None) or {}
    if not prior or camera.get('P') is None or not any(p.get('support') for p in parts):
        return None, {}
    if source.get('version') != '26.3' or source.get('atlas_sha256') != ATLAS_SHA256:
        return None, {}
    ctx = StripEvidence(image,parts,camera,prior,atlas)
    rows = {}
    for (cell,axis),neighbors in candidate_faces(prior).items():
        row = measure(ctx,cell,axis,neighbors)
        if row is not None:
            rows[cell,axis] = row
    return ctx,rows


def select(ctx, rows, prior, mode):
    proposed = {key for key,row in rows.items() if accepted(row,mode)}
    if not proposed:
        return prior
    joint = ctx.joint({cell for cell,axis in proposed})
    confirmed = {cell for cell,axis in proposed
                 if accepted(measure(ctx,cell,axis,rows[cell,axis]['neighboring'],joint),mode)}
    if not confirmed:
        return prior
    return _keep_prior_visible(ctx.size,ctx.P,ctx.eye,prior,confirmed)


def recover(image, parts, camera, prior, atlas, mode=.80):
    if isinstance(mode,float):
        try:
            mode=(weights(WEIGHTS,WEIGHTS.stat().st_mtime_ns),mode)
        except (OSError,TypeError,ValueError,KeyError,OverflowError):
            return prior
    ctx, rows = observations(image,parts,camera,prior,atlas)
    return select(ctx,rows,prior,mode)


def confirm_after_roof(image, parts, camera, prior, atlas):
    """Check vertical faces exposed by the final roof and depth decisions."""
    if not prior or not parts or camera.get('P') is None or atlas is None:
        return prior
    current = prior
    for _ in range(3):
        confirmed = recover(image, parts, camera, current, atlas, mode=.95)
        if confirmed == current:
            break
        current = confirmed
    return current
