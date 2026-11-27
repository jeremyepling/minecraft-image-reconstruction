"""Join photographed wall fragments when a diagonal cut crosses the same repeated texels."""
from itertools import combinations

import numpy as np
from PIL import ImageFilter

from engine.pillar_seams import native_mask, orientation
from engine.plane_consensus import with_mask
from engine.surface_perimeter import _native_alpha


def texture_fit(rgb, mask, box, axis, eye, inv, levels):
    # Cut out one native pixel at the contour before fitting texture.
    padded=np.pad(mask,1)
    interior=mask&padded[1:-1,:-2]&padded[1:-1,2:]&padded[:-2,1:-1]&padded[2:,1:-1]
    y,x=np.nonzero(interior)
    if len(y)<80:
        return None
    if len(y)>3200:
        take=np.random.default_rng(19).choice(len(y),3200,replace=False)
        y,x=y[take],x[take]
    x, y = x + box[0], y + box[1]
    color=rgb[y,x].astype(float)
    intensity=color.mean(1)
    if intensity.std()<3.:
        return None
    ray=np.c_[x+.5,y+.5,np.ones(len(x))]@inv.T
    if np.min(abs(ray[:,axis]))<1.e-7:
        return None
    free=[k for k in range(3) if k!=axis]
    tests=[]
    for level in levels:
        d=(level-eye[axis])/ray[:,axis]
        if d.min()<=0:
            continue
        uv=eye[free]+ray[:,free]*d[:,None]
        tile=np.floor(uv).astype(int)
        fold=tile.sum(1)%2
        texel=np.floor(uv*16).astype(int)%16
        code=16*texel[:,1]+texel[:,0]
        maps=[]; counts=[]
        for side in (0,1):
            take=fold==side
            n=np.bincount(code[take],minlength=256)
            maps.append(np.bincount(code[take],weights=intensity[take],minlength=256)/n.clip(1))
            counts.append(n)
        eligible=(counts[0]>=2)&(counts[1]>=2)
        if eligible.sum()<28:
            continue
        corr=correlation(maps[0][eligible],maps[1][eligible])
        template=(maps[0]*counts[0]+maps[1]*counts[1])/(counts[0]+counts[1]).clip(1)
        tests.append((corr,level,template,eligible))
    tests.sort(key=lambda o:o[0],reverse=True)
    if len(tests)<2:
        return None
    best,level,template,valid=tests[0]
    return dict(corr=best,level=level,template=template,valid=valid,margin=best-tests[1][0])


def correlation(a,b):
    if len(a)<8:
        return -1.
    a=a-a.mean(); b=b-b.mean()
    return float(a@b/max(1.e-7,np.linalg.norm(a)*np.linalg.norm(b)))


def contact_pixels(left, right):
    mask_a, box_a = left
    mask_b, box_b = right
    l, t = max(box_a[0], box_b[0]) - 4, max(box_a[1], box_b[1]) - 4
    r, b = min(box_a[2], box_b[2]) + 4, min(box_a[3], box_b[3]) + 4
    if r <= l or b <= t:
        return None
    def window(mask, box):
        out = np.zeros((b-t, r-l), bool)
        x0, y0 = max(l, box[0]), max(t, box[1])
        x1, y1 = min(r, box[2]), min(b, box[3])
        if x1 <= x0 or y1 <= y0:
            return out
        out[y0-t:y1-t, x0-l:x1-l] = mask[y0-box[1]:y1-box[1], x0-box[0]:x1-box[0]]
        return out
    a, b_mask = window(mask_a, box_a), window(mask_b, box_b)
    if (a & b_mask).sum() > .10 * min(mask_a.sum(), mask_b.sum()):
        return None
    a, b_mask = a & ~b_mask, b_mask & ~a
    contact = np.zeros_like(a)
    for step in range(1, 5):
        contact[:-step] |= a[:-step] & b_mask[step:]
        contact[step:] |= a[step:] & b_mask[:-step]
        contact[:, :-step] |= a[:, :-step] & b_mask[:, step:]
        contact[:, step:] |= a[:, step:] & b_mask[:, :-step]
    y, x = np.nonzero(contact)
    if len(x) < 48 or max(np.ptp(x), np.ptp(y)) < 32:
        return None
    return np.column_stack((x + l + .5, y + t + .5, np.ones(len(y))))


def diagonal_span(contact, inv, axis):
    """Unit-depth length of a straight seam that cuts across both wall axes."""
    rays = contact @ inv.T
    if np.min(abs(rays[:,axis])) < 1.e-7:
        return 0.
    uv = np.delete(rays / rays[:,axis,None], axis, axis=1)
    uv -= uv.mean(axis=0)
    _, singular, basis = np.linalg.svd(uv, full_matrices=False)
    if singular[1] > .10 * singular[0] or min(abs(basis[0])) < np.sin(np.radians(12)):
        return 0.
    return float(np.ptp(uv @ basis[0]))


def refine(image, parts, camera):
    """Return the original parts on insufficient image, camera or texture evidence."""
    if not parts or camera is None or camera.get('P') is None:
        return parts, []
    try:
        projection = np.asarray(camera['P'], float)
    except (ValueError, TypeError):
        return parts, []
    if projection.shape != (3,4) or not np.isfinite(projection).all() or np.linalg.matrix_rank(projection[:,:3]) < 3:
        return parts, []
    p, rotation = orientation(camera, image.size)
    inv = np.linalg.inv(p[:,:3])
    eye = -inv @ p[:,3]
    width, height = image.size
    floor_ray = inv @ [width / 2, .72 * height, 1.]
    if abs(floor_ray[1]) < 1.e-7 or eye[1] * floor_ray[1] >= 0:
        return parts, []
    center = eye - floor_ray * (eye[1] / floor_ray[1])
    candidates = {}
    for i, part in enumerate(parts):
        normal = part.get('normal_camera')
        if normal is None or not part.get('support'):
            continue
        axis = int(np.argmax(abs(rotation.T @ normal)))
        if axis == 1:
            continue
        local = _native_alpha(part['support'], image.size)
        if local is None:
            continue
        alpha, box = local
        mask = np.asarray(alpha, bool)
        if mask.sum() >= 128:
            candidates[i] = (axis, (mask, box))
    rgb = None
    fits = {}
    proposals = []
    for i, j in combinations(candidates, 2):
        axis, left = candidates[i]
        other_axis, right = candidates[j]
        if axis != other_axis or np.dot(parts[i]['normal_camera'], parts[j]['normal_camera']) < .985:
            continue
        contact = contact_pixels(left, right)
        if contact is None:
            continue
        span = diagonal_span(contact, inv, axis)
        if span <= 0:
            continue
        if rgb is None:
            rgb = np.asarray(image.convert('RGB'))
        origin = round(center[axis])
        levels = range(origin - 24, origin + 25)
        for k, (mask, box) in [(i, left), (j, right)]:
            if k not in fits:
                fits[k] = texture_fit(rgb, mask, box, axis, eye, inv, levels)
        a, b = fits[i], fits[j]
        if not a or not b or a['level'] != b['level'] or abs(a['level'] - eye[axis]) * span < 2:
            continue
        if min(a['corr'], b['corr']) < .70 or min(a['margin'], b['margin']) < .12:
            continue
        common = a['valid'] & b['valid']
        if common.sum() < 28:
            continue
        av, bv = a['template'][common], b['template'][common]
        if correlation(av,bv) >= .72 and abs(av.mean()-bv.mean()) <= 8:
            proposals.append((i,j))
    if not proposals:
        return parts, []
    # Abstain on ambiguous chains, where the two outer masks never witnessed the seam.
    involved = [i for pair in proposals for i in pair]
    pairs = [(i,j) for i,j in proposals if involved.count(i) == involved.count(j) == 1]
    if not pairs:
        return parts, []
    replaced = {k for pair in pairs for k in pair}
    result = [part for k, part in enumerate(parts) if k not in replaced]
    gray = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)), np.float32)[...,None]
    groups = []
    for i,j in pairs:
        largest = max((i,j), key=lambda k: candidates[k][1][0].sum())
        fields = {k:v for k,v in parts[largest].items() if k not in ('context_crop','source_crop')}
        merged = native_mask(parts[i], image.size) | native_mask(parts[j], image.size)
        item = with_mask(fields, merged, image.size, gray, np.ones(2))
        result.append(dict(item, id=parts[largest]['id']+'-phase-union'))
        groups.append([parts[i]['id'], parts[j]['id']])
    return result, groups
