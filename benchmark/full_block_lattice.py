"""Screenshot-only whole-sprite anchors, shared lattice phase and observed subface placement."""
from collections import Counter, defaultdict
from functools import lru_cache
import os

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import cv2
import numpy as np
from engine.atlas_count import _shortlist
from engine.camera_atlas_floor import _atlas
from engine.cube_axes import fit_axes
from engine.geometry import project, face
from engine.surface_model import intrinsic
from engine.visible_raster import observed_support

cv2.setNumThreads(1)
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
STRIDE = 4
FALLBACK = 'smooth_stone'


@lru_cache(maxsize=1)
def atlas():
    value = _atlas('26.3')
    if value is None:
        raise ValueError('Missing calibrated Java 26.3 atlas')
    return value


def line_camera(segments, size):
    # Source-only vanishing directions; floor height, recorded FOV, pose and distance are unknown.
    lines = [dict(p=line[:2], q=line[2:]) for line in segments]
    vps = np.asarray(fit_axes(lines, size))
    if len(vps) != 3:
        return None
    center = np.array(size) / 2
    centered = vps[:, :2] - vps[:, 2, None] * center
    f2 = [-centered[i] @ centered[j] / (vps[i, 2] * vps[j, 2])
          for i, j in ((0, 1), (0, 2), (1, 2)) if abs(vps[i, 2] * vps[j, 2]) > 1e-10]
    valid = [x for x in f2 if (.32 * size[1])**2 < x < (3 * size[1])**2]
    if not valid:
        return None
    k = intrinsic(size, np.sqrt(np.median(valid)))
    directions = np.linalg.solve(k, vps.T)
    directions /= np.linalg.norm(directions, axis=0)
    up = int(np.argmax(abs(directions[1])))
    other = [i for i in range(3) if i != up]
    directions = directions[:, [other[0], up, other[1]]]
    if directions[1, 1] > 0:
        directions[:, 1] *= -1
    if directions[0, 0] < 0:
        directions[:, 0] *= -1
    u, _, vt = np.linalg.svd(directions)
    return k @ (u @ vt)


def mask_parts(parts, size, linear):
    shape = (size[1] // STRIDE, size[0] // STRIDE)
    focal = np.linalg.norm(linear[0] - size[0] / 2 * linear[2])
    rotation = np.linalg.solve(intrinsic(size, focal), linear)
    result = []
    for part in parts:
        if not part.get('support') or part.get('normal_camera') is None:
            continue
        mask = observed_support(part['support'], shape, size)
        y, x = np.nonzero(mask)
        if len(x) < 5:
            continue
        cosine = rotation.T @ part['normal_camera']
        axis = int(np.argmax(abs(cosine)))
        if abs(cosine[axis]) < .67:
            continue
        result.append(dict(id=part['id'], mask=mask, axis=axis, x=x, y=y, area=len(x)))
    return sorted(result, key=lambda p: -p['area'])


def rays(linear, xy):
    return np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(linear).T


def rectified(part, rgb, linear):
    # Use an arbitrary unit-distance chart to search independently of any floor scale or phase.
    axis = part['axis']
    free = [i for i in range(3) if i != axis]
    xy = (np.column_stack((part['x'], part['y'])) + .5) * STRIDE
    r = rays(linear, xy)
    sign = np.sign(np.median(r[:, axis]))
    if np.min(sign * r[:, axis]) < .08:
        return None
    coords = r[:, free] / (r[:, [axis]] * sign)
    low, high = coords.min(0), coords.max(0)
    # A chart of at most 132 x 132 makes multiscale sprite correlation practical.
    factor = min(132 / max(high-low), np.sqrt(part['area'] * 16 / max(.00001, np.prod(high-low))))
    low -= 2/factor
    shape = np.ceil((high-low)*factor + 3).astype(int)
    if min(shape) < 9:
        return None
    loc = np.zeros((3, 3))
    loc[axis, 2] = sign
    loc[free, 2] = low
    loc[free[0], 0] = loc[free[1], 1] = 1 / factor
    H = linear @ loc
    u, v = np.meshgrid(np.arange(shape[0])+.5, np.arange(shape[1])+.5)
    pix = project(H, np.stack((u, v), axis=-1))
    x, y = np.moveaxis(pix, -1, 0)
    chart = cv2.remap(rgb, (x-.5).astype('f'), (y-.5).astype('f'), cv2.INTER_LINEAR)
    mask = cv2.remap(part['mask'].astype('B'), (x/STRIDE-.5).astype('f'), (y/STRIDE-.5).astype('f'), cv2.INTER_NEAREST)
    return chart, mask, loc, H, factor


def verify_sprite(rgb, quad):
    """Check the native source, so a resized six-pixel stripe cannot certify a full tile."""
    cat = atlas()
    patch = face(rgb, quad, 16)
    mask = np.ones((16,16), dtype=bool)
    short, prior = _shortlist(patch, mask, cat)
    gray = patch.astype('f').mean(2)
    gray -= gray.mean()
    if gray.std() < 2:
        return None
    corr = (cat.patterns[short] * gray).mean((2,3)) / gray.std()
    score = corr - prior[short,None]/60
    row, orient = np.unravel_index(score.argmax(), score.shape)
    tile = cat.patterns[short[row],orient]
    halves = []
    for axis in (0,1):
        for a,b in zip(np.split(gray,2,axis),np.split(tile,2,axis)):
            ac,bc = a-a.mean(),b-b.mean()
            halves.append(float((ac*bc).mean()/max(.01,ac.std()*bc.std())))
    if corr[row,orient]<.59 or min(halves)<.37 or score[row,orient]<.49:
        return None
    idx = short[row]
    return cat.ids[idx][0].removeprefix('minecraft:'),cat.labels[idx],float(corr[row,orient])


def sprite_candidates(part, rgb, linear):
    result = rectified(part, rgb, linear)
    if result is None:
        return []
    chart, mask, loc, H, factor = result
    if mask.sum() < 100:
        return []
    cat = atlas()
    shortlist, loss = _shortlist(chart, mask.astype(bool), cat)
    gray = chart.astype('f').mean(2)
    integral = cv2.integral(mask)
    winner = None
    # The photographed full sprite must fit inside the actual surface mask.
    # Correlate oriented atlas tiles across discrete absolute scales and phases.
    scales = [7, 9, 11, 13, 15, 18, 21, 25, 30, 36, 43, 52, 62]
    maximum = min(gray.shape)
    for size in scales:
        if size > maximum:
            continue
        support = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
        allowed = support >= .97 * size**2
        if not allowed.any():
            continue
        for key in shortlist[:8]:
            if loss[key] > 20 or cat.spreads[key].mean() < 2.5:
                continue
            for rotation in (0, 2, 4, 6, 1, 3, 5, 7) if part['axis'] == 1 else (0, 2, 4, 6):
                tile = cv2.resize(cat.patterns[key, rotation], (size, size), interpolation=cv2.INTER_LINEAR)
                corr = cv2.matchTemplate(gray, tile, cv2.TM_CCOEFF_NORMED)
                corr[~allowed] = -1
                _, peak, _, (x, y) = cv2.minMaxLoc(corr)
                quality = peak - .002 * loss[key]
                if quality > .59 and (winner is None or quality > winner[0]):
                    winner = quality, int(key), rotation, size, x, y, corr
    if winner is None:
        return []
    quality, key, rotation, size, x, y, response = winner
    # Subpixel scale and origin are refined against the same sprite with a local search.
    for fine in range(max(6, size-3), min(maximum, size+3)+1):
        support = integral[fine:, fine:] - integral[:-fine, fine:] - integral[fine:, :-fine] + integral[:-fine, :-fine]
        tile = cv2.resize(cat.patterns[key, rotation], (fine, fine), interpolation=cv2.INTER_LINEAR)
        corr = cv2.matchTemplate(gray, tile, cv2.TM_CCOEFF_NORMED)
        corr[support < .97*fine**2] = -1
        _, peak, _, (xx, yy) = cv2.minMaxLoc(corr)
        if peak - .002*loss[key] > quality:
            quality, size, x, y, response = peak-.002*loss[key], fine, xx, yy, corr
    out = []
    work = response.copy()
    for _ in range(12):
        _, peak, _, (x, y) = cv2.minMaxLoc(work)
        if peak < max(.65, quality-.14):
            break
        corners = np.array([x, y]) + UNIT * size
        # XYZ coordinates are relative to the optical center and measured in source-derived blocks.
        xyz = (np.column_stack((corners, np.ones(4))) @ loc.T) * factor / size
        quad = project(H, corners)
        verified = verify_sprite(rgb,quad)
        if verified:
            block,texture,native_corr = verified
            out.append(dict(surface=part['id'], axis=part['axis'], texture=texture, block=block,
                            corr=round(native_corr, 5), rectified_corr=round(float(peak),5),
                            xyz=xyz.tolist(), quad=quad.tolist(), pitch_px=size))
        radius = max(3, int(.78*size))
        work[max(0,y-radius):y+radius+1, max(0,x-radius):x+radius+1] = -1
    return out


def known_anchors(part, rgb, projection, eye, levels):
    from engine.atlas_plane_depth import _patch_scores
    axis = part['axis']
    free = [i for i in range(3) if i != axis]
    indices = np.linspace(0, part['area']-1, min(160, part['area']), dtype=int)
    xy = (np.column_stack((part['x'][indices], part['y'][indices])) + .5) * STRIDE
    r = rays(projection[:, :3], xy)
    winners = []
    for level in levels:
        dist = (level-eye[axis]) / r[:, axis]
        if (dist <= 1).any():
            continue
        uv = eye[free] + r[:, free]*dist[:, None]
        scores = _patch_scores(part['mask'], uv, axis, free, level, projection, rgb, atlas(), by_cell=True)
        for pos, score in (scores or {}).items():
            if score < .65:
                continue
            xyz = np.full((4, 3), float(level))
            xyz[:, free] = np.array(pos) + UNIT
            quad = project(projection,xyz)
            verified = verify_sprite(rgb,quad)
            if verified:
                block,texture,corr = verified
                winners.append(dict(surface=part['id'], axis=axis, xyz=(xyz-eye).tolist(),
                                    quad=quad.tolist(), block=block,texture=texture,corr=round(corr,5)))
    return winners


def consensus(anchors):
    if not anchors:
        return [], None, 0
    # Each full-face corner individually predicts the same XYZ translation modulo one.
    points = np.array([a['xyz'][0] for a in anchors])
    errors = abs((points[:, None] - points[None, :] + .5) % 1 - .5).max(2)
    distance = np.linalg.norm(points,axis=1)
    errors[np.maximum(distance[:,None],distance[None,:]) > 1.85*np.minimum(distance[:,None],distance[None,:])] = 1
    weights = np.array([a['corr'] for a in anchors])
    best = max(range(len(anchors)), key=lambda i: (len({anchors[j]['surface'] for j in np.flatnonzero(errors[i] <= .19)}),
                                                    weights[errors[i] <= .19].sum()))
    good = errors[best] <= .19
    centered = (points[good]-points[best]+.5)%1-.5
    phase = points[best] + np.median(centered, axis=0)
    return [a for a,b in zip(anchors, good) if b], -phase, int(good.sum())


def occupancy(parts, anchors, eye, linear, observe=None):
    """Attach perpendicular subfaces only across a photographed junction and cast supported rays."""
    groups = defaultdict(list)
    for anchor in anchors:
        groups[anchor['surface']].append(anchor)
    ids, owners = {}, defaultdict(set)
    witness = []
    for part in parts:
        group = groups.get(part['id'], [])
        if not group:
            continue
        axis = part['axis']
        for a in group:
            xyz = np.array(a['xyz']) + eye
            center = xyz.mean(0)
            center[axis] -= np.sign(eye[axis]-center[axis]) * .02
            cell = tuple(np.floor(center + 1e-7).astype(int).tolist())
            ids[cell] = a.get('block', FALLBACK)
            owners[cell].add(part['id'])
            if observe is not None:
                face_mask = np.zeros_like(part['mask'],dtype='B')
                cv2.fillConvexPoly(face_mask,np.rint(np.array(a['quad'])/STRIDE-.5).astype('i'),1)
                yy,xx = np.nonzero(face_mask & part['mask'])
                observe(part['id'],cell,axis,float(xyz[0,axis]),yy,xx)
        # Only seek neighbors near verified anchor faces, never extrude unobserved surfaces.
        cover = np.zeros_like(part['mask'], dtype=np.uint8)
        for a in group:
            cv2.fillConvexPoly(cover, np.rint(np.array(a['quad']) / STRIDE - .5).astype('i'), 1)
        halo = cv2.dilate(cover, np.ones((5, 5), dtype='B')).astype(bool)
        level = float(np.median([a['xyz'][0][axis] for a in group])) + eye[axis]
        for neighbor in parts:
            na = neighbor['axis']
            if na == axis:
                continue
            boundary = neighbor['mask'] & halo
            by, bx = np.nonzero(boundary)
            if len(bx) < 3:
                continue
            xy = (np.column_stack((bx, by))+.5)*STRIDE
            r = rays(linear, xy)
            wa = eye + r * ((level-eye[axis])/r[:, [axis]])
            measured = wa[:, na]
            if np.quantile(measured,.80)-np.quantile(measured,.20) > .32:
                continue
            # Shared-edge normal coordinate, with partial faces allowed on eighth-block levels.
            cut = round(float(np.median(measured))*8)/8
            near = cv2.dilate(cover, np.ones((15,15),dtype='B')).astype(bool) & neighbor['mask']
            y,x = np.nonzero(near)
            xy = (np.column_stack((x,y))+.5)*STRIDE
            r = rays(linear, xy)
            xyz = eye + r * ((cut-eye[na])/r[:,[na]])
            xyz[:,na] -= np.sign(eye[na]-cut)*.01
            if not np.isfinite(xyz).all():
                continue
            cells, votes = np.unique(np.floor(xyz).astype(int), axis=0, return_counts=True)
            for p, n in zip(cells, votes):
                if n < 4 or min(abs(p[axis] - level), abs(p[axis]+1-level)) > 1.01:
                    continue
                cell = tuple(p.tolist())
                ids.setdefault(cell, FALLBACK)
                owners[cell].add(neighbor['id'])
                if observe is not None:
                    keep = (np.floor(xyz).astype(int)==p).all(1)
                    observe(neighbor['id'],cell,na,cut,y[keep],x[keep])
            witness.append(dict(host=part['id'], neighbor=neighbor['id'], axis=na, level=cut,
                                support=int(len(bx)), samples=int(len(xy))))
    return ids, sorted([list(p) for p in ids if len(owners[p])>=2]), witness


def infer(image, source_parts, camera, offsets, segments, baseline_ids):
    rgb = np.asarray(image.convert('RGB'))
    known = bool(camera and camera.get('P') is not None)
    if known:
        P = np.asarray(camera['P'], dtype=float)
        P /= np.linalg.norm(P[2,:3])
        linear = P[:, :3]
        eye = np.linalg.solve(linear, -P[:,3])
    else:
        linear = line_camera(segments, image.size)
        if linear is None:
            return {}, None, dict(reason='no three independent line directions')
    parts = mask_parts(source_parts, image.size, linear)
    anchors = []
    if known:
        levels = {row['id']:row['level'] for row in offsets}
        cells = np.array(list(baseline_ids))
        for part in parts[:12]:
            axis = part['axis']
            if part['id'] in levels:
                choices = [levels[part['id']]]
            elif len(cells):
                values, counts = np.unique(cells[:,axis] + (eye[axis]>cells[:,axis]+.5), return_counts=True)
                choices = values[np.argsort(-counts)[:3]]
            else:
                continue
            anchors.extend(known_anchors(part, rgb, P, eye, choices))
        selected = anchors
    else:
        for part in parts[:8]:
            anchors.extend(sprite_candidates(part, rgb, linear))
        selected, eye, _ = consensus(anchors)
        if not selected:
            return {}, None, dict(reason='no complete sprite matched', scanned=min(8,len(parts)))
        P = np.column_stack((linear, -linear@eye))
    axes, surfaces = {a['axis'] for a in selected}, {a['surface'] for a in selected}
    ids, shared, witness = occupancy(parts, selected, eye, linear)
    reason = ('cross-plane anchor agreement' if len(axes)>=2 else
              'repeated single plane' if len(selected)>=2 else 'no independent confirmation')
    # Two separated matches on one surface establish the visible sheet. Separate axes
    # are required before modifying an already-supported ordinary build.
    accept = len(axes)>=2 if known else len(selected)>=2
    info = dict(reason=reason, accepted=accept, scanned=min(12 if known else 8,len(parts)),
                raw=len(anchors), confirmed=len(selected), axes=len(axes), surfaces=len(surfaces),
                shared=shared, junctions=witness, anchors=selected,
                extent=(np.ptp(np.array(list(ids)),axis=0)+1).tolist() if ids else [],
                eye=list(map(float,eye)), options=len(ids))
    return ids, P.tolist(), info
