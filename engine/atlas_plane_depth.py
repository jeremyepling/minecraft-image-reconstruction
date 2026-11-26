"""Select the integer depth of an observed surface from its projected texture."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from engine.camera_atlas_floor import _atlas
from engine.visible_placement import place_with_camera
from engine.visible_raster import SAMPLE_STRIDE
from engine.plane_offset_features import NAMES, ranked_choice


WEIGHTS = Path(__file__).with_name('plane_offset_weights.json')


@lru_cache(maxsize=2)
def _model(path, fingerprint):
    data = json.loads(path.read_bytes())
    weights = np.asarray(data['weights'], float)
    if (data['schema'] != 1 or data['version'] != '26.3' or tuple(data['features']) != NAMES
            or weights.shape != (len(NAMES),) or not np.isfinite(weights).all()):
        raise ValueError('Unsupported plane-offset model')
    return weights


def frozen_weights():
    try:
        stat = WEIGHTS.stat()
        return _model(WEIGHTS, (stat.st_mtime_ns,stat.st_size))
    except (OSError,ValueError,TypeError,KeyError):
        return None


def _sample(items, limit):
    if len(items) <= limit:
        return items
    return items[np.linspace(0, len(items)-1, limit, dtype=int)]


def _rays(x, y, inv):
    return x[:, None]*inv[:, 0] + y[:, None]*inv[:, 1] + inv[:, 2]


def _coefficients(rays, axis, free):
    return rays[:, free] / np.where(abs(rays[:, [axis]])>1.e-9, rays[:, [axis]], np.nan)


def _boundary(mask, inv, axis, free):
    # The interface between two neighboring samples lies midway between their centers.
    y,x = np.nonzero(mask[:, 1:] != mask[:, :-1])
    y2,x2 = np.nonzero(mask[1:] != mask[:-1])
    xy = np.column_stack((np.r_[x+1, x2+.5], np.r_[y+.5, y2+1]))*SAMPLE_STRIDE
    xy = _sample(xy, 450)
    return _coefficients(_rays(xy[:, 0], xy[:, 1], inv), axis, free)


def _texel_samples(mask, rgb, inv, axis, free):
    # Edge locations retain information even where adjacent blocks have different IDs.
    # Sample interior source pixels, using neighbors one native pixel away.
    indices = _sample(np.flatnonzero(mask), 800)
    y,x = np.unravel_index(indices, mask.shape)
    offsets = np.array([[0,0], [2,0], [1,1], [3,1], [0,2], [2,2], [1,3], [3,3]])
    xx = (SAMPLE_STRIDE*x[:, None]+offsets[:, 0]).ravel()
    yy = (SAMPLE_STRIDE*y[:, None]+offsets[:, 1]).ravel()
    width = min(rgb.shape[1], mask.shape[1] * SAMPLE_STRIDE)
    height = min(rgb.shape[0], mask.shape[0] * SAMPLE_STRIDE)
    ok = (xx+1<width) & (yy+1<height)
    xx, yy = xx[ok], yy[ok]
    ref = rgb[yy, xx]
    result = []
    for dx,dy in ((1,0),(0,1)):
        inside = mask[(yy+dy)//SAMPLE_STRIDE, (xx+dx)//SAMPLE_STRIDE]
        changed = np.any(ref[inside] != rgb[yy[inside]+dy, xx[inside]+dx], axis=-1)
        start = _rays(xx[inside]+.5, yy[inside]+.5, inv)
        end = start + dx*inv[:, 0] + dy*inv[:, 1]
        result.append((_coefficients(start, axis, free), _coefficients(end, axis, free), changed))
    return result


def _edge_score(samples, eye_uv, distance):
    correlations = []
    for a,b,observed in samples:
        valid = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        a,b,observed = a[valid],b[valid],observed[valid]
        if len(observed)<32 or not .001 < observed.mean() < .999:
            continue
        crossed = np.any(np.floor(16*(eye_uv+a*distance)) != np.floor(16*(eye_uv+b*distance)), axis=1)
        px, py = crossed.mean(), observed.mean()
        covariance = (crossed & observed).mean() - px*py
        correlations.append(covariance/np.sqrt(max(.0001,px*(1-px)*py*(1-py))))
    return float(np.mean(correlations)) if correlations else 0.


def _patch_scores(mask, origins, axis, free, level, P, rgb, atlas, *, by_cell=False):
    if atlas is None:
        return []
    origins = origins[np.isfinite(origins).all(1)]
    cells = np.unique(np.floor(origins).astype(int), axis=0)
    cells = _sample(cells, 120 if by_cell else 30)
    if not len(cells):
        return []
    u,v = np.meshgrid((np.arange(16)+.5)/16, (np.arange(16)+.5)/16)
    xyz = np.full((len(cells),256,3),float(level))
    xyz[:,:,free[0]] = cells[:,0,None]+u.ravel()
    xyz[:,:,free[1]] = cells[:,1,None]+v.ravel()
    screen = xyz @ P[:,:3].T + P[:,3]
    xy = screen[:,:,:2] / np.maximum(1.e-6,screen[:,:,2:])
    h,w = rgb.shape[:2]
    h,w = min(h,mask.shape[0]*SAMPLE_STRIDE), min(w,mask.shape[1]*SAMPLE_STRIDE)
    x,y = np.moveaxis(np.floor(xy).astype(int),-1,0)
    good = ((x>=0)&(y>=0)&(x<w)&(y<h)&(screen[:,:,2]>1.)).all(1)
    good &= (np.linalg.norm(xy[:,0]-xy[:,15],axis=1)>=9) & (np.linalg.norm(xy[:,0]-xy[:,240],axis=1)>=9)
    x,y = x[good],y[good]
    full = mask[y//SAMPLE_STRIDE,x//SAMPLE_STRIDE].mean(1)>=.97
    x,y = x[full],y[full]
    if not len(x):
        return []
    patches = rgb[y,x].astype(np.float32)
    color = patches.mean(1)
    gain = np.clip((color @ atlas.means.T)/np.maximum(1,(atlas.means**2).sum(1)),.30,1.2)
    loss = abs(color[:,None]-gain[:,:,None]*atlas.means).mean(2)
    loss += 3*abs(np.log(gain/.80))
    short = np.argpartition(loss,24,axis=1)[:,:24]
    gray = patches.mean(2)
    gray -= gray.mean(1,keepdims=True)
    gray /= np.maximum(4.,np.linalg.norm(gray,axis=1,keepdims=True))
    ref = atlas.patterns[short].reshape(len(x),24,8,256)
    corr = np.einsum('nkod,nd->nko',ref,gray).max(2)/16
    scores = (corr-np.take_along_axis(loss,short,axis=1)/60).max(1)
    if by_cell:
        return dict(zip(map(tuple,cells[good][full].tolist()),scores.tolist(),strict=True))
    return np.sort(scores)[::-1].tolist()


def options(mask, P, eye, rays, axis, baseline, rgb, atlas, scene_distance=None, radius=6, *, extra_levels=()):
    free = [i for i in range(3) if i != axis]
    inv = np.linalg.inv(P[:,:3])
    boundary = _boundary(mask,inv,axis,free)
    samples = _texel_samples(mask,rgb,inv,axis,free)
    anchor_rays = _sample(rays[mask],180)
    anchors = _coefficients(anchor_rays,axis,free)
    boundary = boundary[np.isfinite(boundary).all(1)]
    uv = eye[free]
    rows = []
    levels=set(range(baseline-radius,baseline+radius+1))
    levels.update(extra_levels)
    scene_level=baseline
    if scene_distance is not None:
        scene_level=round(float(eye[axis]+np.median(rays[mask,axis])*scene_distance))
        levels.update(range(scene_level-radius,scene_level+radius+1))
    for level in sorted(levels):
        distance = level-eye[axis]
        if (baseline-eye[axis])*distance <= 0 or (axis==1 and level-int(distance<0)<0):
            continue
        delta = abs((uv+boundary*distance+.5)%1-.5).min(1)
        boundary_score = float(np.maximum(0,1-delta/.18).mean()) if len(delta) else 0.
        scores = _patch_scores(mask,uv+anchors*distance,axis,free,level,P,rgb,atlas)
        rows.append(dict(level=level,boundary=boundary_score,edge=_edge_score(samples,uv,distance),
                         atlas=float(np.mean(scores[:5])) if scores else 0.,tiles=len(scores),
                         # Votes count distinct observed cell locations; materials may repeat.
                         distinct_votes=sum(v>.55 for v in scores),scene_delta=abs(level-scene_level)))
    return rows



def place_ranked(image, version, parts, counts, camera, *, atlas_path=None, joint_foreground=True):
    """Place image-supported grid cells and remove cells with no visible ray."""
    from engine.foreground_conflicts import ForegroundConflicts
    cells, status = _place(image, version, parts, counts, camera, atlas_path, remove_occluded=True,
                           scene_policy=ForegroundConflicts() if joint_foreground else None)
    if cells and status['foreground_depth']['method'] == 'joint':
        from engine import neighbor_photo_faces, visible_horizontal_faces
        textured = cells | neighbor_photo_faces.recover(image, parts, camera['P'], cells)
        recovered = visible_horizontal_faces.recover(image, parts, camera, textured)
        status['neighbor_photo_blocks'] = len(textured) - len(cells)
        status['dense_horizontal_blocks'] = len(recovered) - len(textured)
        status['added_blocks'] += len(recovered) - len(cells)
        status['placed_blocks'] = len(recovered)
        cells = recovered
    return cells, status


def place_ranked_without_ray(image, version, parts, counts, camera, *, atlas_path=None, joint_foreground=True):
    """Run the same placement before the final native-pixel visibility check."""
    from engine.foreground_conflicts import ForegroundConflicts
    return _place(image, version, parts, counts, camera, atlas_path, remove_occluded=False,
                  scene_policy=ForegroundConflicts() if joint_foreground else None)


def _place(image, version, parts, counts, camera, atlas_path, *, remove_occluded, recover_edges=True, scene_policy=None):
    if camera['P'] is None or not parts:
        cells, status = place_with_camera(image, version, parts, counts, camera)
        status['foreground_depth'] = dict(method='unavailable', changed_planes=0, reason='no supported camera or surfaces')
        return cells, status
    weights = frozen_weights()
    atlas = _atlas(version, atlas_path) if weights is not None else None
    if atlas is None:
        cells, status = place_with_camera(image, version, parts, counts, camera)
        status['plane_depth_reason'] = ('trained plane model unavailable' if weights is None
                                        else 'calibrated atlas unavailable')
        status['foreground_depth'] = dict(method='unavailable', changed_planes=0, reason=status['plane_depth_reason'])
        return cells, status
    rgb = np.asarray(image.convert('RGB'))
    decisions = []
    from engine.junction_depth import JunctionDepth
    from engine.plane_repeat_evidence import frozen_repeat_weights, scored_rows, select_repeat
    repeat_weights = frozen_repeat_weights()
    junction = JunctionDepth(image.size, parts) if repeat_weights is not None else None

    def select(part, mask, projection, eye, rays, axis, baseline, scene_distance):
        if junction is None:
            rows = options(mask, projection, eye, rays, axis, baseline, rgb, atlas, scene_distance)
            level = ranked_choice(rows, baseline, weights)
        else:
            extra = junction.levels(mask, projection, eye, rays, axis)
            rows = scored_rows(mask, projection, eye, rays, axis, baseline, rgb, atlas, scene_distance,
                               extra_levels=extra)
            level = select_repeat(rows, baseline, repeat_weights)
            level = junction.choose(part, mask, projection, eye, rays, axis, baseline, scene_distance,
                                    rows, weights, chosen_before=level)
            junction.record(part, mask, axis, level, rows)
        if scene_policy is not None and junction is not None:
            scene_policy.record(part['id'], mask, eye, rays, axis, baseline, rows, level, repeat_weights)
        decisions.append(dict(id=part['id'], axis=axis, baseline=baseline, level=level))
        return level

    from engine.partial_atlas_face import verify
    from engine.secondary_plane_depth import wrap_selector
    select_level, refine = wrap_selector(select, rgb, atlas, final_levels=scene_policy.levels if scene_policy else None)

    foreground_status = dict(method='independent' if scene_policy is None else 'joint', changed_planes=0)
    if junction is None:
        foreground_status.update(method='unavailable', reason='repeat plane model unavailable')

    def reconcile(selections):
        try:
            levels = scene_policy.resolve()
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as err:
            scene_policy.levels.clear()
            foreground_status.update(method='independent', reason=f'foreground evidence unusable: {err}')
            return selections
        foreground_status['changed_planes'] = sum(levels.get(row['id'], row['level']) != row['level']
                                                  for row in decisions)
        for row in decisions:
            row['level'] = levels.get(row['id'], row['level'])
        return [(i, a, levels.get(parts[i]['id'], l)) for i, a, l in selections]

    def partial(mask, projection, cell, axis, sign):
        return verify(rgb, atlas, mask, projection, cell, axis, sign)

    def recover(cells):
        from engine.visible_edge_faces import extend
        return extend(image, parts, camera['P'], cells, decisions, atlas)

    cells, status = place_with_camera(image, version, parts, counts, camera, select_level,
                                      partial=partial, refine_candidates=refine, remove_occluded=remove_occluded,
                                      walls_first=junction is not None, reconcile=reconcile if scene_policy and junction is not None else None,
                                      recover_edges=recover if recover_edges and junction is not None else None)
    status.update(plane_depth_method='ranked', plane_offsets=decisions, foreground_depth=foreground_status,
                  plane_depth_search='texture-and-contacts' if junction is not None else 'original-window')
    return cells, status
