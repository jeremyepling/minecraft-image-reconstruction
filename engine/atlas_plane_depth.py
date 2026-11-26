"""Select the integer depth of an observed surface from its projected texture."""
import numpy as np

from engine.camera_atlas_floor import _atlas
from engine.visible_placement import place_with_camera
from engine.visible_raster import SAMPLE_STRIDE
from engine.plane_offset_features import ranked_choice


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
    ok = (xx+1<rgb.shape[1]) & (yy+1<rgb.shape[0])
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


def _patch_scores(mask, origins, axis, free, level, P, rgb, atlas):
    if atlas is None:
        return []
    origins = origins[np.isfinite(origins).all(1)]
    cells = np.unique(np.floor(origins).astype(int), axis=0)
    cells = _sample(cells, 30)
    if not len(cells):
        return []
    u,v = np.meshgrid((np.arange(16)+.5)/16, (np.arange(16)+.5)/16)
    xyz = np.full((len(cells),256,3),float(level))
    xyz[:,:,free[0]] = cells[:,0,None]+u.ravel()
    xyz[:,:,free[1]] = cells[:,1,None]+v.ravel()
    screen = xyz @ P[:,:3].T + P[:,3]
    xy = screen[:,:,:2] / np.maximum(1.e-6,screen[:,:,2:])
    h,w = rgb.shape[:2]
    x,y = np.moveaxis(xy.astype(int),-1,0)
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
    return np.sort(scores)[::-1].tolist()


def options(mask, P, eye, rays, axis, baseline, rgb, atlas, scene_distance=None, radius=6):
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


def choose(rows, baseline, method):
    if not rows or all(r['level']!=baseline for r in rows):
        return baseline
    if method=='combined':
        def confidence(r):
            texture = min(r['tiles']/3,1)*r['atlas']
            return .53*texture+.30*r['edge']+.17*r['boundary']
        selected = max(rows,key=lambda r:confidence(r)-.004*min(abs(r['level']-baseline),8))
        before = next(r for r in rows if r['level']==baseline)
        return selected['level'] if confidence(selected)>.38 and confidence(selected)-confidence(before)>.07 else baseline
    key = {'boundary':'boundary','edge':'edge','atlas':'atlas'}[method]
    usable = [r for r in rows if key!='atlas' or r['tiles']>=2]
    if not usable:
        return baseline
    winner = max(usable, key=lambda r:r[key]-.006*abs(r['level']-baseline))
    base = next(r for r in rows if r['level']==baseline)
    if winner[key]-base[key] < {'boundary':.06,'edge':.10,'atlas':.12}[method]:
        return baseline
    return winner['level'] if winner[key] >= {'boundary':.52,'edge':.28,'atlas':.57}[method] else baseline


def place(image,version,parts,counts,camera,method,atlas_path=None,weights=None):
    if camera['P'] is None or not parts:
        return place_with_camera(image,version,parts,counts,camera)
    atlas = _atlas(version,atlas_path) if method in ('atlas','combined','ranked') else None
    if method in ('atlas','combined','ranked') and atlas is None:
        cells,status=place_with_camera(image,version,parts,counts,camera)
        status['plane_depth_reason']='calibrated atlas unavailable'
        return cells,status
    rgb = np.asarray(image.convert('RGB'))
    decisions = []
    def select(part,mask,P,eye,rays,axis,baseline,scene_distance):
        rows = options(mask,P,eye,rays,axis,baseline,rgb,atlas,
                       scene_distance if method in ('combined','ranked') else None)
        level=(ranked_choice(rows,baseline,weights) if method=='ranked' else choose(rows,baseline,method))
        decisions.append(dict(id=part['id'],axis=axis,baseline=baseline,level=level))
        return level
    cells,status=place_with_camera(image,version,parts,counts,camera,select)
    status.update(plane_depth_method=method,plane_offsets=decisions)
    return cells,status


def predict(image,version,parts,counts,camera):
    return place(image,version,parts,counts,camera,'boundary')


def atlas_only(image,version,parts,counts,camera):
    return place(image,version,parts,counts,camera,'atlas')


def edge_only(image,version,parts,counts,camera):
    return place(image,version,parts,counts,camera,'edge')


def combined(image,version,parts,counts,camera):
    return place(image,version,parts,counts,camera,'combined')
