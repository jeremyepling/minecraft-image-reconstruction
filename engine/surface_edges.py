"""Fit visible surface patches from connected color boundaries and their junctions."""
from collections import defaultdict
from itertools import combinations

import numpy as np
from PIL import Image, ImageFilter

from engine.camera_lines import _outline
from engine.geometry import bilinear
from engine.plane_candidates import foreground
from engine.structural_lines import detect


SETTINGS = dict(analysis_width=580, blur_px=1.4, region_scale=185., min_region_px=18,
                min_face_purity=.80, min_quad_area=42., max_patches=65)


def _components(rgb):
    """Segment a four-neighbor edge graph using each component's internal contrast."""
    h, w = rgb.shape[:2]
    pixels = np.arange(w*h).reshape(h, w)
    a = np.r_[pixels[:, :-1].ravel(), pixels[:-1].ravel()]
    b = np.r_[pixels[:, 1:].ravel(), pixels[1:].ravel()]
    flat = rgb.reshape(-1, 3).astype(np.int16)
    costs = np.max(np.abs(flat[a]-flat[b]), axis=1)
    order = np.argsort(costs, kind='stable')
    edges = list(zip(a[order].tolist(), b[order].tolist(), costs[order].tolist()))
    size = [1]*len(flat)
    parent = list(range(len(flat)))
    k = SETTINGS['region_scale']
    allowance = [k]*len(flat)

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for small in (False, True):
        for a, b, cost in edges:
            a, b = root(a), root(b)
            if a == b:
                continue
            if small:
                if min(size[a], size[b]) >= SETTINGS['min_region_px']:
                    continue
            elif cost > min(allowance[a], allowance[b]):
                continue
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            allowance[a] = cost+k/size[a]
    return np.array([root(i) for i in range(len(flat))]).reshape(h, w)


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _order(quad):
    """Clockwise perimeter, starting with the left end of the upper cross edge."""
    v = np.roll(quad, -1, axis=0)-quad
    if np.min(_cross(v, np.roll(v, -1, axis=0))) <= 0:
        quad = quad[::-1]
    ends = np.roll(quad, -1, axis=0)
    # For the two pairs of opposite sides, put the more nearly horizontal
    # pair at top and bottom. This also covers sloping roof faces.
    v = ends-quad
    horizontal = np.abs(v[:, 0])/np.maximum(.01, np.linalg.norm(v, axis=1))
    a = int(horizontal[1::2].mean() > horizontal[::2].mean())
    y = quad[:, 1]+ends[:, 1]
    i = a if y[a] < y[a+2] else a+2
    return np.roll(quad, -i, axis=0)


def _intersections(lines):
    meet = np.cross(lines, np.roll(lines, 1, axis=0))
    if np.min(np.abs(meet[:, 2])) < .025:
        return None
    return meet[:, :2]/meet[:, 2, None]


def _fit_boundary(points, quad):
    # Use the long measured part of each boundary. A nick in an outer corner
    # should not steer an otherwise straight border.
    edges = np.roll(quad, -1, axis=0)-quad
    length = np.linalg.norm(edges, axis=1)
    if min(length) < 3:
        return None
    directions = edges/length[:, None]
    offset = points[:, None, :]-quad
    lengths = (offset*directions).sum(2)
    distances = np.abs(_cross(offset, directions))
    distances[(lengths < 0) | (lengths > length)] = np.inf
    sides = distances.argmin(1)
    lines, coverage = [], []
    for i, direction in enumerate(directions):
        near = (sides == i) & (distances[:, i] <= max(2., .045*length[i]))
        samples = points[near]
        normal = np.array([-direction[1], direction[0]])
        center = (quad[i]+quad[(i+1)%4])/2
        if len(samples) >= 5:
            mid = np.median(samples, axis=0)
            delta = samples-mid
            residual = delta@normal
            tight = np.abs(residual-np.median(residual)) < max(1.25, 2.5*np.median(np.abs(residual-np.median(residual))))
            if tight.sum() >= 5:
                center = samples[tight].mean(0)
                _, vectors = np.linalg.eigh((samples[tight]-center).T@(samples[tight]-center))
                fitted = vectors[:, 0]
                if abs(fitted@normal) > .95:
                    normal = fitted
        lines.append([*normal, -(normal@center)])
        bins = np.unique(np.clip((12*lengths[near, i]/length[i]).astype(int), 0, 11))
        coverage.append(len(bins)/12)
    adjusted = _intersections(np.asarray(lines))
    if adjusted is None or np.max(np.linalg.norm(adjusted-quad, axis=1)) > max(5, length.min()*.20):
        adjusted = quad
    return adjusted, coverage


def _region_outlines(component, mask):
    """Retain outer line segments from concave regions as well as intact quadrilaterals."""
    h, w = mask.shape
    # Each boundary vertex uses the half-pixel position between different
    # labels. Both regions can therefore vote for the same shared edge.
    boundary = defaultdict(list)
    for ax in (0, 1):
        i = (slice(None, -1), slice(None)) if ax == 0 else (slice(None), slice(None, -1))
        j = (slice(1, None), slice(None)) if ax == 0 else (slice(None), slice(1, None))
        y, x = np.nonzero(component[i] != component[j])
        left, right = component[i][y, x], component[j][y, x]
        px = 2*x+1+(ax == 1)
        py = 2*y+1+(ax == 0)
        for a, b, xx, yy in zip(left.tolist(), right.tolist(), px.tolist(), py.tolist()):
            boundary[a].append((xx, yy))
            boundary[b].append((xx, yy))
    touching = set(np.r_[component[0],component[-1],component[:,0],component[:,-1]].tolist())
    areas = np.bincount(component.ravel())
    foreground_area = np.bincount(component.ravel(), weights=mask.ravel(), minlength=len(areas))
    patches, segments = [], []
    for region, raw in boundary.items():
        if region in touching or areas[region] < SETTINGS['min_quad_area'] or foreground_area[region] < .4*areas[region]:
            continue
        quad = _outline(raw)/2
        if len(quad) != 4:
            continue
        pts = np.asarray(raw)/2
        fit = _fit_boundary(pts, quad)
        if fit is None:
            continue
        quad, support = fit
        if min(support) > .38:
            patches.append((quad, min(support), 'region boundary'))
        for a, b, score in zip(quad, np.roll(quad, -1, axis=0), support):
            if score > .66 and np.linalg.norm(a-b) > 7:
                segments.append((a,b,score))
    return patches, segments


def _cycles(segments):
    """Join four observed segments at L or T junctions into a projective rectangle."""
    if len(segments)<4:
        return
    data = sorted(segments, key=lambda s:-np.linalg.norm(s[1]-s[0])*s[2])[:100]
    ends = np.array([[a,b] for a,b,_ in data])
    midpoint = ends.mean(1)
    vectors = ends[:,1]-ends[:,0]
    length = np.linalg.norm(vectors,axis=1)
    unit = vectors/length[:,None]
    nn = np.stack([-unit[:,1],unit[:,0]],axis=1)
    lines = np.c_[nn,-(nn*midpoint).sum(1)]
    meet = []
    for i in range(len(ends)):
        pt = np.cross(lines[i],lines)
        good = abs(pt[:,2]) > .14
        xy = pt[:,:2]/np.where(good,pt[:,2],1.)[:,None]
        dist_i = abs((xy-midpoint[i])@unit[i]) - .5*length[i]
        dist_j = abs(((xy-midpoint)*unit).sum(1)) - .5*length
        pad = np.clip(.15*np.minimum(length[i],length),2.,12.)
        meet.append({int(j):xy[j] for j in np.flatnonzero(good & (dist_i < pad) & (dist_j < pad))})
    for i,j in combinations(range(len(ends)),2):
        if abs(unit[i] @ unit[j]) < .94:
            continue
        for a,b in combinations(sorted(meet[i].keys() & meet[j].keys()),2):
            if min(a,b) < i or abs(unit[a] @ unit[b]) < .94:
                continue
            quad=np.array([meet[i][a],meet[i][b],meet[j][b],meet[j][a]])
            edges=np.roll(quad,-1,axis=0)-quad
            turn=_cross(edges,np.roll(edges,-1,axis=0))
            if not (np.all(turn > .8) or np.all(turn < -.8)):
                continue
            sides=np.linalg.norm(edges,axis=1)
            if min(sides) < 3.2:
                continue
            # For each edge, require its observed segment to reach both corners.
            support=np.minimum(1., length[[i,b,j,a]]/sides)
            if min(support) >= .72:
                yield quad, float(min(support)), 'four joined edges'


def _subject_lines(small, mask):
    """Trace long structural edges across region splits caused by texture or lighting."""
    yy, xx = np.nonzero(mask)
    x0, y0 = max(0,xx.min()-8), max(0,yy.min()-8)
    box = (x0,y0,min(small.width,xx.max()+9),min(small.height,yy.max()+9))
    found = []
    for line in detect(small.crop(box)):
        a, b = np.array([line['p'],line['q']])+[x0,y0]
        if line['length'] < 7:
            continue
        d = b-a
        # The main mask can end a few pixels inside the silhouette.
        mid = (a+b)/2
        n = np.array([-d[1],d[0]])*3/line['length']
        points = np.clip(np.array([mid, mid+n, mid-n]).astype(int), 0, np.array(small.size)-1)
        if np.any(mask[points[:,1],points[:,0]]):
            found.append((a,b,line['support']))
    return found


def _sample_grid(quad):
    axis = (np.arange(13)+.5)/13
    top = quad[0]+axis[:,None]*(quad[1]-quad[0])
    bottom = quad[3]+axis[:,None]*(quad[2]-quad[3])
    return top+axis[:,None,None]*(bottom-top)


def _score(quad, edge_support, labels, mask, colors):
    edges = np.roll(quad,-1,axis=0)-quad
    corners = _cross(edges,np.roll(edges,-1,axis=0))
    lengths = np.linalg.norm(edges,axis=1)
    area = .5*abs(_cross(quad,np.roll(quad,-1,axis=0)).sum())
    if area < SETTINGS['min_quad_area'] or np.min(corners)<.5 or min(lengths)<3.2:
        return None
    unit = edges/lengths[:,None]
    if min(-unit[0]@unit[2], -unit[1]@unit[3]) < .91:
        return None
    xy = _sample_grid(quad)
    h,w = mask.shape
    if np.any(xy<0) or np.any(xy >= [w,h]):
        return None
    x,y=xy.astype(int).transpose(2,0,1)
    observed=mask[y,x]
    windows=np.lib.stride_tricks.sliding_window_view(observed,(3,3))
    if observed.mean()<.87 or not windows.any(axis=(-1,-2)).all():
        return None
    ids, counts = np.unique(labels[y,x],return_counts=True)
    region = labels[y,x] == ids[counts.argmax()]
    # A face split into multiple weak/texture fragments can still be supported
    # by four observed structural edges. Enforce appearance consistency across
    # the patch with a robust color/spatial coverage check.
    rgb=colors[y,x]
    similar=np.max(abs(rgb-np.median(rgb.reshape(-1,3),axis=0)),axis=2)<14.
    inside=region if region.mean()>similar.mean() else similar
    purity = float(inside.mean())
    if purity<SETTINGS['min_face_purity'] or min(inside.mean(0).min(),inside.mean(1).min()) < .43:
        return None
    return .58*purity+.42*edge_support, area, purity


def _lighting_step(quad, field):
    # Sustained luminance steps are extra evidence of a plane turn. Remove
    # contrast from a continuous lighting ramp sampled on the same side.
    edges = np.roll(quad,-1,axis=0)-quad
    lengths = np.linalg.norm(edges,axis=1)
    unit = edges/lengths[:,None]
    t = np.linspace(.14,.86,17)
    pts = quad[:,None,:]+t[None,:,None]*edges[:,None,:]
    normal = np.stack([-unit[:,1],unit[:,0]],axis=1)[:,None,:]
    radius = np.clip(.10*np.roll(lengths,1),1.3,4.)[:,None,None]
    offset = radius*normal
    shade = [bilinear(field,pts+k*offset)[...,0] for k in (-2,-1,1,2)]
    across = shade[2]-shade[1]-(shade[3]-shade[2]+shade[1]-shade[0])
    # Compare medians along all four sides, limiting a single sky silhouette.
    return float(np.sort(abs(np.median(across,axis=1)))[1:3].mean())


def _corner_junctions(patches, tolerance):
    """Record nearby image corners without assuming that they coincide in depth."""
    for p in patches:
        p['corner_junctions'] = []
    for a,b in combinations(patches,2):
        delta = np.array(a['quad'])[:,None,:] - np.array(b['quad'])[None,:,:]
        distances = np.linalg.norm(delta,axis=2)
        for i,j in np.argwhere(distances <= tolerance):
            score = round(float(distances[i,j]),2)
            a['corner_junctions'].append(dict(corner=int(i),neighbor=b['id'],neighbor_corner=int(j),distance_px=score))
            b['corner_junctions'].append(dict(corner=int(j),neighbor=a['id'],neighbor_corner=int(i),distance_px=score))


def detect_planes(image, version='26.3', shading=False):
    """Return clockwise native-pixel quads using only the supplied image."""
    if version != '26.3':
        raise ValueError('Visible surface estimator targets Java 26.3')
    image = image.convert('RGB')
    factor = min(1., SETTINGS['analysis_width']/max(image.size))
    size = tuple(max(1,round(v*factor)) for v in image.size)
    if min(size)<12:
        return []
    small = image.resize(size,Image.Resampling.BOX).filter(ImageFilter.GaussianBlur(SETTINGS['blur_px']))
    rgb = np.asarray(small)
    _, raw_mask = foreground(image)
    mask = np.asarray(Image.fromarray(raw_mask).resize(size,Image.Resampling.NEAREST))
    if mask.mean()>.92 or not mask.any():
        return []
    components = _components(rgb)
    candidates, segments = _region_outlines(components,mask)
    candidates.extend(_cycles(_subject_lines(small,mask)+segments))
    field = np.asarray(small.convert('L'),dtype=float)[:,:,None] if shading else None
    colors = np.asarray(small.filter(ImageFilter.GaussianBlur(2.)),dtype=float)
    scale = np.array(image.size)/size
    scored = []
    for index, (quad, edge_support, source) in enumerate(candidates):
        quad = _order(quad)
        info = _score(quad, edge_support, components, mask, colors)
        if info is None:
            continue
        geometry, area, purity = info
        # Candidate geometry is shared across the ablation. Shading tests if
        # broad steps persist along the border after suppressing a local ramp.
        shade = _lighting_step(quad, field) if shading else None
        confidence = .78*geometry + .22*np.clip(shade/12.,0.,1.) if shading else geometry
        if confidence < .77:
            continue
        scored.append((confidence*np.sqrt(area),index,quad,confidence,edge_support,purity,shade,source))
    kept, occupied = [], np.zeros(mask.shape,bool)
    for _, index, quad, confidence, support, purity, shade, source in sorted(scored,key=lambda p:-p[0]):
        lo = np.maximum(0,np.floor(quad.min(0)).astype(int))
        hi = np.minimum(size,np.ceil(quad.max(0)).astype(int))
        x,y = np.meshgrid(np.arange(lo[0],hi[0])+.5,np.arange(lo[1],hi[1])+.5)
        pts = np.stack((x,y),axis=2)
        edges = np.roll(quad,-1,axis=0)-quad
        inside = np.min(_cross(edges,pts[:,:,None,:]-quad),axis=2)>=0
        local = occupied[lo[1]:hi[1],lo[0]:hi[0]]
        if inside.sum()<1 or np.count_nonzero(inside&local) > .20*inside.sum():
            continue
        local |= inside
        coords = np.round(quad*scale,2)
        evidence = dict(source=source, boundary_support=round(float(support),4),
                        region_purity=round(float(purity),4), shading_used=bool(shading))
        if shading:
            evidence['median_luminance_step']=round(float(shade),3)
        kept.append(dict(id=f'surface-{index}',
                         quad=coords.tolist(),confidence=round(float(confidence),4),evidence=evidence))
        if len(kept) >= SETTINGS['max_patches']:
            break
    _corner_junctions(kept,2.0*scale.mean())
    return kept
