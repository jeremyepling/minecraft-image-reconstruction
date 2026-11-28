"""Challenge dense Floating wall rulers with complete atlas faces in disjoint source regions."""
from collections import Counter

import cv2
import numpy as np

from benchmark.expanded_wall_rulers import resample
from benchmark.expanded_wall_source_resample import preserve_and_rank
from benchmark.floating_face_proposals import rectifications
from benchmark.floating_package_rechecks import border_filter, native_mask
from benchmark.full_block_lattice import atlas, verify_sprite
from benchmark.relative_oracle_charts import fit_homography
from engine.atlas_count import _shortlist
from engine.geometry import UNIT, project

MIN_CELLS = 256


def chart_mask(chart, plane, size):
    """Respect native support, depth partitions and the old sky veto when refitting."""
    mask = native_mask(plane, size)
    keep = (chart.get('export_uv') if 'export_uv' in chart else
            [(c['u'], c['v']) for c in chart['cells']])
    allowed = np.array(keep, dtype='int64').reshape(-1, 2)
    result = np.zeros_like(mask)
    if not len(allowed):
        return result
    y, x = np.nonzero(mask)
    grid = np.floor(project(np.linalg.inv(chart['H']), np.column_stack((x, y)) + .5)).astype('int64')
    lower = allowed.min(0)
    span = allowed.max(0) - lower + 1
    inside = ((grid >= lower) & (grid < lower + span)).all(1)
    encode = lambda points: (points[:, 0] - lower[0]) * span[1] + points[:, 1] - lower[1]
    accepted = inside & np.isin(encode(grid), encode(allowed))
    result[y[accepted], x[accepted]] = True
    return result


def views(rgb, mask, chart, axes):
    """Use the incumbent vanishing directions and independent straight-edge/normal estimates."""
    ys, xs = np.nonzero(mask[::4, ::4])
    points = np.column_stack((xs, ys)) * 4 + .5
    initial = np.array(chart['H'])
    proposals = [('incumbent', initial @ np.diag([1., -1., 1.]))]
    proposals.extend((name, H) for name, _, _, H in rectifications(rgb, mask, chart['normal_camera'], axes))
    for name, H in proposals:
        uv = project(np.linalg.inv(H), points)
        low, high = uv.min(0), uv.max(0)
        span = high - low
        if not np.isfinite(span).all() or min(span) <= 0:
            continue
        scale = 230 / max(span)
        size = np.ceil(span * scale + 4).astype(int)
        if min(size) < 18:
            continue
        R = H @ np.array([[1/scale, 0, low[0]-2/scale], [0, 1/scale, low[1]-2/scale], [0, 0, 1]])
        yy, xx = np.indices(size[::-1])
        xy = project(R, np.stack((xx+.5, yy+.5), -1))
        coords = (xy - .5).astype('f')
        patch = cv2.remap(rgb, coords, None, cv2.INTER_LINEAR)
        valid = cv2.remap(mask.astype('B'), coords, None, cv2.INTER_NEAREST).astype(bool)
        yield name, patch, valid, R, xy


def palette(patch, mask):
    library = atlas()
    votes = Counter()
    for ys in np.array_split(np.arange(mask.shape[0]), 3):
        for xs in np.array_split(np.arange(mask.shape[1]), 3):
            idx = np.ix_(ys, xs)
            selected = mask[idx]
            if selected.sum() < 24:
                continue
            short, loss = _shortlist(patch[idx], selected, library)
            for rank, k in enumerate(short[:12]):
                if loss[k] < 22 and library.spreads[k].mean() >= 2:
                    votes[int(k)] += 1/(rank+1)
    return [k for k, _ in votes.most_common(10)]


def pitch_ratio(proposed, original, center):
    lengths = []
    for matrix in (proposed, original):
        uv = project(np.linalg.inv(matrix), center)
        sides = project(matrix, uv + np.eye(2)) - center
        lengths.append(abs(np.linalg.det(sides)))
    return (lengths[0]/max(1e-8, lengths[1]))**.5


def discover(patch, mask, R, original, center):
    """Search only the discovery region. Reserve the rest for phased, fixed-ruler validation."""
    gray = patch.astype('f').mean(2)
    library = atlas()
    keys = palette(patch, mask)
    occupancy = cv2.integral(mask.astype('B'))
    offers = []
    for width in (6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 23, 26, 30, 35, 40, 46, 54, 64):
        if width >= min(gray.shape):
            continue
        quad_matrix = R @ np.diag([width, width, 1.])
        ratio = pitch_ratio(quad_matrix, original, center)
        if not 1.45 <= ratio <= 22:
            continue
        s = width
        keep = occupancy[s:,s:] - occupancy[:-s,s:] - occupancy[s:,:-s] + occupancy[:-s,:-s] >= .985*s*s
        if not keep.any():
            continue
        # Restrict correlations to the discovery rectangle before ranking.
        y, x = np.nonzero(keep)
        x0, x1, y0, y1 = x.min(), x.max()+1, y.min(), y.max()+1
        local = np.ascontiguousarray(gray[y0:y1+s-1, x0:x1+s-1])
        valid = keep[y0:y1, x0:x1]
        for k in keys:
            best = None
            for orient in range(8):
                template = cv2.resize(library.patterns[k,orient], (s,s), interpolation=cv2.INTER_LINEAR)
                response = cv2.matchTemplate(local, template, cv2.TM_CCOEFF_NORMED)
                response[~valid] = -2
                _, peak, _, (x,y) = cv2.minMaxLoc(response)
                if peak >= .52 and (best is None or peak > best[0]):
                    H = R @ np.array([[s,0,x+x0],[0,s,y+y0],[0,0,1.]])
                    best = peak, H, ratio
            if best:
                offers.append(best)
    return sorted(offers, key=lambda row: row[0], reverse=True)[:10]


def independent_coverage(old, new, anchors, mask, rgb):
    """Compare source-only atlas matches at common visible pixels outside the two discovery tiles."""
    keep = mask.astype('B')
    for quad in anchors:
        cv2.fillConvexPoly(keep, np.rint(quad).astype('int32'), 0)
    y, x = np.nonzero(keep[::4,::4])
    if not len(x):
        return dict(accept=False, probes=0)
    points = np.column_stack((x,y))*4+.5
    # One foreground pixel at a fixed quantile in each occupied source-image bin.
    xmin,ymin = points.min(0)
    span = np.maximum(1, points.max(0)-points.min(0))
    regions = np.minimum(9, np.floor((points-[xmin,ymin])*10/span).astype(int))
    probes = []
    for cell in np.unique(regions,axis=0):
        chunk=points[(regions==cell).all(1)]
        center=chunk.mean(0)
        probes.append(chunk[np.argmin(np.linalg.norm(chunk-center,axis=1))])
    probes=np.asarray(probes)
    samples=np.linspace(.03,.97,7)
    interior=np.stack(np.meshgrid(samples,samples),-1)
    def match(H):
        observed = {}
        for uv in map(tuple,np.floor(project(np.linalg.inv(H),probes)).astype(int)):
            if uv in observed:
                continue
            coords=project(H,interior+uv).astype(int)
            if (coords<0).any() or (coords>=mask.shape[::-1]).any() or not keep[coords[...,1],coords[...,0]].all():
                observed[uv]=False
                continue
            quad=project(H,UNIT+uv)
            if np.linalg.norm(np.roll(quad,1,axis=0)-quad,axis=1).min()<6:
                observed[uv]=False
                continue
            hit=verify_sprite(rgb,quad)
            observed[uv]=bool(hit and hit[2]>=.65)
        votes=[observed[tuple(uv)] for uv in np.floor(project(np.linalg.inv(H),probes)).astype(int)]
        return votes,sum(observed.values())
    old_votes,old_tiles = match(old)
    new_votes,new_tiles = match(new)
    win=sum(b and not a for a,b in zip(old_votes,new_votes))
    lose=sum(a and not b for a,b in zip(old_votes,new_votes))
    return dict(accept=bool(new_tiles>=2 and win-lose>=max(2,.05*len(probes))),probes=len(probes),
                old_tiles=old_tiles,new_tiles=new_tiles,incumbent_only=lose,candidate_only=win)


def challenge(chart, plane, rgb, axes, sky, *, confirm_coverage=False):
    mask = chart_mask(chart, plane, rgb.shape[1::-1])
    y, x = np.nonzero(mask[::3, ::3])
    if len(x)<512:
        return chart, dict(reason='small visible support')
    split, divider = np.quantile(3*x+.5, [.47,.56])
    center = np.median(np.column_stack((x,y))*3+.5, axis=0)
    options = []
    for name, patch, support, rectified, xy in views(rgb, mask, chart, axes):
        def independent(region):
            local = []
            for rough, H, ratio in discover(patch, support & region, rectified, np.asarray(chart['H']), center):
                hit = verify_sprite(rgb, project(H, UNIT))
                if hit and hit[2] >= .70:
                    local.append((hit[2], H, ratio, hit[0], rough))
            return local
        near = independent(xy[...,0] < split)
        if not near:
            continue
        far = independent(xy[...,0] >= divider)
        for correlation, H, ratio, material, rough in near:
            for other_corr, other_H, other_ratio, other_material, other_rough in far:
                if abs(np.log(ratio/other_ratio)) > np.log(1.20):
                    continue
                same = material == other_material
                if not same and min(correlation,other_corr) < .83:
                    continue
                # The source regions chose complete sprites independently. Estimate their
                # integer separation from their common vanishing directions and local pitches.
                a, b = project(H,UNIT), project(other_H,UNIT)
                shift = np.mean(project(np.linalg.inv(H), b)-UNIT,axis=0)
                reciprocal = np.mean(UNIT-project(np.linalg.inv(other_H), a),axis=0)
                offset = np.rint((shift+reciprocal)/2)
                if max(abs(offset))<2 or max(abs(shift-reciprocal)) > 1.5:
                    continue
                fit,_ = cv2.findHomography(np.vstack((UNIT,UNIT+offset)), np.vstack((a,b)))
                if fit is None or not np.isfinite(fit).all():
                    continue
                uv_error = max(np.linalg.norm(project(np.linalg.inv(fit), np.vstack((a,b))) -
                                              np.vstack((UNIT,UNIT+offset)),axis=1))
                if uv_error>.2:
                    continue
                evidence = dict(axis=name, scale_ratio=ratio, near_material=material, near_correlation=correlation,
                                far_material=other_material, far_correlation=other_corr, far_ratio=other_ratio,
                                template_correlation=rough, far_template_correlation=other_rough,
                                near_quad=a.tolist(), far_quad=b.tolist(), integer_offset=offset.tolist(),
                                corner_residual=float(uv_error))
                value = min(correlation,other_corr) + .1*same - .15*uv_error
                options.append((value, fit, evidence))
    options.sort(key=lambda row:row[0],reverse=True)
    inspected = []
    for _, H, item in options:
        inspected.append(item)
        if confirm_coverage:
            evidence = independent_coverage(np.asarray(chart['H']), H,
                                            (item['near_quad'],item['far_quad']),mask,rgb)
            item['independent_coverage'] = evidence
            if not evidence['accept']:
                continue
        corrected = fit_homography([0,0], project(H,UNIT)[[3,2,1,0]])
        y, x = np.nonzero(mask)
        points = np.column_stack((x,y))+.5
        clean = {k:v for k,v in chart.items() if k!='export_uv'}
        new = resample(clean, corrected, mask, rgb, points)
        ranked = preserve_and_rank(dict(clean,cells=[]),new,rgb,mask)
        if not ranked['cells'] or len(ranked['cells']) > .7*len(chart['cells']):
            continue
        ranked['confidence'] = chart['confidence']
        guarded = border_filter([ranked], sky)[0][0]
        return guarded, dict(accepted=True, candidates=len(options), selected=item, before=len(chart['cells']),
                             after=len(ranked['cells']), trials=inspected)
    return chart, dict(accepted=False, candidates=len(options), trials=inspected)

