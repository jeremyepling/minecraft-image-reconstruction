"""Score visible region masks and legacy quad footprints in the same image space."""
from collections import defaultdict

import numpy as np

from benchmark.camera_labels import rasterize
from benchmark.surface_metrics import DOWNSAMPLE, _one_to_one, observed_support
from benchmark.surface_topology import opening_mask
from engine.camera_lines import _runs


PROTOCOL = dict(iou=.5, sample_stride=DOWNSAMPLE, boundary_tolerance_px=8, corner_tolerance_px=12,
                contour_simplification_px=6, minimum_corner_turn_deg=30, minimum_contact_samples=2,
                contact_depth_blocks=.5, filled_gap_fraction=.5)


def footprint(region, shape, size, use_mask):
    if use_mask:
        return observed_support(region['support'], shape, size)
    quad = np.column_stack((np.asarray(region['quad']) / DOWNSAMPLE, np.ones(4)))
    result = rasterize(quad, shape[1], shape[0])
    mask = np.zeros(shape, bool)
    if result is not None:
        mask[result[0]] = result[1]
    return mask


def outline(mask):
    """All visible perimeter samples, omitting arbitrary image clipping edges."""
    inner = np.pad(mask, 1, mode='edge')
    border = mask & ~(inner[:-2,1:-1] & inner[2:,1:-1] & inner[1:-1,:-2] & inner[1:-1,2:])
    return np.column_stack(np.nonzero(border))


def simplified(points, epsilon):
    if len(points) <= 2:
        return points
    a, b = points[0], points[-1]
    direction = b - a
    offsets = points - a
    t = np.clip((offsets @ direction) / max(1, direction @ direction), 0, 1)
    distances = np.linalg.norm(offsets - t[:,None] * direction, axis=1)
    i = int(np.argmax(distances))
    if distances[i] <= epsilon:
        return points[[0,-1]]
    return np.concatenate((simplified(points[:i+1],epsilon)[:-1], simplified(points[i:],epsilon)))


def corners(mask):
    """Trace every closed ring, holes included. Reduce raster stair-steps before scoring turns."""
    m = np.pad(mask, 1)
    sides = [(m[:-2,1:-1],(0,0),(0,1)), (m[1:-1,2:],(0,1),(1,1)),
             (m[2:,1:-1],(1,1),(1,0)), (m[1:-1,:-2],(1,0),(0,0))]
    edges = defaultdict(list)
    for neighbor, start, end in sides:
        y,x = np.nonzero(mask & ~neighbor)
        for a,b in zip(zip(y+start[0],x+start[1]),zip(y+end[0],x+end[1])):
            edges[a].append(b)
    result = []
    while edges:
        start = next(iter(edges))
        ring, at = [], start
        while at in edges:
            ring.append(at)
            choices = edges[at]
            if len(choices) > 1 and len(ring) > 1:
                dy,dx = at[0]-ring[-2][0], at[1]-ring[-2][1]
                right = (at[0]+dx,at[1]-dy)
                nxt = choices.pop(choices.index(right) if right in choices else 0)
            else:
                nxt = choices.pop()
            if not choices:
                del edges[at]
            at = nxt
            if at == start:
                break
        if len(ring) < 4:
            continue
        points = np.asarray(ring)
        k = int(np.argmax(((points-points[0])**2).sum(1)))
        eps = PROTOCOL['contour_simplification_px']/DOWNSAMPLE
        curve = np.concatenate((simplified(points[:k+1],eps)[:-1],
                                simplified(np.vstack((points[k:],points[0])),eps)[:-1]))
        if len(curve) < 3:
            continue
        prev,after = curve-np.roll(curve,1,axis=0), np.roll(curve,-1,axis=0)-curve
        cos = (prev*after).sum(1)/np.maximum(1,np.linalg.norm(prev,axis=1)*np.linalg.norm(after,axis=1))
        valid = np.all((curve>0)&(curve<mask.shape),axis=1)
        result.extend(curve[valid & (cos <= np.cos(np.radians(PROTOCOL['minimum_corner_turn_deg'])))])
    return np.asarray(result).reshape(-1,2)


def distances(a,b):
    if not len(a) or not len(b):
        return np.empty(0),np.empty(0)
    forward = []
    reverse = np.full(len(b), np.inf)
    for lo in range(0, len(a), 100):
        delta = a[lo:lo+100,None,:]-b[None,:,:]
        sq = np.einsum('ijk,ijk->ij',delta,delta)
        forward.extend(np.min(sq,axis=1))
        reverse = np.minimum(reverse,np.min(sq,axis=0))
    return np.sqrt(forward)*DOWNSAMPLE,np.sqrt(reverse)*DOWNSAMPLE


def adjacencies(ids):
    pairs = defaultdict(list)
    for axis in (0,1):
        a,b = (ids[:-1],ids[1:]) if axis==0 else (ids[:,:-1],ids[:,1:])
        ys,xs = np.nonzero((a != b) & (a>0) & (b>0))
        for y,x in zip(ys,xs):
            pair = tuple(sorted((int(a[y,x]),int(b[y,x]))))
            pairs[pair].append(((x+.5+.5*(axis==1))*DOWNSAMPLE,(y+.5+.5*(axis==0))*DOWNSAMPLE,1.))
    return {pair:points for pair,points in pairs.items() if len(points)>=PROTOCOL['minimum_contact_samples']}


def physical_contacts(reference, eligible):
    pairs = set()
    for (a,b), points in adjacencies(reference['id_map']).items():
        if a not in eligible or b not in eligible:
            continue
        rays = np.asarray(points).T
        # Both recorded planes must meet at the same camera ray, including at silhouettes of recesses.
        za,zb = [1/np.linalg.solve(reference['charts'][i],rays)[2] for i in (a,b)]
        delta = abs(za-zb)
        if np.count_nonzero(delta<=PROTOCOL['contact_depth_blocks']) >= PROTOCOL['minimum_contact_samples']:
            pairs.add((a,b))
    return pairs


def measure(regions, reference, *, use_mask):
    ids = reference['id_map']
    eligible = {i for i,p in reference['patches'].items() if p['eligible']}
    areas = np.bincount(ids.ravel())
    pred_ids = np.zeros_like(ids)
    masks = [footprint(p,ids.shape,reference['image_size'],use_mask) for p in regions]
    counts = [int(m.sum()) for m in masks]
    pairs = []
    for i in sorted(range(len(masks)), key=lambda k:-counts[k]):
        mask = masks[i]
        pred_ids[mask] = i+1  # Smaller footprints own overlaps; identical areas use saved order.
        hits = np.bincount(ids[mask],minlength=len(areas))
        for pid in eligible:
            iou = hits[pid]/(counts[i]+areas[pid]-hits[pid])
            if iou >= PROTOCOL['iou']:
                pairs.append((float(iou),i,pid))
    matches = _one_to_one(pairs)
    lookup = {i+1:pid for i,pid in matches}
    unmatched_categories = dict(overlaps_small_surface=0,mostly_background=0,fragment_or_mixed=0)
    for i,mask in enumerate(masks):
        if i+1 in lookup:
            continue
        hits = np.bincount(ids[mask],minlength=len(areas))
        best = int(np.argmax(hits))
        if best>0 and best not in eligible and hits[best]/(counts[i]+areas[best]-hits[best])>=PROTOCOL['iou']:
            unmatched_categories['overlaps_small_surface']+=1
        elif not counts[i] or hits[0]>.5*counts[i]:
            unmatched_categories['mostly_background']+=1
        else:
            unmatched_categories['fragment_or_mixed']+=1
    eligible_pixels = np.isin(ids,list(eligible))
    covered = pred_ids>0
    corr = int(sum(np.count_nonzero((ids==pid)&(pred_ids==i+1)) for i,pid in matches))
    openings = opening_mask(ids)
    filled = openings & covered
    gap_sizes = []
    for runs in _runs(openings):
        n = sum(b-a for _,a,b in runs)
        hits = sum(int(covered[y,a:b].sum()) for y,a,b in runs)
        gap_sizes.append((n,hits))
    true_edges = physical_contacts(reference,eligible)
    claimed = adjacencies(pred_ids)
    good = {tuple(sorted((lookup[a],lookup[b]))) for a,b in claimed if a in lookup and b in lookup} & true_edges
    out = dict(eligible=len(eligible),proposed=len(regions),matched=len(matches),
               **{'unmatched_'+k:v for k,v in unmatched_categories.items()},
               eligible_pixels=int(eligible_pixels.sum()),correct_pixels=int(corr),
               wrong_plane_pixels=int((eligible_pixels & covered).sum())-corr,
               unclaimed_pixels=int((eligible_pixels & ~covered).sum()),background_claimed=int(((ids==0)&covered).sum()),
               opening_pixels=int(openings.sum()),opening_filled=int(filled.sum()),gaps=len(gap_sizes),
               gaps_half_filled=sum(hits >= n * .5 for n,hits in gap_sizes),
               real_connections=len(true_edges),proposed_connections=len(claimed),correct_connections=len(good))
    for name,extract,tolerance in (('edge',outline,8),('corner',corners,12)):
        true_points = {pid:extract(ids==pid) for pid in eligible}
        pred_points = [extract(m) for m in masks]
        out[name+'_true'] = sum(map(len,true_points.values()))
        out[name+'_proposed'] = sum(map(len,pred_points))
        ds,ns,near_true,near_pred = 0.,0,0,0
        for i,pid in matches:
            u,v = distances(pred_points[i],true_points[pid])
            ds += u.sum()+v.sum()
            ns += len(u)+len(v)
            near_pred += np.count_nonzero(u<=tolerance)
            near_true += np.count_nonzero(v<=tolerance)
        out.update({name+'_distance_sum':float(ds),name+'_distance_count':ns,
                    name+'_true_near':int(near_true),name+'_proposed_near':int(near_pred)})
    return {k:(v.item() if isinstance(v,np.generic) else v) for k,v in out.items()}, matches


def aggregate(rows):
    total = {k:sum(r[k] for r in rows) for k in rows[0]}
    def ratio(a,b):
        return a/b if b else None
    total.update(f1=ratio(2*total['matched'],total['proposed']+total['eligible']),
                 assigned_fraction=ratio(total['correct_pixels'],total['eligible_pixels']),
                 opening_fill=ratio(total['opening_filled'],total['opening_pixels']))
    for name in ('edge','corner'):
        total[name+'_error_px'] = ratio(total[name+'_distance_sum'],total[name+'_distance_count'])
        total[name+'_recall'] = ratio(total[name+'_true_near'],total[name+'_true'])
        total[name+'_precision'] = ratio(total[name+'_proposed_near'],total[name+'_proposed'])
    return total
