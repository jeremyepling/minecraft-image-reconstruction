"""Geometry-only rejection of small, elongated appendages attached at one end."""
import numpy as np

from engine.camera_lines import _runs
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha

SETTINGS = dict(radius_at_900px=4, minimum_tail_length_in_radii=4,
                maximum_mean_width_in_diameters=1, minimum_aspect_ratio=3,
                minimum_core_fraction=.60, maximum_tail_fraction=.14,
                maximum_contact_extent_fraction=.30)


def neighborhood(mask, radius):
    padded = np.pad(mask, radius).astype(np.int32)
    sums = np.pad(padded, ((1, 0), (1, 0))).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
    size = 2 * radius + 1
    return sums[size:,size:] - sums[:-size,size:] - sums[size:,:-size] + sums[:-size,:-size]


def appendages(owned, radius):
    size = 2 * radius + 1
    interior = neighborhood(owned, radius) == size**2
    body = neighborhood(interior, radius) > 0
    removed = np.zeros_like(owned)
    total = int(owned.sum())
    if body.sum() < SETTINGS['minimum_core_fraction'] * total:
        return removed
    contact = neighborhood(body, 1) > 0
    for runs in _runs(owned & ~body):
        area = sum(b-a for _,a,b in runs)
        if area < 4*radius or area > SETTINGS['maximum_tail_fraction']*total:
            continue
        y = np.repeat([r[0] for r in runs], [b-a for _,a,b in runs])
        x = np.concatenate([np.arange(a,b) for _,a,b in runs])
        attached = contact[y,x]
        if not attached.any():
            continue
        coords = np.column_stack((x, y))
        centered = coords - coords.mean(axis=0)
        axis = np.linalg.eigh(centered.T @ centered)[1][:,-1]
        projection = centered @ axis
        length = np.ptp(projection) + 1
        width = area/length
        if length < 4*radius or width > size or length < SETTINGS['minimum_aspect_ratio']*width:
            continue
        # Reject an edge fringe, an internal bridge or an island; accept a terminal ribbon.
        reach = min(projection[attached].max()-projection.min(), projection.max()-projection[attached].min())
        if reach <= SETTINGS['maximum_contact_extent_fraction'] * length:
            removed[y,x] = True
    return removed


def refine(photo, parts, gate=None):
    radius = max(1, round(SETTINGS['radius_at_900px'] * photo.height / 900))
    output, changes = [], []
    for part in parts:
        native = _native_alpha(part['support'], photo.size)
        if native is None:
            output.append(part)
            continue
        alpha, box = native
        owned = np.asarray(alpha, dtype=bool)
        removed = appendages(owned, radius)
        if removed.any() and gate is not None:
            accepted = gate(np.asarray(photo.crop(box)), owned, removed)
            if accepted.shape != owned.shape or accepted.dtype != bool or (accepted & ~removed).any():
                raise ValueError('Evidence gate must return a subset of proposed removals')
            removed = accepted
        if not removed.any():
            output.append(part)
            continue
        changes.append((part['id'], box, removed))
        kept = owned & ~removed
        x,y,_,_ = box
        runs = np.asarray(row_runs(kept)) + [y, x, x]
        cleaned = {k:v for k,v in part.items() if k not in
                   ('outline','corner_junctions','source_crop','context_crop','sampling_chart')}
        cleaned.update(support=dict(size=list(photo.size),runs=runs.tolist()),support_area=int(kept.sum()))
        output.append(cleaned)
    affected = {p for p,_,_ in changes}
    if affected:
        output = [dict(p, corner_junctions=[j for j in p['corner_junctions'] if j['neighbor'] not in affected])
                  if 'corner_junctions' in p else p for p in output]
    return output, changes
