"""Recover unclaimed surfaces and optionally split existing masks with Pixel-wise Planarity."""
from itertools import combinations

import numpy as np
from PIL import Image, ImageFilter

from benchmark.mask_metrics import footprint
from benchmark.mobile_wall_split import children
from engine.camera_lines import _runs
from engine.normal_regions import _patch
from engine.surface_model import canonical


SETTINGS = dict(min_samples=24, max_add_area_fraction=.22, min_novel_fraction=.60,
                contact_radius=4, min_contact_samples=3, max_duplicate_iou=.45,
                min_split_containment=.72, min_split_share=.12, min_split_coverage=.72,
                split_angle_deg=18, split_plane_distance_fraction=.035)
PRESETS = {
    'standard-add': ('labels',False),
    'standard-add-split': ('labels',True),
    'thin-add-split': ('thin',True),
    'relaxed-add-split': ('relaxed',True),
}


def distinct_planes(i, j, normals, centers):
    if normals[i] @ normals[j] < np.cos(np.radians(SETTINGS['split_angle_deg'])):
        return True
    # A sloping wall changes in depth along its width. Compare the distance from the
    # other centroid to each plane, rather than comparing the two camera depths.
    delta = centers[i]-centers[j]
    separation = min(abs(normals[i] @ delta), abs(normals[j] @ delta))
    return separation > SETTINGS['split_plane_distance_fraction'] * min(centers[i,2],centers[j,2])


def split_parent(image, part, counts, areas, labels, normals, centers):
    parent_area = counts.sum()
    usable = np.flatnonzero((counts >= SETTINGS['min_samples']) &
                            (counts >= parent_area * SETTINGS['min_split_share']) &
                            (counts >= areas * SETTINGS['min_split_containment']))
    usable = [int(i) for i in usable if i and centers[i,2]>0 and np.linalg.norm(normals[i])>.9]
    pairs = [(int(counts[i]+counts[j]),i,j) for i,j in combinations(usable,2)
             if distinct_planes(i,j,normals,centers)]
    if not pairs:
        return None
    explained, i, j = max(pairs)
    if explained < SETTINGS['min_split_coverage'] * parent_area:
        return None
    masks = np.stack((labels==i,labels==j))
    minimum_seed = 8*np.prod(part['support']['size'])/labels.size
    pieces = children(image,part,[0,1],masks,[{'quality':1.},{'quality':1.}],minimum_seed_area=minimum_seed)
    if not {0,1}.issubset(p['mobile_sam_proposal'] for p in pieces):
        return None
    for k,piece in enumerate(pieces):
        seed = piece.pop('mobile_sam_proposal')
        label = [i,j][seed] if seed is not None else None
        piece.update(id=f"{part['id']}-planarity-split-{k}", planarity_label=label)
        if label is not None:
            piece['normal_camera'] = normals[label].tolist()
    return pieces


def new_regions(image, union, labels, normals, counts, areas):
    """Seed with existing silhouettes but permit unsupported pixels and attached new faces."""
    size = labels.shape[::-1]
    if not union.any():
        return []
    radius = SETTINGS['contact_radius']
    near = np.asarray(Image.fromarray(union.astype(np.uint8)).filter(ImageFilter.MaxFilter(2*radius+1)))>0
    contacts = np.bincount(labels[near], minlength=len(areas))
    covered = np.bincount(labels[union], minlength=len(areas))
    # The support of a large floor cannot justify a plane-sized patch beyond the building.
    rim = np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
    roofline = np.where(near,np.arange(size[1])[:,None],size[1]).min(axis=0)
    groundline = np.where(near,np.arange(size[1])[:,None],-1).max(axis=0)
    gray = np.asarray(image.convert('L').resize(size,Image.Resampling.BOX)
                      .filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    scale = np.asarray(image.size)/size
    output = []
    for i in range(1,len(areas)):
        area = areas[i]
        if (i in rim or area<SETTINGS['min_samples'] or area>labels.size*SETTINGS['max_add_area_fraction']
                or covered[i]>area*(1-SETTINGS['min_novel_fraction'])
                or contacts[i]<SETTINGS['min_contact_samples'] or np.linalg.norm(normals[i])<.9):
            continue
        if counts.size and np.any(counts[:,i] >= SETTINGS['max_duplicate_iou'] *
                                  (counts.sum(axis=1)+area-counts[:,i])):
            continue
        candidate = (labels==i)&~union
        # Small islands in the same full-frame model ID can disconnect after either
        # resampling or exclusion of owned pixels. Test their image contact separately.
        for runs in _runs(candidate):
            n = sum(b-a for _,a,b in runs)
            if n<SETTINGS['min_samples'] or sum(int(near[y,a:b].sum()) for y,a,b in runs)<SETTINGS['min_contact_samples']:
                continue
            if abs(normals[i,1])>.75:
                spans = sum(int(((y>=roofline[a:b])&(y<=groundline[a:b])).sum()) for y,a,b in runs)
                if spans < .5*n:
                    continue
            result = dict(id=f'planarity-new-{i}-{len(output)}', planarity_label=i, normal_camera=normals[i].tolist(),
                          relative_inverse_depth=None, geometry_status='mask only', support_area=n,
                          support=dict(size=list(size), runs=runs))
            chart = _patch(runs,gray,scale,size,normals[i])
            if chart:
                result.update(quad=canonical(chart['quad']), geometry_status='mask with sampling chart')
            output.append(result)
    return output


def predict(image, parts, fields, preset):
    """Image and numerical predictions only. The original mask constrains every split."""
    key, split = PRESETS[preset]
    labels = fields[key]
    normals, centers = fields[key+'_normals'], fields[key+'_centers']
    if labels.shape!=(image.height//4,image.width//4) or not np.isfinite(normals).all() or not np.isfinite(centers).all():
        raise ValueError('Misaligned or nonfinite planarity fields')
    areas = np.bincount(labels.ravel())
    union = np.zeros_like(labels,bool)
    counts = np.zeros((len(parts),len(areas)),int)
    result, events = [], []
    for k,part in enumerate(parts):
        m = footprint(part,labels.shape,image.size,True)
        union |= m
        counts[k] = np.bincount(labels[m],minlength=len(areas))
        pieces = split_parent(image,part,counts[k],areas,labels,normals,centers) if split else None
        if pieces:
            events.append(dict(action='split', parent=part['id'], children=[p['id'] for p in pieces]))
            result.extend(pieces)
        else:
            result.append(part)
    additions = new_regions(image,union,labels,normals,counts,areas)
    result.extend(additions)
    events.extend(dict(action='add',id=p['id'],label=p['planarity_label']) for p in additions)
    return result,events
