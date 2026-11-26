"""Recover a plane when two independent model outlines agree on the same visible region."""
import numpy as np
from PIL import Image, ImageFilter

from benchmark.mask_metrics import footprint
from engine.camera_lines import _runs
from engine.normal_regions import _patch
from engine.surface_model import canonical


PRESETS = {
    'agreement-only': {'min_model_iou': .50, 'check_neighbor': False},
    'agreement-and-step': {'min_model_iou': .42, 'check_neighbor': True},
    'partial-agreement-and-step': {'min_model_iou': .30, 'check_neighbor': True},
}
SETTINGS = dict(min_area=24, min_sam_quality=.70, min_sam_stability=.65, min_sam_coverage=.60,
                min_building_share=.35, duplicate_iou=.50, protected_normal_deg=28,
                protected_part_share=.70, minimum_changed_share=.025, min_component_area=12,
                minimum_extension_iou=.50, neighbor_radius=8, neighbor_samples=12,
                step_angle_deg=18, relative_plane_step=.035, extension_share=.18)


def model_pairs(labels, masks, records, min_iou):
    """Best SAM match for each raw plane, using only the half-open SAM box."""
    areas = np.bincount(labels.ravel())
    best = {}
    for index, record in enumerate(records):
        if record['quality'] < SETTINGS['min_sam_quality'] or record['stability'] < SETTINGS['min_sam_stability']:
            continue
        x0, y0, x1, y1 = record['bbox']
        pixels = labels[y0:y1, x0:x1][masks[index, y0:y1, x0:x1]]
        if len(pixels) < SETTINGS['min_area']:
            continue
        hits = np.bincount(pixels, minlength=len(areas))
        iou = hits / (areas + len(pixels) - hits)
        for label in np.flatnonzero((iou >= min_iou) & (hits >= SETTINGS['min_sam_coverage'] * areas)):
            if label:
                rank = (float(iou[label]), float(record['quality']), -index)
                if label not in best or rank > best[label]:
                    best[int(label)] = rank
    return best


def resize(mask, size):
    if mask.shape[::-1] == tuple(size):
        return mask
    return np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))


def run_mask(support):
    w, h = support['size']
    mask = np.zeros((h, w), bool)
    for y, a, b in support['runs']:
        mask[y, a:b] = True
    return mask


def row_runs(mask):
    y, x = np.nonzero(np.diff(np.pad(mask, ((0, 0), (1, 1))), axis=1))
    return np.column_stack((y[::2], x[::2], x[1::2])).tolist()


def with_mask(part, mask, size, gray, scale):
    runs = row_runs(mask)
    normal = np.asarray(part['normal_camera'])
    # Construct the geometry from the new visible pixels. An old corner or outline
    # attached to the residual would incorrectly span the recovered face.
    result = {k:v for k,v in part.items() if k not in
              ('quad','sampling_chart','outline','geometry_status','support','support_area')}
    result.update(support=dict(size=list(size), runs=runs), support_area=int(mask.sum()), geometry_status='mask only')
    chart = _patch(runs, gray, scale, size, normal)
    if chart:
        result.update(quad=canonical(chart['quad']), geometry_status='mask with sampling chart')
    return result


def step_supported(region, parent, labels, label, normals, centers):
    # Look across the proposed boundary within the old surface. Use the most
    # observed neighboring plane, so a single isolated normal cannot prove a cut.
    width = 2 * SETTINGS['neighbor_radius'] + 1
    ring = np.asarray(Image.fromarray(region.astype('uint8')).filter(ImageFilter.MaxFilter(width)), bool)
    counts = np.bincount(labels[ring & parent & ~region], minlength=len(normals))
    counts[0] = 0
    neighbor = int(np.argmax(counts))
    if counts[neighbor] < SETTINGS['neighbor_samples']:
        return False
    if normals[label] @ normals[neighbor] < np.cos(np.radians(SETTINGS['step_angle_deg'])):
        return True
    delta = centers[label] - centers[neighbor]
    distance = min(abs(delta @ normals[label]), abs(delta @ normals[neighbor]))
    depth = min(centers[label,2], centers[neighbor,2])
    return depth > 0 and distance >= SETTINGS['relative_plane_step'] * depth


def predict(image, parts, fields, preset, *, sam_masks, sam_records):
    labels = fields['relaxed']
    normals = fields['relaxed_normals']
    centers = fields['relaxed_centers']
    if labels.shape != (image.height//4, image.width//4) or sam_masks.shape != (len(sam_records), *labels.shape):
        raise ValueError('Model outputs must share the screenshot sampling grid')
    if not parts or not len(sam_records):
        return list(parts), []
    pairs = model_pairs(labels, sam_masks, sam_records, PRESETS[preset]['min_model_iou'])
    border = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    prior = np.stack([footprint(part, labels.shape, image.size, True) for part in parts])
    union = prior.any(axis=0)
    areas = prior.sum(axis=(1,2))
    part_normals = np.asarray([p['normal_camera'] for p in parts])
    # Use the finest common support grid when its cells exactly subdivide every old mask.
    # Otherwise the source pixel grid preserves the old visible pixels.
    sizes = [p['support']['size'] for p in parts] + [labels.shape[::-1]]
    size = tuple(max(s[i] for s in sizes) for i in range(2))
    if any(size[0] % w or size[1] % h for w,h in sizes):
        size = image.size
    raw = [resize(run_mask(p['support']), size) for p in parts]
    building = np.logical_or.reduce(raw)
    outside = None
    assigned = np.zeros_like(building, dtype=np.int32)
    events = []
    for label, (iou, quality, minus_index) in sorted(pairs.items()):
        region = labels == label
        area = int(region.sum())
        if (label in border or area < SETTINGS['min_area'] or np.linalg.norm(normals[label]) < .9
                or union[region].mean() < SETTINGS['min_building_share']):
            continue
        hits = prior[:, region].sum(axis=1)
        if np.any(hits >= SETTINGS['duplicate_iou'] * (areas + area - hits)):
            continue
        sam_region = region & sam_masks[-minus_index]
        extension = np.count_nonzero(sam_region & ~union) / area
        extends_outline = iou >= SETTINGS['minimum_extension_iou'] and extension >= SETTINGS['extension_share']
        if (PRESETS[preset]['check_neighbor'] and not extends_outline and
                not step_supported(region, prior[int(np.argmax(hits))], labels, label, normals, centers)):
            continue
        # A planarity label can cross an earlier outline. A small differently facing
        # surface entirely inside it remains owned by that earlier surface.
        protect = ((hits > 0) & ((hits < SETTINGS['minimum_changed_share'] * areas) |
                   ((hits >= SETTINGS['protected_part_share'] * areas) &
                    (part_normals @ normals[label] < np.cos(np.radians(SETTINGS['protected_normal_deg']))))))
        mask = resize(region, size) & building
        if extends_outline:
            if outside is None:
                # A proposed extension can reach beyond the silhouette. The
                # shared models cannot turn an enclosed doorway into opaque wall.
                outside = np.zeros_like(building)
                for runs in _runs(~building):
                    if any(y in (0,size[1]-1) or a == 0 or b == size[0] for y,a,b in runs):
                        for y,a,b in runs:
                            outside[y,a:b] = True
            mask |= resize(sam_region, size) & outside
        for index in np.flatnonzero(protect):
            mask &= ~raw[index]
        # The constrained mask can closely duplicate an existing surface even
        # when the unconstrained full-image label looked new.
        if not mask.any() or any(np.count_nonzero(mask & old) >= SETTINGS['duplicate_iou'] *
                                 np.count_nonzero(mask | old) for old in raw):
            continue
        # A disconnected fragment remains separate, including where protection splits a raw label.
        minimum = SETTINGS['min_component_area'] * mask.size / labels.size
        for component in _runs(mask):
            if sum(b - a for _,a,b in component) < minimum:
                continue
            number = len(events) + 1
            for y, a, b in component:
                assigned[y, a:b] = number
            events.append(dict(label=label, sam_proposal=-minus_index, model_iou=iou,
                               sam_quality=quality, id=f'plane-consensus-{label}-{number}'))
    if not events:
        return list(parts), []
    gray = np.asarray(image.convert('L').resize(size, Image.Resampling.BOX)
                      .filter(ImageFilter.GaussianBlur(.9)), dtype=np.float32)[...,None]
    scale = np.asarray(image.size) / size
    result = []
    for part, original in zip(parts, raw):
        if not np.any(original & (assigned > 0)):
            result.append(part)
        else:
            rest = original & (assigned == 0)
            if rest.any():
                result.append(with_mask(part, rest, size, gray, scale))
    for number, event in enumerate(events, 1):
        label = event['label']
        part = dict(id=event['id'], relative_inverse_depth=None, normal_camera=normals[label].tolist(),
                    planarity_label=label, mobile_sam_proposal=event['sam_proposal'])
        result.append(with_mask(part, assigned == number, size, gray, scale))
    return result, events
