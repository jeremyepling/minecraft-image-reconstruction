"""Carve independent plane labels out of a larger mask when its neighbors support a depth step or turn."""
import numpy as np
from PIL import Image, ImageFilter

from benchmark.mask_metrics import footprint
from benchmark.mobile_wall_split import children


# Distances are relative to the depth of the nearer plane, never interpreted as block counts.
PRESETS = {
    'clear-step': dict(containment=.85, min_share=.04, plane_distance=.05, turn_degrees=30,
                       neighbor_samples=12, distinct_share=.75),
    'supported-step': dict(containment=.72, min_share=.025, plane_distance=.035, turn_degrees=23,
                           neighbor_samples=8, distinct_share=.65),
    'small-step': dict(containment=.62, min_share=.015, plane_distance=.025, turn_degrees=18,
                       neighbor_samples=6, distinct_share=.60),
}
SETTINGS = dict(min_samples=24, max_parent_share=.45, neighbor_radius=9, min_seed_samples=8,
                max_host_angle_degrees=25)


def split_labels(parent, host_normal, labels, areas, normals, centers, settings):
    counts = np.bincount(labels[parent], minlength=len(areas))
    area = int(parent.sum())
    usable = np.flatnonzero((counts >= max(SETTINGS['min_samples'], area * settings['min_share']))
                            & (counts <= area * SETTINGS['max_parent_share'])
                            & (counts >= areas * settings['containment']))
    valid = (np.linalg.norm(normals,axis=1) > .90) & (centers[:,2] > 0)
    valid[0] = False
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1])))
    host_direction = normals @ host_normal > np.cos(np.radians(SETTINGS['max_host_angle_degrees']))
    selections = []
    for raw_label in usable[valid[usable]]:
        label = int(raw_label)
        if label in border:
            continue
        seed = (labels == label) & parent
        # Restrict neighbor evidence to the same original parent. Other surfaces across an
        # established outline cannot authorize another split in this wall.
        dilation = Image.fromarray(seed).filter(ImageFilter.MaxFilter(2 * SETTINGS['neighbor_radius'] + 1))
        ring = np.asarray(dilation, bool) & parent & ~seed
        neighbors = np.bincount(labels[ring], minlength=len(areas))
        neighbors[~(valid & host_direction)] = 0
        support = int(neighbors.sum())
        if support < settings['neighbor_samples']:
            continue
        offset = centers - centers[label]
        # Distance along each plane's normal discounts depth changes due to perspective.
        separation = np.minimum(np.abs(offset @ normals[label]), np.abs((offset * normals).sum(axis=1)))
        distance = separation / np.maximum(1e-7, np.minimum(centers[:,2], centers[label,2]))
        turned = (normals @ normals[label]) < np.cos(np.radians(settings['turn_degrees']))
        distinct = int(neighbors[turned | (distance >= settings['plane_distance'])].sum())
        if distinct >= settings['neighbor_samples'] and distinct >= support * settings['distinct_share']:
            selections.append(dict(label=label, area=int(counts[label]), neighbor_samples=support,
                                   distinct_samples=distinct))
    return selections


def predict(image, parts, fields, preset):
    """Refine only existing support. Camera poses and recorded surfaces are never used here."""
    labels = fields['relaxed']
    normals, centers = fields['relaxed_normals'], fields['relaxed_centers']
    if labels.shape != (image.height // 4, image.width // 4):
        raise ValueError('Plane labels and screenshot dimensions differ')
    areas = np.bincount(labels.ravel(), minlength=len(normals))
    settings = PRESETS[preset]
    output, events = [], []
    for part in parts:
        parent = footprint(part, labels.shape, image.size, True)
        selected = split_labels(parent, np.asarray(part['normal_camera']), labels, areas, normals, centers, settings)
        if not selected:
            output.append(part)
            continue
        minimum = SETTINGS['min_seed_samples'] * np.prod(part['support']['size']) / labels.size
        original = part
        for event in sorted(selected, key=lambda e: (-e['distinct_samples']/e['neighbor_samples'],
                                                     -e['area'], e['label'])):
            label = event['label']
            seed = labels == label
            masks = np.stack((~seed, seed))
            pieces = children(image, part, [0, 1], masks, [{'quality':0.}, {'quality':1.}],
                              minimum_seed_area=minimum)
            # A single coherent inset must leave one coherent remainder. Ignore a label
            # that would break the parent into many disconnected guesses.
            by_side = {piece.pop('mobile_sam_proposal'):piece for piece in pieces}
            if len(pieces) != 2 or set(by_side) != {0, 1}:
                continue
            inset, remainder = by_side[1], by_side[0]
            inset.update(id=f"{original['id']}-inset-{label}", planarity_label=label,
                         normal_camera=normals[label].tolist())
            remainder['id'] = f"{original['id']}-remainder-{label}"
            events.append(dict(action='split', parent=part['id'], plane=event,
                               children=[inset['id'], remainder['id']]))
            output.append(inset)
            part = remainder
        output.append(part)
    return output, events
