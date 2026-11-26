"""Partition existing visible surfaces using registered close-up plane proposals."""
import numpy as np

from benchmark.mask_metrics import footprint
from engine.planarity_fusion import split_parent
from engine.planarity_source import SETTINGS as MODEL_SETTINGS


def predict(image, parts, tiles):
    """Choose one supported crop split per original surface, keeping the first crop on a tie."""
    stride = MODEL_SETTINGS['stride']
    shape = (image.height // stride, image.width // stride)
    parents = [footprint(part, shape, image.size, use_mask=True) for part in parts]
    choices = {}
    for tile, labels, normals, centers in tiles:
        if labels.shape != shape:
            raise ValueError('Crop plane labels are on a different image grid')
        areas = np.bincount(labels.ravel())
        for index, (part, parent) in enumerate(zip(parts, parents)):
            hits = np.bincount(labels[parent], minlength=len(areas))
            children = split_parent(image, part, hits, areas, labels, normals, centers)
            if children:
                seeds = {p['planarity_label'] for p in children if p['planarity_label'] is not None}
                support = sum(int(hits[label]) for label in seeds)
                if index not in choices or support > choices[index][0]:
                    choices[index] = (support, tile, children)

    result, events = [], []
    for index, part in enumerate(parts):
        if index in choices:
            support, tile, children = choices[index]
            result.extend(children)
            events.append(dict(action='split', parent=part['id'], tile=tile, support_samples=support,
                               children=[p['id'] for p in children]))
        else:
            result.append(part)
    return result, events
