"""Benchmark ablations for transferring photographed texture phase across a block boundary."""
import hashlib
import json
import os
from pathlib import Path

from benchmark.visible_depth_score import integrated
from engine.neighbor_photo_faces import rows, select_joint


def select(rows, variant):
    if variant.startswith('joint'):
        return select_joint(rows, seed_fraction=.08 if variant == 'joint_partial' else .35,
                            grow=variant == 'joint_growth')
    accepted = set()
    for cell, axis, seen, area, matches in rows:
        if variant == 'oracle_pool':
            yes = bool(matches)
        elif variant == 'permissive':
            yes = any(n >= 16 and err <= 10 and gain >= 1.2 for n, err, gain, _ in matches)
        elif variant == 'phase':
            yes = any(n >= 20 and err <= 7 and gain >= 2.5 and low >= -.3
                      for n, err, gain, low in matches)
        elif variant == 'strict':
            yes = any(n >= 24 and err <= 4 and gain >= 2.5 and low >= .3
                      for n, err, gain, low in matches)
        else:
            raise ValueError(variant)
        if yes:
            accepted.add(tuple(cell))
    return accepted


def predict(image, version, parts, counts, camera):
    # Private benchmark cache contains image-only intermediate rows and placement.
    cache = os.environ.get('MCIR_NEIGHBOR_PHOTO_CACHE')
    key = hashlib.sha256(image.convert('RGB').tobytes())
    key.update(json.dumps([version, image.size, parts, counts, camera], sort_keys=True).encode())
    path = Path(cache)/f'{key.hexdigest()}.json' if cache else None
    if path and path.exists():
        data = json.loads(path.read_text())
        base = set(map(tuple, data['base']))
        features = data['rows']
        status = {}
    else:
        base, status = integrated(image, version, parts, counts, camera)
        features = rows(image, parts, camera['P'], base)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(base=sorted(base), rows=features)))
    variant = os.environ.get('MCIR_NEIGHBOR_PHOTO_VARIANT','phase')
    extra = select(features, variant)
    if extra and variant.startswith('joint'):
        # Recheck every source and target pixel used in a match after resolving
        # visibility of all selected candidates together.
        extra &= select(rows(image, parts, camera['P'], base, competing=extra), variant)
    if extra:
        from engine.visible_ray_filter import fully_occluded_cells
        extra -= fully_occluded_cells(image.size, camera['P'], base | extra)
    return base | extra, status
