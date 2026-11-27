"""Verify straight visible walls where placed top and base rows agree at a corner."""
from collections import defaultdict

import numpy as np

from engine.pillar_seams import native_mask, orientation
from engine.projected_appearance import STEP, Scene


def arm(x, z, dx, dz, length):
    return [(x + i * dx, z + i * dz) for i in range(length)]


def proposals(cells):
    rows = defaultdict(set)
    for x, y, z in cells:
        rows[y].add((x, z))
    candidates = []
    for lower, bottom in rows.items():
        for upper, top in rows.items():
            if upper - lower < 3:
                continue
            anchors = bottom & top
            for x, z in sorted(anchors):
                for dx in (-1, 1):
                    for dz in (-1, 1):
                        if (x-dx, z) in anchors or (x, z-dz) in anchors:
                            continue
                        width = depth = 1
                        while (x+dx*width, z) in anchors:
                            width += 1
                        while (x, z+dz*depth) in anchors:
                            depth += 1
                        if min(width, depth) < 3:
                            continue
                        xwall = arm(x, z, 0, dz, depth)
                        zwall = arm(x, z, dx, 0, width)
                        cap = set(xwall + zwall)
                        # A one-block outward shift exposes an additional end cell on each arm.
                        outward = set(arm(x-dx, z-dz, 0, dz, depth+1) +
                                      arm(x-dx, z-dz, dx, 0, width+1))
                        removed, added = set(), set()
                        for y in range(lower+1, upper):
                            row = rows.get(y, set())
                            if (x, z) in row or not {(x-dx, z-dz), (x, z-dz), (x-dx, z)} <= row:
                                break
                            removed.update((a, y, b) for a, b in row & outward)
                            added.update((a, y, b) for a, b in cap - row)
                        else:
                            candidates.append(dict(lower=lower, upper=upper, xwall=xwall, zwall=zwall,
                                                   removed=removed, added=added))
    return candidates


def face_samples(scene, lookup, arm_cells, axis, y):
    samples = []
    for x, z in arm_cells:
        index = lookup[x, y, z]
        pixels = scene.owned.get(index)
        if pixels is not None:
            samples.append(pixels[scene.axes[pixels] == axis])
    return np.concatenate(samples) if samples else np.array([], int)


def recheck(image, parts, camera, cells):
    """Accept a corner revision only when its two faces agree with continuous source masks."""
    if camera is None or camera.get('P') is None or not parts:
        return cells, None
    candidates = proposals(cells)
    if len(candidates) != 1:
        return cells, None
    candidate = candidates[0]
    updated = (cells - candidate['removed']) | candidate['added']
    before = Scene(image, parts, camera, cells)
    after = Scene(image, parts, camera, updated)
    indices = np.arange(len(before.owner))
    gain = before.loss(indices, before.owner, before.axes) - after.loss(indices, after.owner, after.axes)
    # No existing silhouette or face-direction agreement can get worse.
    if np.any(gain < 0) or np.count_nonzero(gain > 0) < 4:
        return cells, None

    lookup = {cell:i for i, cell in enumerate(after.cells)}
    # The source must support each new block, even where one of the arms ends.
    for cell in candidate['added']:
        pixels = after.owned.get(lookup[cell])
        if pixels is None or len(pixels) < 4 or after.masks[after.axes[pixels], pixels].mean() < .90:
            return cells, None
    _, rotation = orientation(camera, image.size)
    masks = defaultdict(list)
    for part in parts:
        normal = part.get('normal_camera')
        if normal is None or not part.get('support'):
            continue
        axis = np.argmax(np.abs(rotation.T @ normal))
        if axis in (0, 2):
            mask = native_mask(part, image.size)[::STEP, ::STEP].ravel()
            masks[axis].append((part['id'], mask))
    sponsors = []
    for axis, line in ((0, candidate['xwall']), (2, candidate['zwall'])):
        samples = [face_samples(after, lookup, line, axis, y)
                   for y in range(candidate['lower'], candidate['upper'] + 1)]
        if any(len(pixels) < 4 for pixels in samples):
            return cells, None
        sponsor = next((name for name, mask in masks[axis]
                        if all(mask[pixels].mean() >= .90 for pixels in samples)), None)
        if sponsor is None:
            return cells, None
        sponsors.append(sponsor)
    return updated, dict(wall_masks=sponsors, improved_samples=int(np.count_nonzero(gain > 0)),
                         added=sorted(candidate['added']), removed=sorted(candidate['removed']))
