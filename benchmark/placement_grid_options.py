"""Private quad choices at the two placement consumers of visible_placement._charts."""
from contextlib import contextmanager
from hashlib import sha256
import json
from itertools import product
from unittest.mock import patch

import numpy as np

from engine import counted_ground_corner, texture_guided_count, visible_placement
from engine.surface_sample import sample_surface
from engine.visible_grid import _carriers
from engine.visible_raster import SAMPLE_STRIDE, observed_support


def boundary_tree(part, size):
    support = part['support']
    width, height = support['size']
    mask = np.zeros((height, width), bool)
    for y, start, stop in support['runs']:
        mask[y, start:stop] = True
    padded = np.pad(mask, 1)
    interior = np.logical_and.reduce([padded[dy:dy+height, dx:dx+width]
                                      for dy in range(3) for dx in range(3)])
    y, x = np.nonzero(mask & ~interior)
    return (np.c_[x, y] + .5) * size / np.array([width, height])


def corner_loss(quad, boundary, neighbors):
    # Probe only the outer tenth of each edge, excluded from the envelope's
    # central fit. Compare both endpoints when asking another chart for a seam.
    fractions = np.array([.02, .06, .10, .90, .94, .98])
    ends = np.roll(quad, -1, axis=0)
    points = quad[:, None] + fractions[None, :, None] * (ends - quad)[:, None]
    distance = [np.linalg.norm(boundary - p, axis=1).min() for p in points.reshape(-1, 2)]
    adjacency = []
    for index in range(4):
        first, last = quad[index], ends[index]
        residual = 8.
        for neighbor in neighbors:
            for start, stop in zip(neighbor, np.roll(neighbor, -1, axis=0)):
                for a, b in ((start, stop), (stop, start)):
                    residual = min(residual, max(np.linalg.norm(first-a), np.linalg.norm(last-b)))
        adjacency.append(residual)
    return dict(boundary=float(np.mean(np.minimum(distance, 12.))), adjoining=float(min(adjacency)))


def choices(image, parts, counts):
    rows = {r['id']: r for r in counts['surfaces']}
    options = []
    for part in parts:
        key = part['id']
        row = rows.get(key, {})
        if row.get('method') != texture_guided_count.METHOD:
            continue
        method = row['evidence']['texture_check']['chart']
        if method == 'original':
            continue
        accepted = np.array(row['quad'])
        original_sample = sample_surface(image, part)
        if original_sample is None:
            continue
        original = np.array(original_sample['quad'])
        if np.allclose(original, accepted):
            continue
        # Ensure a private context/measurement region has the same UV axes as
        # the counter's input by reproducing its named accepted proposal.
        axis = int(original_sample['mask'].shape[0] > original_sample['mask'].shape[1])
        verified = next((s for n, s in texture_guided_count.chart_options(image, part, original_sample, axis)
                         if n == method), None)
        if verified is None or not np.allclose(verified['quad'], accepted, atol=.001, rtol=0):
            continue
        boundary = boundary_tree(part, image.size)
        neighbors = [np.array(r['quad']) for r in rows.values() if r['id'] != key and r.get('quad') is not None]
        old, new = (corner_loss(q, boundary, neighbors) for q in (accepted, original))
        # One native pixel avoids unstable decisions from the finite mask grid.
        # Equal weights in native pixels, no trade against a worse nearby seam.
        source = (new['boundary'] + new['adjoining'] + 1 < old['boundary'] + old['adjoining']
                  and new['adjoining'] <= old['adjoining'])
        options.append(dict(id=key, method=method, counts=row['counts'], phase=row['phase'],
                            dimensions=row['dimensions'], accepted=accepted.tolist(), original=original.tolist(),
                            accepted_loss=old, original_loss=new, source=bool(source)))
    return options


def replace(charts, replacement):
    return [dict(c, quad=replacement[c['id']]) if c['id'] in replacement else c for c in charts]


def distinct_options(image, parts, counts, camera, options):
    charts = visible_placement._charts(image, '26.3', parts, counts)
    projection = np.array(camera['P'])
    shape = tuple((n + (SAMPLE_STRIDE - 1)//2)//SAMPLE_STRIDE for n in image.size[::-1])
    masks = [observed_support(p['support'], shape, image.size) if p.get('support') else np.zeros(shape, bool)
             for p in parts]
    groups = {}
    for bits in product((False, True), repeat=len(options)):
        selection = tuple(i for i, bit in enumerate(bits) if bit)
        override = {options[i]['id']: options[i]['original'] for i in selection}
        selected = replace(charts, override)
        carriers = _carriers(image, selected, masks, projection)
        planes = tuple((plane, status) for _, plane, status in carriers)
        anchors = counted_ground_corner.grounded_pairs(image, selected, parts, masks, projection)
        signature = planes, tuple(sorted(anchors.items()))
        groups.setdefault(signature, []).append(selection)
    return list(groups.values())


@contextmanager
def private_charts(counts, replacement):
    ordinary = visible_placement._charts
    before = sha256(json.dumps(counts, sort_keys=True).encode()).digest()
    calls = []

    def with_choices(image, version, parts, handoff):
        charts = ordinary(image, version, parts, handoff)
        if handoff is not counts:
            return charts
        private = replace(charts, replacement)
        assert [c['pitch'] for c in charts] == [c['pitch'] for c in private]
        calls.append(True)
        return private

    with patch.object(visible_placement, '_charts', with_choices):
        yield
    assert calls, 'private charts were never consumed'
    assert sha256(json.dumps(counts, sort_keys=True).encode()).digest() == before
