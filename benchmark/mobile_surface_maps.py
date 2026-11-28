"""Render recorded visible surfaces and compare the saved Finder masks at source resolution."""
import numpy as np
from PIL import Image

from benchmark.floating_stage_report import hash_file, read


def visible_edges(ids):
    """Trace native boundaries on pixels owned by visible labels."""
    keep = ids != 0
    borders = np.zeros_like(keep)
    dx = ids[:, 1:] != ids[:, :-1]
    dy = ids[1:, :] != ids[:-1, :]
    borders[:, 1:] |= dx & keep[:, 1:]
    borders[:, :-1] |= dx & keep[:, :-1]
    borders[1:, :] |= dy & keep[1:, :]
    borders[:-1, :] |= dy & keep[:-1, :]
    return borders


def labels_by_pixel(labels, owner):
    counts = np.bincount(owner.ravel(), minlength=1)
    table = np.zeros(len(counts), dtype=np.int32)
    visible_ids = np.flatnonzero(counts[1:]) + 1
    faces = {face['id']: face for face in labels['faces']}
    if set(visible_ids) != faces.keys() or any(
            counts[i] != faces[i]['visible_pixels'] or faces[i]['surface_id'] <= 0 for i in visible_ids):
        raise ValueError('Recorded native face owners differ from the visible surface labels.')
    for i in visible_ids:
        table[i] = faces[i]['surface_id']
    return table[owner]


def tinted(rgb, mask, color, weight):
    rgb[mask] = np.rint(rgb[mask] * (1 - weight) + np.asarray(color) * weight)


def maps(image, frame, proof, data=None, handoffs=()):
    from engine.surface_perimeter import _native_alpha

    labels_path = frame / 'labels.json'
    if not labels_path.is_file():
        return [], None
    if hash_file(labels_path) != proof['labels.json']:
        raise ValueError(f'Visible surface labels differ from the scored frame: {frame}')
    with np.load(frame / 'labels.npz', allow_pickle=False) as channels:
        owner = channels['face_id']
    if owner.shape != image.size[::-1]:
        raise ValueError(f'Native reference dimensions differ: {frame}')
    gold = labels_by_pixel(read(labels_path), owner)
    gold_pixels = np.bincount(gold.ravel())
    visible = gold > 0
    rgb = np.asarray(image)
    reference = rgb.copy()
    palette = ((249, 212, 102), (246, 175, 83), (251, 226, 149), (235, 192, 112))
    for i, color in enumerate(palette):
        tinted(reference, visible & (gold % len(palette) == i), color, .34)
    reference[visible_edges(gold)] = (249, 212, 102)
    layers = [('surface-reference', 'Perfect surfaces',
               'Gold shows recorded surface boundaries and their first-hit pixels in this source image.',
               Image.fromarray(reference))]
    if not data:
        return layers, None
    provenance = data['provenance']
    expected = provenance.get('surfaceSha256') or provenance['arms']['Candidate'].get('surfaceSha256')
    handoff = next((p for p in handoffs if p.is_file() and hash_file(p) == expected), None)
    if handoff is None:
        return layers, None
    by_id = {p['id']:p for p in read(handoff)['planes']}
    predicted = np.zeros_like(visible)
    edges = np.zeros_like(visible)
    for row in data['surfaces']:
        if row.get('arm') == 'Baseline':
            continue
        plane = by_id.get(row.get('sourceSurface', row['id']))
        if plane is None:
            continue
        result = _native_alpha(plane['support'], image.size)
        if result is None:
            continue
        alpha, (left, top, right, bottom) = result
        kept = np.asarray(alpha) > 0
        predicted[top:bottom, left:right] |= kept
        inside = np.pad(kept, 1)
        eroded = inside[:-2, 1:-1] & inside[2:, 1:-1] & inside[1:-1, :-2] & inside[1:-1, 2:]
        edges[top:bottom, left:right] |= kept & ~eroded
        overlaps = np.bincount(gold[top:bottom, left:right][kept], minlength=len(gold_pixels))
        pixels = int(kept.sum())
        iou = overlaps[1:] / np.maximum(1, gold_pixels[1:] + pixels - overlaps[1:])
        if len(iou) and iou.max() > 0:
            sid = int(iou.argmax()) + 1
            other = [dict(surfaceId=int(i), pixels=int(overlaps[i])) for i in np.argsort(-overlaps[1:]) + 1
                     if i != sid and overlaps[i]]
            row['maskOverlap'] = dict(overlap=int(overlaps[sid]), reference=int(gold_pixels[sid]),
                                      predicted=pixels, surfaceId=sid, otherSurfaces=other[:3])
    agreement = visible & predicted
    missing = visible & ~predicted
    extra = predicted & ~visible
    comparison = rgb.copy()
    for mask, color in [(agreement, (119, 218, 159)), (missing, (250, 213, 108)), (extra, (246, 116, 174))]:
        tinted(comparison, mask, color, .48)
    comparison[visible_edges(gold)] = (249, 212, 102)
    comparison[edges] = (120, 241, 252)
    layers.append(('surface-compare', 'Mask comparison',
                   'Gold outlines reference surfaces. Cyan outlines saved Finder masks. Green fill is covered, gold is missed, and pink is outside the reference.',
                   Image.fromarray(comparison)))
    return layers, dict(reference=int(visible.sum()), covered=int(agreement.sum()), extra=int(extra.sum()),
                        referenceSurfaces=int(np.count_nonzero(gold_pixels[1:])))
