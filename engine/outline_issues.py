"""Locate short, repeated sharp turns in a predicted outer boundary for review."""
import numpy as np

from engine.surface_perimeter import outline


def _simplify_arc(points, tolerance):
    """Return retained vertices in arc order, omitting the shared end vertex."""
    keep = {0}
    pending = [(0, len(points)-1)]
    while pending:
        start, end = pending.pop()
        if end-start <= 1:
            continue
        segment = points[end]-points[start]
        offsets = points[start+1:end]-points[start]
        t = np.clip((offsets @ segment)/max(segment @ segment, 1), 0, 1)
        distances = ((offsets-t[:,None]*segment)**2).sum(axis=1)
        pivot = int(distances.argmax())
        if distances[pivot] > tolerance**2:
            mid = start+1+pivot
            keep.add(mid)
            pending.extend([(start,mid),(mid,end)])
    return points[sorted(keep)]


def _simplify_ring(points):
    if len(points) <= 6:
        return points
    pivot = int(((points-points[0])**2).sum(axis=1).argmax())
    return np.vstack((_simplify_arc(points[:pivot+1],3), _simplify_arc(np.vstack((points[pivot:],points[0])),3)))


def outline_issues(parts, image_size):
    """Report geometry warnings only. Positions are in source-image coordinates."""
    warnings = []
    for part in parts:
        perimeter = part.get('outline') or outline(part['support'])
        scale = np.asarray(image_size)/perimeter['size']
        rings = [_simplify_ring(np.asarray(r['points'])*scale) for r in perimeter['rings'] if r['kind']=='outer']
        if not rings:
            continue
        points = max(rings,key=len)
        edges = points-np.roll(points,1,axis=0)
        previous = np.roll(edges,1,axis=0)
        length = np.linalg.norm(edges,axis=1)
        prior = np.roll(length,1)
        cosine = (edges*previous).sum(axis=1)/np.maximum(length*prior,.01)
        sharp = (length<=18) & (prior<=18) & (cosine<=np.cos(np.pi/4))
        if sharp.sum() >= 8:
            turns = np.roll(points,1,axis=0)[sharp]
            warnings.append(dict(surface=part['id'],issue='repeated_short_turns',native_turn_xy=turns.round(2).tolist()))
    return warnings
