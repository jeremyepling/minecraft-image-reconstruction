"""Exact visible outlines and native photo crops from a Surface Finder support mask."""
import numpy as np
from PIL import Image, ImageDraw

from engine.normal_regions import _local_mask


# In image coordinates clockwise rings enclose foreground. Direct each pixel
# edge with foreground on its right so holes naturally get the opposite winding.
_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
DEFAULT_CONTEXT_PX = 8


def _rings(mask, left, top):
    padded = np.pad(mask, 1)
    neighbors = (padded[:-2, 1:-1], padded[1:-1, 2:], padded[2:, 1:-1], padded[1:-1, :-2])
    offsets = ((0, 0), (1, 0), (1, 1), (0, 1))
    edges = {}
    for direction, (neighbor, (ox, oy)) in enumerate(zip(neighbors, offsets)):
        ys, xs = np.nonzero(mask & ~neighbor)
        for x, y in zip((xs + left + ox).tolist(), (ys + top + oy).tolist()):
            edges[x, y, direction] = None

    visited = set()
    for edge in edges:
        if edge in visited:
            continue
        path = []
        positions = {}
        while edge not in visited:
            visited.add(edge)
            x, y, direction = edge
            positions[x, y] = len(path)
            path.append((x, y))
            dx, dy = _DIRECTIONS[direction]
            x, y = x + dx, y + dy
            # At a diagonal touch, keep the two sides of a one-pixel opening
            # distinct. A walk can revisit a vertex if two loops touch there.
            if (x, y) in positions:
                start = positions[x, y]
                loop = path[start:]
                yield _ring(loop)
                for point in loop:
                    del positions[point]
                del path[start:]
            for turn in (1, 0, -1):
                edge = (x, y, (direction + turn) % 4)
                if edge in edges:
                    break


def _ring(loop):
    vertices = np.asarray(loop)
    before = vertices - np.roll(vertices, 1, axis=0)
    after = np.roll(vertices, -1, axis=0) - vertices
    # Removing only collinear points preserves every stair, recess and opening.
    xy = vertices[np.any(before != after, axis=1)]
    x, y = xy.T
    area2 = x @ np.roll(y, -1) - y @ np.roll(x, -1)
    return dict(kind='outer' if area2 > 0 else 'hole', points=xy.tolist())


def outline(support):
    """Closed rings of pixel edges, in the support grid's xy coordinates.

    A point list omits the duplicate closing vertex. An outer ring can lie
    inside a hole of another component. Even-odd filling recovers the mask.
    """
    runs = support['runs']
    rings = list(_rings(*_local_mask(np.asarray(runs)))) if len(runs) else []
    return dict(size=list(support['size']), rings=rings)


def with_outline(part):
    """Derive the physical perimeter from the final support, including legacy proposals."""
    return dict(part, outline=outline(part['support']), sampling_chart=part.get('quad'))


def draw_outline(image, part, color='#49ffff', width=3):
    """Stroke each closed ring in place on the full source image, without joining rings."""
    perimeter = part.get('outline') or outline(part['support'])
    scale = np.asarray(image.size) / perimeter['size']
    draw = ImageDraw.Draw(image)
    for ring in perimeter['rings']:
        xy = np.clip(np.asarray(ring['points']) * scale, 0, np.asarray(image.size) - 1)
        points = list(map(tuple, xy))
        draw.line(points + points[:1], fill=color, width=width)


def _native_alpha(support, size):
    runs = support['runs']
    if not len(runs):
        return None
    mask, left, top = _local_mask(np.asarray(runs))
    sw, sh = support['size']
    w, h = size
    # Integer arithmetic also handles a pixel exactly on a support-grid edge.
    xs = ((2 * np.arange(w) + 1) * sw) // (2 * w)
    ys = ((2 * np.arange(h) + 1) * sh) // (2 * h)
    x0, x1 = map(int, np.searchsorted(xs, [left, left + mask.shape[1]]))
    y0, y1 = map(int, np.searchsorted(ys, [top, top + mask.shape[0]]))
    if x0 == x1 or y0 == y1:
        return None
    inside = mask[np.ix_(ys[y0:y1] - top, xs[x0:x1] - left)]
    alpha = Image.fromarray(inside.astype(np.uint8) * 255)
    bounds = alpha.getbbox()
    if bounds is None:
        return None
    a, b, c, d = bounds
    box = (x0 + a, y0 + b, x0 + c, y0 + d)
    return alpha.crop(bounds), box


def masked_crop(image, part):
    """Return original RGBA pixels and a tight half-open native xyxy box, or None if empty.

    Each native pixel center samples the same half-open support grid as the
    outline. The crop spans the entire support and is independent of any chart.
    """
    result = _native_alpha(part['support'], image.size)
    if result is None:
        return None
    alpha, box = result
    crop = image.crop(box).convert('RGBA')
    crop.putalpha(alpha)
    return crop, box


def context_crop(image, part, margin_px=DEFAULT_CONTEXT_PX, *, mask_box=None):
    """Return (original RGB neighborhood, native xyxy, tight mask xyxy in neighborhood).

    All boxes are half-open. Supply a box from masked_crop when it is already
    available. Neighbors come from the supplied screenshot, including holes.
    """
    if not isinstance(margin_px, int) or margin_px < 0:
        raise ValueError('The context margin must be a nonnegative whole number of source pixels.')
    if mask_box is None:
        result = _native_alpha(part['support'], image.size)
        if result is None:
            return None
        _, mask_box = result
    l, t, r, b = mask_box
    x = max(0, l - margin_px)
    y = max(0, t - margin_px)
    box = (x, y, min(image.width, r + margin_px), min(image.height, b + margin_px))
    return image.crop(box).convert('RGB'), box, (l-x, t-y, r-x, b-y)
