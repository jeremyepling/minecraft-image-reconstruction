"""Flag smooth visible gaps bracketed by the same Surface Finder mask."""
import numpy as np

from engine.camera_lines import _runs
from engine.surface_perimeter import _native_alpha


def scan(labels, rgb, blocked, limit):
    """Look through unassigned spans with three source pixels of support at each end."""
    votes = np.zeros_like(labels)
    for y, line in enumerate(labels):
        edges = np.flatnonzero(np.diff(np.r_[False, line == 0, False]))
        for start, end in edges.reshape(-1, 2):
            if not 4 <= end-start <= limit or start < 3 or end+3 > len(line):
                continue
            owner = line[start-1]
            if owner <= 0 or (line[start-3:start] != owner).any() or (line[end:end+3] != owner).any():
                continue
            if blocked[y,start:end].any():
                continue
            strip = rgb[y,start-3:end+3]
            if np.abs(np.diff(strip, axis=0)).max() <= 3 and np.ptp(strip, axis=0).max() <= 22:
                votes[y,start:end] = owner
    return votes


def candidates(image, parts):
    """Return local flags plus their proposed native pixels; inputs stay unmodified."""
    raw = [_native_alpha(p['support'], image.size) for p in parts]
    boxes = [item[1] for item in raw if item is not None]
    if not boxes:
        return []
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]
    labels = np.zeros((bottom-top, right-left), np.int32)
    for index, item in enumerate(raw, 1):
        if item is None:
            continue
        alpha, (x,y,r,b) = item
        window = labels[y-top:b-top,x-left:r-left]
        support = np.asarray(alpha, bool)
        window[support] = np.where(window[support] == 0, index, -1)
    rgb = np.asarray(image.convert('RGB'), dtype=np.int16)
    # Both exterior margins must agree before a row supplies a background veto.
    # Smooth sky, clouds and distant pavement can all resemble a pale wall.
    side = max(1, round(image.width*.015))
    sides = np.stack([np.median(rgb[top:bottom, sl], axis=1)
                      for sl in (slice(None,side),slice(-side,None))]).astype(np.int16)
    cropped = rgb[top:bottom,left:right]
    background = ((abs(sides[0]-sides[1]).max(axis=1) <= 6)[:,None] &
                  (abs(cropped-sides[0,:,None]).max(axis=2) <= 8))
    limit = max(12, round(.065*image.width))
    h = scan(labels, cropped, background, limit)
    v = scan(labels.T, cropped.transpose(1,0,2), background.T, limit).T
    voted = np.where((h>0) & (v>0) & (h!=v), 0, np.maximum(h,v))
    events = []
    for owner in np.unique(voted):
        if owner == 0:
            continue
        for spans in _runs(voted==owner):
            area = sum(b-a for _,a,b in spans)
            if area < 32:
                continue
            x0, y0 = min(a for _,a,_ in spans), min(y for y,_,_ in spans)
            x1, y1 = max(b for _,_,b in spans), 1+max(y for y,_,_ in spans)
            if min(x1-x0,y1-y0) < 4:
                continue
            runs = [[int(y+top),int(a+left),int(b+left)] for y,a,b in spans]
            both = sum(np.count_nonzero((h[y,a:b]==owner) & (v[y,a:b]==owner)) for y,a,b in spans)
            events.append(dict(surface=parts[owner-1]['id'], pixels=area, both_axes=int(both),
                               box=[int(x0+left),int(y0+top),int(x1+left),int(y1+top)], runs=runs))
    return events
