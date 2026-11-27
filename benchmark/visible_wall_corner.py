"""Source-only common-corner trial over integer wall-plane depths."""
from itertools import combinations

import numpy as np

from engine.visible_raster import SAMPLE_STRIDE


def shared_boundary(left, right):
    """One facing run boundary per row, with up to two unassigned samples."""
    measurements = []
    for y in np.flatnonzero(left.any(1) & right.any(1)):
        a, b = np.flatnonzero(left[y]), np.flatnonzero(right[y])
        for ends in ((a, b), (b, a)):
            before, after = ends
            gap = after[0] - before[-1]
            if 1 <= gap <= 3 and len(before) >= 3 and len(after) >= 3:
                measurements.append(((before[-1] + after[0] + 1) / 2, y + .5))
                break
    if len(measurements) < 9:
        return None
    points = np.array(measurements)
    segments = np.split(points, np.flatnonzero(np.diff(points[:, 1]) > 2) + 1)
    longest = max(segments, key=len)
    first, last = int(longest[0, 1]), int(longest[-1, 1])
    # If either wall extends beyond both ends, its continuous face can be an
    # occluder in a T junction. Require each mask to end at one end of the seam.
    for mask in (left, right):
        rows = np.flatnonzero(mask.any(1))
        if min(first-rows[0], rows[-1]-last) > 4:
            return None
    if len(longest) < 9 or len(longest) / (last-first+1) < .8:
        return None
    return longest * SAMPLE_STRIDE


def ground_anchor(mask, union, eye, rays, axis, level):
    yy = np.arange(mask.shape[0])[:, None]
    foot = mask & (yy == np.where(union, yy, -1).max(0)) & (yy < mask.shape[0]-1)
    selected = rays[foot & (rays[:, :, 1] < -.001)]
    if len(selected) < max(5, .15 * mask.any(0).sum()):
        return False
    depths = -eye[1] / selected[:, 1]
    return np.quantile(abs(eye[axis] + depths * selected[:, axis] - level), .8) <= .32


def error(points, projection, axis, level, other):
    corner = np.array([level, 0., other] if axis == 0 else [other, 0., level])
    base = projection[:, :3] @ corner + projection[:, 3]
    line = np.cross(base, projection[:, 1])
    norm = np.linalg.norm(line[:2])
    if norm < 1e-8:
        return np.inf
    return float(np.quantile(abs(points @ line[:2] + line[2]) / norm, .8))


def recommend(policy, levels, projection, union):
    walls = [key for key, p in policy.planes.items() if p.axis != 1]
    anchors = {}
    reasons = {}
    offers = []
    seams = []
    for key in walls:
        plane, level = policy.planes[key], levels[key]
        row = policy.source_rows[key][level]
        anchors[key] = (ground_anchor(plane.mask, union, policy.eye, policy.rays, plane.axis, level)
                        or (row['atlas'] >= .68 and row['distinct_votes'] >= 2))
    for left, right in combinations(walls, 2):
        p, q = policy.planes[left], policy.planes[right]
        if p.axis == q.axis:
            continue
        points = shared_boundary(p.mask, q.mask)
        if points is None:
            continue
        first, second = levels[left], levels[right]
        incumbent = error(points, projection, p.axis, first, second)
        record = dict(pair=[left, right], axis=p.axis, levels=[first, second], rows=len(points),
                      seam_px=points.tolist(), incumbent_error_px=incumbent,
                      anchors=[bool(anchors[left]), bool(anchors[right])])
        seams.append(record)
        if anchors[left] == anchors[right]:
            record['declined'] = 'two anchors' if anchors[left] else 'no anchor'
            continue
        target, fixed = (right, left) if anchors[left] else (left, right)
        t = policy.planes[target]
        options = []
        for new in policy.source_rows[target]:
            if not 1 <= abs(new-levels[target]) <= 2:
                continue
            e = error(points, projection, t.axis, new, levels[fixed])
            options.append((e, int(new)))
        options.sort()
        if not options:
            record['declined'] = 'no local option'
            continue
        best, new = options[0]
        record.update(proposed_level=new, proposed_error_px=best, target=target)
        if best > 5 or incumbent-best < 5 or (len(options)>1 and options[1][0]-best < 2):
            record['declined'] = 'weak or ambiguous reprojection'
            continue
        offers.append((incumbent-best, target, new))
    offers.sort(reverse=True)
    # One changed wall per frame bounds the chooser to the incumbent and a
    # single challenger, and prevents correlated shifts among several pairs.
    chosen = {offers[0][1]: offers[0][2]} if offers else {}
    return chosen, dict(walls=len(walls), seams=seams, changes=chosen)

