"""Rank existing observed plane candidates with pairwise foreground ownership evidence."""
from dataclasses import dataclass
from functools import partial

import numpy as np

from engine.plane_repeat_evidence import feature_rows


@dataclass
class Plane:
    mask: np.ndarray
    axis: int
    base: int
    options: dict


def border(mask):
    """Return a three-sample exterior halo, with no wrapping at image boundaries."""
    near = mask.copy()
    for _ in range(3):
        grow = near.copy()
        grow[:-1] |= near[1:]
        grow[1:] |= near[:-1]
        grow[:, :-1] |= near[:, 1:]
        grow[:, 1:] |= near[:, :-1]
        near = grow
    return near & ~mask


def _tile_keys(uv):
    return uv[:, 0].astype(np.int64) * (1 << 31) + uv[:, 1]


def first_hit(mask_rays, probes, eye, axis, level):
    """Trace the incident ray through the observed candidate's one-cell slab.

    Only tiles with at least four observed face samples can occlude another mask.
    Return the first hit before the back of the slab for each competing ray.
    """
    free = [i for i in range(3) if i != axis]
    normal = mask_rays[:, axis]
    own_depth = np.divide(level-eye[axis], normal, out=np.zeros(len(normal)), where=abs(normal)>1e-9)
    valid = (own_depth > 1.5) & (own_depth < 150.)
    if not valid.any():
        return np.full(len(probes), np.inf)
    uv = np.floor(eye[free] + (own_depth[valid, None]*mask_rays[valid])[:, free]).astype(int)
    occupied, n = np.unique(_tile_keys(uv), return_counts=True)
    occupied = occupied[n >= 4]
    coord = probes[:, axis]
    near = np.divide(level-eye[axis], coord, out=np.full(len(probes), np.inf), where=abs(coord)>1e-9)
    far = near + 1/np.maximum(abs(coord), 1e-9)
    valid = (near>1.5) & (near < far)
    hits = np.full(len(probes), np.inf)
    # These probes follow the candidate's camera-facing plane and its observed tile sidewalls.
    pos = np.flatnonzero(valid)
    depth = near[pos] + 1e-6
    for _ in range(16):
        if not len(pos):
            break
        d = probes[pos][:, free]
        uv = eye[free] + depth[:, None]*d
        tile = np.floor(uv).astype(int)
        found = np.isin(_tile_keys(tile), occupied)
        hits[pos[found]] = depth[found]
        edge = tile + (d >= 0)
        steps = np.divide(edge-uv, d, out=np.full_like(d, np.inf), where=abs(d)>1e-9)
        depth = depth + np.min(steps, axis=1) + 1e-6
        keep = ~found & (depth < far[pos])
        pos, depth = pos[keep], depth[keep]
    return hits


class ForegroundConflicts:
    def __init__(self, strength=1.6, threshold=.50):
        self.strength, self.threshold = strength, threshold
        self.planes = {}
        self.levels = {}

    def record(self, key, mask, eye, rays, axis, baseline, rows, chosen, weights):
        if not rows:
            return
        self.eye, self.rays = eye, rays
        scores = feature_rows(rows, baseline) @ weights
        if not np.isfinite(scores).all():
            return
        best = set(np.argsort(scores)[-4:])
        base_idx = next((i for i, r in enumerate(rows) if r['level']==chosen), None)
        if base_idx is None:
            return
        best.add(base_idx)
        options = {rows[i]['level']: (float(scores[i]-scores[base_idx]), rows[i]['atlas']) for i in best}
        self.planes[key] = Plane(mask, axis, chosen, options)

    def resolve(self):
        # Directed pair costs measure foreground theft. Re-evaluate both directions
        # after a neighbor moves, with the atlas/repeat score as the unary term.
        levels = {k: p.base for k, p in self.planes.items()}
        pairs = {}
        for key, p in self.planes.items():
            if len(p.options) < 2:
                continue
            near = border(p.mask)
            for other, q in self.planes.items():
                if key == other:
                    continue
                ids = np.flatnonzero(near & q.mask)
                if len(ids) < 8:
                    continue
                indices = ids[np.linspace(0, len(ids)-1, min(128, len(ids)), dtype=int)]
                probes = self.rays.reshape(-1, 3)[indices]
                own = self.rays[p.mask]
                for proposed in p.options:
                    # Trace once using a far destination, then check each competing
                    # plane depth using the actual stop distance of the first hit.
                    hit = first_hit(own, probes, self.eye, p.axis, proposed)
                    for target in q.options:
                        den = probes[:,q.axis]
                        depth = np.divide(target-self.eye[q.axis], den, out=np.zeros(len(probes)), where=abs(den)>1e-9)
                        active = (depth > 1.5) & (depth < 150)
                        if active.sum() < 8:
                            continue
                        wrong = hit[active] < depth[active]-.12
                        confidence = .35 + .65*np.clip(q.options[target][1], 0., 1.)
                        pairs[key, other, proposed, target] = float(wrong.mean()*confidence*min(1.,len(ids)/30))
        for _ in range(2):
            changed = False
            for key, p in self.planes.items():
                def score(level):
                    wrong = sum(pairs.get((key, j, level, cur), 0.) +
                                pairs.get((j, key, cur, level), 0.) for j, cur in levels.items() if j != key)
                    return min(0., p.options[level][0]) - self.strength*wrong - self.threshold*(level!=p.base)
                current = levels[key]
                winner = max(p.options, key=score)
                if score(winner) > score(current)+1e-6:
                    levels[key] = winner
                    changed = True
            if not changed:
                break
        self.levels.update(levels)
        return levels


def run(strength, threshold, image, version, parts, counts, camera):
    from engine.atlas_plane_depth import _place
    return _place(image, version, parts, counts, camera, None, remove_occluded=True,
                  scene_policy=ForegroundConflicts(strength, threshold))


conservative = partial(run, 1.6, .50)
balanced = partial(run, 3.2, .28)
assertive = partial(run, 5.0, .15)
