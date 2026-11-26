"""Propose visible wall depths from observed contacts between the build and floor."""
import numpy as np

from engine.plane_offset_features import features, ranked_choice
from engine.surface_model import intrinsic
from engine.visible_raster import observed_support, SAMPLE_STRIDE


class JunctionDepth:
    def __init__(self, size, parts, policy):
        self.size = size
        self.parts = parts
        self.policy = policy
        self.anchors = None
        self.walls = []

    def prepare(self, P, eye, rays):
        shape = rays.shape[:2]
        masks = [observed_support(p['support'], shape, self.size) if p.get('support')
                 else np.zeros(shape, bool) for p in self.parts]
        yy = np.arange(shape[0])[:, None]
        union = np.logical_or.reduce(masks)
        bottom = union & (yy == np.where(union, yy, -1).max(0)) & (yy < shape[0]-1)
        foot = np.divide(-eye[1], rays[:,:,1], out=np.zeros(shape), where=rays[:,:,1] < -1.e-6)
        focal = np.linalg.norm(P[0,:3] - self.size[0]*.5*P[2,:3])
        rotation = np.linalg.solve(intrinsic(self.size, focal), P[:,:3])
        self.anchors = {}
        self.ground_points = []
        for part, mask in zip(self.parts, masks, strict=True):
            if part.get('normal_camera') is None:
                continue
            axis = int(np.argmax(abs(rotation.T @ part['normal_camera'])))
            contact = mask & bottom & (foot > 1.5) & (foot < 100)
            if axis == 1 or contact.sum() < max(5, .15*mask.any(0).sum()):
                continue
            points = eye + foot[contact,None] * rays[contact]
            level = round(float(np.median(points[:,axis])))
            if np.quantile(abs(points[:,axis]-level), .80) > .32:
                continue
            self.anchors[part['id']] = (axis, level)
            self.ground_points.append(np.median(points, axis=0))

    def levels(self, mask, P, eye, rays, axis):
        if self.anchors is None:
            self.prepare(P, eye, rays)
        if axis == 1:
            levels = self.crease_levels(mask, eye, rays) if self.policy in ('creases','single','supported') else ()
        else:
            levels = ()
            if len(self.anchors) >= 2:
                q = np.quantile(np.array(self.ground_points)[:,axis], [.2,.8])
                if q[1]-q[0] <= 30:
                    levels = range(int(np.floor(q[0]))-6, int(np.ceil(q[1]))+7)
        self.proposed = set(levels)
        return self.proposed

    def record(self, part, mask, axis, level, rows):
        if axis == 1:
            return
        chosen = next((r for r in rows if r['level'] == level), None)
        grounded = self.anchors.get(part['id']) == (axis, level)
        if chosen and (grounded or (chosen['atlas'] >= .68 and chosen['distinct_votes'] >= 2)):
            self.walls.append((mask,axis,level))

    def crease_levels(self, mask, eye, rays):
        votes = {}
        for wall, axis, level in self.walls:
            # Search up to two sampling steps for a vertical mask across an edge.
            # Slices avoid joining unrelated masks across the viewport border.
            near = wall.copy()
            for d in (1,2):
                near[d:] |= wall[:-d]
                near[:-d] |= wall[d:]
                near[:,d:] |= wall[:,:-d]
                near[:,:-d] |= wall[:,d:]
            edge = mask & near & ~wall & (abs(rays[:,:,axis]) > .05)
            y,x = np.nonzero(edge)
            if len(x) < 8 or max(np.ptp(x),np.ptp(y)) < 6:
                continue
            heights = eye[1]+(level-eye[axis])*rays[edge,1]/rays[edge,axis]
            height = round(float(np.median(heights)))
            if height < 0 or np.quantile(abs(heights-height), .8) > .23:
                continue
            votes.setdefault(height, set()).add((axis,level))
        self.votes = votes
        return votes.keys()

    def choose(self, part, mask, P, eye, rays, axis, baseline, scene, rows, weights, *, chosen_before=None):
        scene_level = round(float(eye[axis]+np.median(rays[mask,axis])*scene))
        original = [r for r in rows if min(abs(r['level']-baseline), abs(r['level']-scene_level)) <= 6]
        before = ranked_choice(original, baseline, weights) if chosen_before is None else chosen_before
        if not rows:
            return before
        previous = next((r for r in rows if r['level'] == before), None)
        if previous is None:
            return before
        if axis == 1 and self.policy in ('creases','single','supported'):
            votes = self.votes
            if len(votes) == 1:
                height, anchors = next(iter(votes.items()))
                proposed = next((r for r in rows if r['level'] == height), None)
                if proposed and previous['atlas'] < .60:
                    corroborated = len({axis for axis,_ in anchors}) == 2
                    photo = (proposed['edge'] > max(.05,previous['edge']+.05)
                             or (proposed['distinct_votes'] >= 2 and proposed['atlas'] > previous['atlas']+.1)
                             or proposed['boundary'] > max(.55,previous['boundary']+.12))
                    if corroborated or self.policy == 'single' or (self.policy == 'supported' and photo):
                        return height
        # Outside the usual search window, require several source tiles to fit
        # the calibrated atlas at the same integer plane. The geometric anchor
        # supplies candidates, never additional blocks.
        usual = {r['level'] for r in original}
        promoted = [r for r in rows if r['level'] in self.proposed and r['level'] not in usual and r['distinct_votes'] >= 2
                    and r['atlas'] > max(.65, previous['atlas'] + .12) and r['edge'] > .08]
        if promoted:
            best = max(promoted, key=lambda r: features([r], baseline)[0,:9] @ weights[:9])
            if features([best], baseline)[0,:9] @ weights[:9] > features([previous], baseline)[0,:9] @ weights[:9]+1.:
                return best['level']
        if self.policy == 'art':
            return before
        anchor = self.anchors.get(part['id'])
        if anchor is not None:
            row = next((r for r in rows if r['level'] == anchor[1]), None)
            weak_art = previous['distinct_votes'] < 2 or (self.policy in ('creases','single','supported') and previous['atlas'] < .60)
            if (row is not None and weak_art and row['edge'] > .10
                    and row['edge'] > previous['edge']+.04):
                return row['level']
        return before


def _run(image,version,parts,counts,camera,policy):
    from engine.atlas_plane_depth import _place
    return _place(image,version,parts,counts,camera,None,remove_occluded=True,junction=policy)


def art(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,'art')


def contacts(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,'contacts')


def creases(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,'creases')


def single(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,'single')


def supported(image,version,parts,counts,camera):
    return _run(image,version,parts,counts,camera,'supported')
