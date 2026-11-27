"""Propose visible wall depths from observed contacts between the build and floor."""
from engine import floating_grid_policy as floating
import numpy as np

from engine.plane_offset_features import features
from engine.surface_model import intrinsic
from engine.visible_raster import observed_support


class JunctionDepth:
    """Track visible floor contacts and selected walls for a single photograph."""
    def __init__(self, size, parts):
        self.size = size
        self.parts = parts
        self.anchors = None
        self.walls = []

    def prepare(self, P, eye, rays):
        if floating.active():
            self.anchors = {}
            self.ground_points = []
            return
        shape = rays.shape[:2]
        masks = [observed_support(p['support'], shape, self.size) if p.get('support')
                 else np.zeros(shape, bool) for p in self.parts]
        yy = np.arange(shape[0])[:, None]
        union = np.logical_or.reduce(masks)
        # Only a column's lowest visible build pixel can establish a floor contact.
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
            levels = self.crease_levels(mask, eye, rays)
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
            if (height < 0 and not floating.active()) or np.quantile(abs(heights-height), .8) > .23:
                continue
            votes.setdefault(height, set()).add((axis,level))
        self.votes = votes
        return votes.keys()

    def choose(self, part, mask, P, eye, rays, axis, baseline, scene, rows, weights, *, chosen_before):
        previous = next((r for r in rows if r['level'] == chosen_before), None)
        if previous is None:
            return chosen_before
        if axis == 1 and len(self.votes) == 1 and previous['atlas'] < .60:
            height, anchors = next(iter(self.votes.items()))
            perpendicular = len({wall_axis for wall_axis,_ in anchors}) == 2
            if perpendicular and any(r['level'] == height for r in rows):
                return height
        # Outside the usual search window, require several source tiles to fit
        # the calibrated atlas at the same integer plane. The geometric anchor
        # supplies candidates, never additional blocks.
        scene_level = round(float(eye[axis]+np.median(rays[mask,axis])*scene))
        promoted = [r for r in rows if r['level'] in self.proposed
                    and min(abs(r['level']-baseline), abs(r['level']-scene_level)) > 6 and r['distinct_votes'] >= 2
                    and r['atlas'] > max(.65, previous['atlas'] + .12) and r['edge'] > .08]
        if promoted:
            def visual_score(row):
                # First nine features measure the photographed texture and outline.
                return features([row], baseline)[0,:9] @ weights[:9]
            best = max(promoted, key=visual_score)
            if visual_score(best) > visual_score(previous) + 1.:
                return best['level']
        anchor = self.anchors.get(part['id'])
        if anchor is not None:
            row = next((r for r in rows if r['level'] == anchor[1]), None)
            weak_art = previous['distinct_votes'] < 2 or previous['atlas'] < .60
            if (row is not None and weak_art and row['edge'] > .10
                    and row['edge'] > previous['edge']+.04):
                return row['level']
        return chosen_before
