"""Fit a union of integer cuboids by rendering their silhouettes into one view."""
from itertools import product

import numpy as np
from PIL import Image, ImageFilter


CORNERS = np.array(list(product((0, 1), repeat=3)), dtype=int)
EDGES = [(i, i ^ bit) for bit in (4, 2, 1) for i in range(8) if not i & bit]


def _project(P, xyz):
    uvw = xyz @ P[:, :3].T + P[:, 3]
    return uvw[..., :2] / np.maximum(uvw[..., 2:], 1.e-7)


def _prefix(array, axis):
    pad = [(0, 0)] * array.ndim
    pad[axis] = (1, 0)
    return np.pad(array.cumsum(axis=axis), pad)


def _seeds(hull):
    points = np.argwhere(hull)
    low, high = points.min(0), points.max(0) + 1
    def windows(a, b):
        span = b-a
        units = [(0,1), (0,.5), (.5,1), (0,.25), (.75,1), (.25,.75)]
        return sorted({(a+round(u*span), a+round(v*span)) for u,v in units if round(v*span)>round(u*span)})
    # Include thin strata and posts, which whole-building proposals can skip.
    intervals = [windows(a,b) for a,b in zip(low,high)]
    for axis in range(3):
        intervals[axis].extend([(int(low[axis]), int(low[axis]+1)),
                                (int(high[axis]-1), int(high[axis]))])
    seeds = set()
    for x,y,z in product(*intervals):
        seeds.add((x[0],y[0],z[0],x[1],y[1],z[1]))
    # Cuboids tightly bounded by each cross section expose abrupt setbacks.
    for axis in range(3):
        for a,b in intervals[axis]:
            sliced = points[(points[:,axis]>=a) & (points[:,axis]<b)]
            if len(sliced):
                seeds.add(tuple(np.r_[sliced.min(0), sliced.max(0)+1]))
    return np.array(sorted(seeds))


class _View:
    def __init__(self, scene):
        self.mask = np.asarray(scene['mask'], bool)
        self.P = np.asarray(scene['P'], float)
        self.hull = np.asarray(scene['hull'], bool)
        try:
            self.eye = np.linalg.solve(self.P[:,:3], -self.P[:,3])
        except np.linalg.LinAlgError:
            self.eye = np.full(3, np.nan)
        integral = self.hull.astype(np.int64)
        for axis in range(3):
            integral = _prefix(integral, axis)
        self.integral = integral
        image = Image.fromarray(np.asarray(scene['image'], np.uint8)).convert('L')
        self.gray = np.asarray(image.filter(ImageFilter.GaussianBlur(.8)), float)
        self.rendered = np.zeros_like(self.mask)
        self.grid = np.zeros_like(self.hull)
        self.height, self.width = self.mask.shape
        self.area = max(1, self.mask.sum())
        self.ratio = self.area / max(1, self.hull.sum())
        self.rows = np.arange(self.height)[None,:]

    def spans(self, uv):
        """Scan-convert the projected convex box in batches using its 12 edges."""
        yy = self.rows + .5
        left = np.full((len(uv),self.height), np.inf)
        right = np.full_like(left,-np.inf)
        for i,j in EDGES:
            a,b = uv[:,i],uv[:,j]
            y0,y1 = a[:,1,None],b[:,1,None]
            crossing = (yy>=np.minimum(y0,y1)) & (yy<np.maximum(y0,y1))
            fraction = (yy-y0) / np.where(abs(y1-y0)>1.e-9, y1-y0, 1.)
            x = a[:,0,None] + fraction*(b-a)[:,0,None]
            left = np.where(crossing, np.minimum(left,x), left)
            right = np.where(crossing, np.maximum(right,x), right)
        left = np.ceil(left-.5).clip(0,self.width).astype(int)
        right = np.ceil(right-.5).clip(0,self.width).astype(int)
        return np.minimum(left,right),right

    def creases(self, boxes, uv):
        """Contrast across the three edges shared by visible faces of a box."""
        if not np.isfinite(self.eye).all():
            return np.zeros(len(boxes))
        low,high = boxes[:,:3],boxes[:,3:]
        side = self.eye > high
        visible = side | (self.eye < low)
        corner = side @ np.array([4,2,1])
        index = np.arange(len(boxes))
        start = uv[index,corner]
        total = np.zeros(len(boxes))
        for axis,bit in enumerate((4,2,1)):
            delta = uv[index,corner^bit] - start
            length = np.linalg.norm(delta,axis=1)
            normal = np.stack((-delta[:,1],delta[:,0]),axis=1) / np.maximum(length[:,None],.001)
            along = start[:,None] + np.linspace(.11,.89,15)[None,:,None]*delta[:,None]
            def sample(distance):
                xy = np.floor(along + distance*normal[:,None]).astype(int)
                return self.gray[xy[...,1].clip(0,self.height-1),xy[...,0].clip(0,self.width-1)]
            # A smooth gradient across one planar face offers little crease evidence.
            jump = sample(1.5)-sample(-1.5)
            drift = .5*((sample(4.5)-sample(1.5))+(sample(-1.5)-sample(-4.5)))
            contrast = np.abs(np.median(jump-drift,axis=1))
            axes = [a for a in range(3) if a!=axis]
            valid = visible[:,axes].all(1) & (length>4)
            total += np.where(valid, length*np.clip((contrast-4.)/18.,-.2,1.), 0.)
        return total

    def prepare(self):
        # Reward covering unexplained foreground, charge new spill past the contour.
        self.reward = _prefix(np.where(self.rendered,0.,np.where(self.mask,1.,-1.8)),1)

    def evaluate(self, boxes):
        low,high = boxes[:,:3],boxes[:,3:]
        vertices = np.where(CORNERS[None],high[:,None],low[:,None])
        uv = _project(self.P, vertices)
        left,right = self.spans(uv)
        gain = (self.reward[self.rows,right] - self.reward[self.rows,left]).sum(1)
        if self.grid.any():
            # Internal edges of another primitive can be hidden by existing faces.
            # Give no crease bonus to primitives with negligible new visible area.
            visibility = np.clip(gain / max(8,.012*self.area),0,1)
        else:
            visibility = np.ones(len(boxes))
        gain += visibility * self.creases(boxes,uv)
        solid = np.zeros(len(boxes),int)
        for k,corner in enumerate(CORNERS):
            idx = vertices[:,k].T
            solid += (-1 if (3-corner.sum())%2 else 1)*self.integral[tuple(idx)]
        volume = np.prod(high-low,axis=1)
        gain -= self.ratio*(.14*volume + 1.3*(volume-solid))
        return np.where(solid>=.63*volume,gain,-np.inf)

    def score(self, boxes):
        return np.concatenate([self.evaluate(batch) for batch in np.array_split(boxes,(len(boxes)+95)//96)])

    def add(self, box):
        low,high = box[:3],box[3:]
        self.grid[tuple(slice(a,b) for a,b in zip(low,high))] = True
        vertices = np.where(CORNERS,high,low)
        left,right = self.spans(_project(self.P,vertices)[None])
        for y,(a,b) in enumerate(zip(left[0],right[0])):
            self.rendered[y,a:b] = True


def predict(scene):
    """Greedy integer cuboid union with coarse proposals and one-block refinement."""
    view = _View(scene)
    if not view.hull.any() or not view.mask.any():
        return view.grid
    seeds = _seeds(view.hull)
    for _ in range(10):
        view.prepare()
        scores = view.score(seeds)
        order = np.argsort(-scores,kind='stable')[:4]
        candidates = seeds[order]
        for step in (4,2,1):
            for _ in range(3):
                shifts = np.vstack([np.zeros((1,6),int),np.eye(6,dtype=int)*step,
                                    -np.eye(6,dtype=int)*step])
                options = np.unique((candidates[:,None]+shifts).reshape(-1,6),axis=0)
                low,high = options[:,:3],options[:,3:]
                valid = ((low>=0)&(high<=view.hull.shape)&(high>low)).all(1)
                choices = options[valid]
                scores = view.score(choices)
                order = np.argsort(-scores,kind='stable')[:4]
                improved = choices[order]
                if np.array_equal(candidates,improved):
                    break
                candidates = improved
        if scores[order[0]] < max(5.,.006*view.area):
            break
        view.add(candidates[0])
    return view.grid
