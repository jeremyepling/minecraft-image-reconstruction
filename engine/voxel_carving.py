"""Carve an image-derived voxel hull using its rendered silhouettes and creases."""
import numpy as np
from PIL import Image, ImageFilter


def _rays(projection, xs, ys, shape):
    """Cache intersected unit cells and the entering face for each screen ray."""
    inverse = np.linalg.inv(projection[:, :3])
    eye = -inverse @ projection[:, 3]
    pixel = np.column_stack((xs.ravel()+.5, ys.ravel()+.5, np.ones(xs.size)))
    ray = pixel @ inverse.T
    ray /= np.maximum(1.e-12, np.linalg.norm(ray, axis=1, keepdims=True))
    direction = np.where(ray >= 0, 1, -1)
    reciprocal = np.divide(1., ray, out=np.full_like(ray, 1.e16), where=abs(ray)>1.e-12)
    entry = np.minimum(-eye*reciprocal, (shape-eye)*reciprocal)
    leave = np.maximum(-eye*reciprocal, (shape-eye)*reciprocal).min(axis=1)
    t = np.maximum(0., entry.max(axis=1))
    axis = np.argmax(entry, axis=1)
    point = eye + ray*(t[:, None]+1.e-6)
    cell = np.clip(np.floor(point), 0, shape-1).astype(int)
    step = (cell+(direction>0)-eye)*reciprocal
    valid = leave > t
    n = int(np.prod(shape))
    slot = np.arange(xs.size)
    ids, faces = [], []
    stride = np.array([shape[1]*shape[2],shape[2],1])
    for _ in range(int(shape.sum())+1):
        if not valid.any():
            break
        ids.append(np.where(valid, cell@stride, n).astype(np.int32))
        # Parallel faces at different depths have different codes; adjacent
        # blocks on one continuous face share their code.
        code = axis*(shape.max()+1) + cell[slot, axis] + (direction[slot, axis] < 0)
        faces.append(np.where(valid, code, -1).astype(np.int32))
        axis = step.argmin(axis=1)
        t = step[slot, axis]
        cell[slot, axis] += direction[slot, axis]
        step[slot, axis] += abs(reciprocal[slot, axis])
        valid &= (t<leave-1.e-8) & ((cell>=0)&(cell<shape)).all(1)
    if not ids:
        return np.full((xs.size,1),n,dtype=np.int32), np.full((xs.size,1),-1,dtype=np.int32)
    return np.stack(ids,1), np.stack(faces,1)


def _surface(volume):
    return sum(np.count_nonzero(np.diff(volume, axis=k)) +
               np.count_nonzero(volume.take(0,k)) + np.count_nonzero(volume.take(-1,k))
               for k in range(3))


class _ImageFit:
    def __init__(self,scene):
        hull = np.asarray(scene['hull'],bool)
        mask = np.asarray(scene['mask'],bool)
        yy,xx = np.nonzero(mask)
        stride = max(1,int(np.ceil(max(np.ptp(xx)+1,np.ptp(yy)+1)/120)))
        ymin,xmin = max(0,int(yy.min())-3*stride),max(0,int(xx.min())-3*stride)
        ymax,xmax = min(mask.shape[0],int(yy.max())+3*stride+1),min(mask.shape[1],int(xx.max())+3*stride+1)
        ys,xs = np.mgrid[ymin:ymax:stride,xmin:xmax:stride]
        self.mask = mask[ys,xs]
        projection = np.asarray(scene['P'],float)
        self.ids,self.faces = _rays(projection,xs,ys,np.array(hull.shape))
        self.index = np.arange(xs.size)
        self.shape = hull.shape
        self.foot_x = np.flatnonzero(self.mask.any(0))
        self.foot_y = self.mask.shape[0]-1-self.mask[::-1,self.foot_x].argmax(0)
        foot = self.foot_y,self.foot_x
        inverse = np.linalg.inv(projection[:,:3])
        eye = -inverse@projection[:,3]
        rays = np.column_stack((xs[foot]+.5,ys[foot]+.5,np.ones(len(self.foot_x))))@inverse.T
        visible = rays[:,1]*eye[1] < -1.e-8
        self.contacts = eye-rays[visible]*(eye[1]/rays[visible,1,None])
        self.initial_surface,self.initial_volume = max(1,_surface(hull)),max(1,hull.sum())
        # A tangent lighting ramp is weaker evidence than a sharp, sustained
        # normal step. Smooth before subsampling so native texels don't alias.
        image = Image.fromarray(np.asarray(scene['image'],np.uint8)).convert('RGB')
        rgb = np.asarray(image.filter(ImageFilter.GaussianBlur(max(.7,.64*stride))),float)[ys,xs]
        brightness = rgb.mean(2)
        observed = brightness[self.mask]
        self.shading = np.clip((brightness - np.median(observed)) /
                               max(8., observed.std()), -2., 2.)
        self.edges = []
        for axis in range(2):
            magnitude = abs(np.diff(rgb,axis=axis)).mean(2)
            inside = np.take(self.mask,range(1,self.mask.shape[axis]),axis) & np.take(
                self.mask,range(self.mask.shape[axis]-1),axis)
            cutoff = max(4.,np.quantile(magnitude[inside],.82)) if inside.any() else 4.
            graded = np.uint8(np.clip(magnitude/(2*cutoff),0,1)*255)
            nearby = (np.asarray(Image.fromarray(graded).filter(ImageFilter.MaxFilter(3))) / 255.
                      if graded.size else magnitude)
            self.edges.append((inside,nearby))

    def score(self,volume):
        solid = np.r_[volume.ravel(),False][self.ids]
        first = solid.argmax(1)
        hit = solid[self.index,first].reshape(self.mask.shape)
        faces = self.faces[self.index,first].reshape(self.mask.shape)
        match = (hit&self.mask).sum()/max(1,(hit|self.mask).sum())
        crease = 0.
        for axis,(inside,weights) in enumerate(self.edges):
            both = np.take(hit,range(1,hit.shape[axis]),axis) & np.take(hit,range(hit.shape[axis]-1),axis)
            boundary = inside & both & (np.diff(faces,axis=axis)!=0)
            crease += (weights[boundary]-.54).sum()
        # A valid front, side or roof often has a characteristic brightness.
        # Estimate each orientation's response afresh on this image.
        valid = hit & self.mask
        axes = faces[valid]//(max(self.shape)+1)
        counts = np.bincount(axes,minlength=3)
        light = self.shading[valid]
        sums = np.bincount(axes,weights=light,minlength=3)
        directional = (sums**2/np.maximum(1,counts)).sum()/max(1,self.mask.sum())
        if len(light):
            directional -= light.sum()**2/(len(light)*self.mask.sum())
        foot = self.foot_y,self.foot_x
        nearest = self.ids[self.index,first].reshape(self.mask.shape)[foot]
        height = (nearest//self.shape[2])%self.shape[1]
        ground = np.mean(hit[foot]*(height==0))
        # Penalize the jagged surface produced by carving one perspective
        # cone. This favors long planar sides even in ambiguous hidden areas.
        return (match + 1.5*crease/max(1,self.mask.sum()) + .16*directional + .10*ground -
                .035*_surface(volume)/self.initial_surface - .006*volume.sum()/self.initial_volume)


def _cuts(volume,local):
    points = np.nonzero(volume)
    lo = np.array([p.min() for p in points])
    hi = np.array([p.max()+1 for p in points])
    for axis in range(3):
        width = hi[axis]-lo[axis]
        if width<2:
            continue
        bands = [tuple(slice(None) for _ in range(3))]
        if local:
            # Keep one contiguous horizontal tier, pillar, or bay distinct
            # while removing an unsupported depth from another region.
            bands = []
            for across in range(3):
                if across==axis or hi[across]-lo[across]<4:
                    continue
                mid = (lo[across]+hi[across])//2
                for side in (slice(lo[across],mid),slice(mid,hi[across])):
                    part = [slice(None)]*3
                    part[across]=side
                    bands.append(tuple(part))
        sizes = sorted({1,max(1,round(width*.23)),max(1,width//2)})
        for band in bands:
            occupied = volume[band].any(axis=tuple(i for i in range(3) if i!=axis))
            levels = np.flatnonzero(occupied)
            if len(levels)<2:
                continue
            for size in sizes:
                if size>=len(levels):
                    continue
                for side in (slice(levels[0],levels[size-1]+1),
                             slice(levels[-size],levels[-1]+1)):
                    cut = list(band)
                    cut[axis]=side
                    yield tuple(cut)


def predict(scene):
    """Return a solid XYZ volume in the provided integer camera frame."""
    volume = np.asarray(scene['hull'],bool).copy()
    if not volume.any() or not np.any(scene['mask']):
        return np.zeros_like(volume)
    try:
        fit = _ImageFit(scene)
    except np.linalg.LinAlgError:
        return volume
    best = fit.score(volume)
    # Taking one slab at a time can lock the hull onto the wrong parallel
    # plane. Jointly remove all four side slabs implied by bottom contacts
    # when their reprojection improves the same objective.
    if len(fit.contacts) >= 4:
        lower,upper = np.quantile(fit.contacts[:,[0,2]],[.025,.975],axis=0)
        starts = np.clip(np.rint(lower).astype(int),0,np.array(volume.shape)[[0,2]])
        ends = np.clip(np.rint(upper).astype(int),starts,np.array(volume.shape)[[0,2]])
        region = (slice(starts[0],ends[0]),slice(None),slice(starts[1],ends[1]))
        grounded = np.zeros_like(volume)
        grounded[region] = volume[region]
        if grounded.any() and (score := fit.score(grounded)) > best:
            volume,best = grounded,score
    for local,iterations in ((False,6),(True,5)):
        for _ in range(iterations):
            winner = None
            trial_best = best + .0001
            for cut in _cuts(volume,local):
                reduced = volume.copy()
                reduced[cut] = False
                value = fit.score(reduced)
                if value>trial_best:
                    winner,trial_best = cut,value
            if winner is None:
                break
            volume[winner] = False
            best=trial_best
    return volume
