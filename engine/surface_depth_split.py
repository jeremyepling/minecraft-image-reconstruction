"""Separate connected, parallel surface candidates using relative depth and an image edge."""
import numpy as np
from PIL import Image, ImageFilter

from engine.camera_lines import _runs
from engine.mask_first import _image_depth
from engine.normal_regions import _local_mask, _patch
from engine.surface_model import canonical


SETTINGS = dict(min_child_area=96, min_child_fraction=.06, min_core_coverage=.92,
                min_shared_edge=12, min_step=.008, min_local_step=.003,
                min_rgb_jump=7., min_joint_fraction=.50, sample_radius=5)


def _labels(signal, mask):
    values = signal[mask]
    centers = np.percentile(values, [20, 80])
    for _ in range(8):
        side = values > centers.mean()
        if side.all() or not side.any():
            return None
        updated = np.array([np.median(values[~side]), np.median(values[side])])
        if np.max(abs(centers-updated)) < .0001:
            break
        centers = updated
    return signal > centers.mean()


def _connected_sides(mask, side, minimum):
    if side is None:
        return None
    label = np.zeros(mask.shape, np.uint8)
    for i, condition in enumerate((~side, side), 1):
        subset = mask & condition
        area = subset.sum()
        if area < minimum:
            return None
        groups = _runs(subset)
        largest = max(groups, key=lambda g:sum(b-a for _,a,b in g))
        size = sum(b-a for _,a,b in largest)
        if size < minimum or size < SETTINGS['min_core_coverage']*area:
            return None
        for y,a,b in largest:
            label[y,a:b] = i
    # Detached pixels of either cluster are assigned to an adjacent connected core.
    # Every original pixel stays in exactly one output surface.
    while np.any((label == 0) & mask):
        pad = np.pad(label, 1)
        neighbors = np.maximum.reduce((pad[1:-1,:-2],pad[1:-1,2:],pad[:-2,1:-1],pad[2:,1:-1]))
        update = (label == 0) & mask & (neighbors > 0)
        if not update.any():
            return None
        label[update] = neighbors[update]
    return label == 2


def _slope(depth, mask, axis):
    z, m = (depth,mask) if axis == 0 else (depth.T,mask.T)
    delta = z[:,4:] - z[:,:-4]
    valid = m[:,4:] & m[:,:-4] & m[:,2:-2]
    return float(np.median(delta[valid])/4) if valid.any() else 0.


def _edge_vote(side, mask, rgb, z):
    """Check slope-corrected depth on both sides of the actual connected interface."""
    # A smoothed binary derivative supplies a normal at stair steps in the raster.
    soft = Image.fromarray(np.uint8(255*side)).filter(ImageFilter.GaussianBlur(2))
    dy, dx = np.gradient(np.asarray(soft, np.float32))
    inside = side & mask
    other = np.pad(mask & ~side,1)
    y_edges, x_edges = np.where(inside & (
        other[1:-1,:-2] | other[1:-1,2:] | other[:-2,1:-1] | other[2:,1:-1]))
    if len(x_edges) < SETTINGS['min_shared_edge']:
        return None
    normal = np.column_stack((dx[y_edges,x_edges],dy[y_edges,x_edges]))
    normal /= np.maximum(1.,np.linalg.norm(normal,axis=1))[:,None]
    origin = np.column_stack((x_edges,y_edges)) - .5 * normal
    valid = np.ones(len(origin),bool)
    def coordinates(distance):
        pt = np.rint(origin + normal*distance).astype(int)
        x,y = pt.T
        valid[:] &= (x>=0) & (x<mask.shape[1]) & (y>=0) & (y<mask.shape[0])
        x,y = x.clip(0,mask.shape[1]-1),y.clip(0,mask.shape[0]-1)
        valid[:] &= mask[y,x] & (side[y,x] == (distance>0))
        return y,x
    near,far = SETTINGS['sample_radius'],2*SETTINGS['sample_radius']
    # For an affine inverse-depth plane z(x,y) the extrapolated jump is zero.
    z_jump = 2*(z[coordinates(near)] - z[coordinates(-near)]) - (z[coordinates(far)] - z[coordinates(-far)])
    rgb_jump = np.max(abs(rgb[coordinates(2)]-rgb[coordinates(-2)]),axis=1)
    if valid.sum() < max(SETTINGS['min_shared_edge'],.42*len(origin)):
        return None
    jump = z_jump[valid]
    sign = np.sign(np.median(jump))
    joint = (sign*jump >= SETTINGS['min_local_step']) & (rgb_jump[valid] >= SETTINGS['min_rgb_jump'])
    if joint.mean() < SETTINGS['min_joint_fraction']:
        return None
    return float(np.median(jump)),float(joint.mean())


def _partition(mask, z, rgb):
    minimum = max(SETTINGS['min_child_area'],SETTINGS['min_child_fraction']*mask.sum())
    if min(mask.shape) < 12 or mask.sum() < minimum*2:
        return None
    yy,xx = np.indices(mask.shape)
    residual = z-xx*_slope(z,mask,0)-yy*_slope(z,mask,1)
    # Either channel can suggest two broad regions. Both must support their common edge.
    sample = rgb[mask]
    centered = sample - np.mean(sample,axis=0)
    _,axes = np.linalg.eigh(centered.T @ centered)
    color = rgb @ axes[:,-1]
    best = None
    for field in (color,residual):
        side = _connected_sides(mask,_labels(field,mask),minimum)
        if side is None:
            continue
        step = np.median(residual[mask & side])-np.median(residual[mask & ~side])
        if abs(step) < SETTINGS['min_step']:
            continue
        evidence = _edge_vote(side,mask,rgb,z)
        if evidence is None or np.sign(evidence[0]) != np.sign(step):
            continue
        score = abs(evidence[0])*evidence[1]
        if best is None or score > best[0]:
            best = score,side,dict(relative_offset=float(step),local_offset=evidence[0],joint_edge_fraction=evidence[1])
    return best


def split(image, parts, depth, version='26.3'):
    """Return a mask-preserving split only for a spatially corroborated relative-depth step."""
    if version != '26.3':
        raise ValueError('Parallel-plane splitting targets Java 26.3.')
    if not parts:
        return []
    size = tuple(parts[0]['support']['size'])
    if any(tuple(part['support']['size'])!=size for part in parts):
        raise ValueError('All proposals must use the same support grid')
    raw = np.asarray(depth)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Expected a finite relative inverse depth image')
    z = _image_depth(raw,size)
    small = image.convert('RGB').resize(size,Image.Resampling.BOX)
    rgb = np.asarray(small.filter(ImageFilter.GaussianBlur(1.25)),np.float32)
    gray = np.asarray(small.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    scale = np.asarray(image.size)/size
    output = []
    for part in parts:
        local,x0,y0 = _local_mask(np.asarray(part['support']['runs']))
        box = np.s_[y0:y0+local.shape[0],x0:x0+local.shape[1]]
        cut = _partition(local,z[box],rgb[box])
        if cut is None:
            output.append(part)
            continue
        _,side,evidence = cut
        for i,subset in enumerate((local & ~side,local & side)):
            component, = _runs(subset)
            runs = [[v+y0,a+x0,b+x0] for v,a,b in component]
            child = dict(id=f"{part['id']}-depth-{i}",parent_id=part['id'],
                         support=dict(size=list(size),runs=runs),support_area=int(subset.sum()),
                         normal_camera=part['normal_camera'],relative_inverse_depth=float(np.median(z[box][subset])),
                         split_by_depth=True,depth_split=evidence,geometry_status='mask only')
            chart = _patch(runs,gray,scale,size,np.asarray(child['normal_camera']))
            if chart is not None:
                child.update(quad=canonical(chart['quad']),geometry_status='mask with sampling chart')
            output.append(child)
    return output
