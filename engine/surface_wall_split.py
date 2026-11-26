"""Split a merged wall using image-supported edges and relative inverse-depth offsets."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.camera_lines import _runs
from engine.mask_first import _image_depth
from engine.normal_regions import _local_mask, _patch
from engine.surface_depth_split import _connected_sides, _edge_vote, _labels, _slope
from engine.surface_model import canonical


SETTINGS = dict(min_area=96, min_fraction=.05, min_shared_samples=12,
                min_step=.007, min_rect_agreement=.76, max_fragments=6,
                min_plane_gain=.38, min_broad_step=.012, min_edge_contrast=9.,
                rectangle_offsets=(0., -2., 2.))


def _rect_sides(rectangles, size, scale):
    for rect in rectangles:
        xy = np.asarray(rect['quad']) / scale
        if len(xy) != 4 or not np.isfinite(xy).all():
            continue
        mask = Image.new('1',size)
        ImageDraw.Draw(mask).polygon(list(map(tuple,xy)),fill=1)
        yield np.asarray(mask),np.asarray(rect['normal_camera']),xy


def _rectangle_edges(xy,x,y,mask):
    for start,end in zip(xy,np.roll(xy,-1,axis=0)):
        vx,vy=end-start
        length=np.hypot(vx,vy)
        if length<12:
            continue
        dx,dy=x+.5-start[0],y+.5-start[1]
        side=(dy*vx-dx*vy)/length
        along=(dx*vx+dy*vy)/length
        collar=(abs(side)<=1.5)&mask
        support=collar&(along>=0)&(along<=length)
        if support.sum()>=max(24,.6*collar.sum()):
            for offset in SETTINGS['rectangle_offsets']:
                yield side>offset


def _components(side, mask, minimum):
    """A cut through a beam can disconnect more than two pillars. Retain every pixel."""
    components=[]
    for inside in (mask & side,mask & ~side):
        if inside.sum() < minimum:
            return []
        sets=_runs(inside)
        if len(sets)+len(components)>SETTINGS['max_fragments']:
            return []
        for runs in sets:
            if sum(b-a for _,a,b in runs) < minimum:
                return []
            components.append(runs)
    return components


def _plane_error(design,depth):
    weights=np.ones(len(depth))
    for _ in range(3):
        coef=np.linalg.lstsq(design*weights[:,None],depth*weights,rcond=None)[0]
        error=np.abs(depth-design@coef)
        weights=np.sqrt(np.minimum(1.,max(.001,2*np.median(error))/np.maximum(error,1.e-8)))
    return coef,float(np.median(error))


def _affine_evidence(mask,side,z,rgb):
    """Allow a blurred step if the opposite interiors fit parallel inverse-depth planes."""
    samples=[]
    for region in (mask&~side,mask&side):
        core=np.asarray(Image.fromarray(np.pad(region,3)).filter(ImageFilter.MinFilter(7)))[3:-3,3:-3]
        ys,xs=np.nonzero(core)
        if len(xs)<32:
            return None
        indices=np.linspace(0,len(xs)-1,min(500,len(xs)),dtype=int)
        samples.append((ys[indices],xs[indices]))
    y,x=np.concatenate(samples,axis=1)
    coords=np.column_stack((x-x.mean(),y-y.mean(),np.ones(len(x))))
    groups=np.column_stack((coords,side[y,x]))
    _,single=_plane_error(coords,z[y,x])
    params,double=_plane_error(groups,z[y,x])
    offset=float(params[-1])
    gain=1-double/max(single,.001)
    if abs(offset)<SETTINGS['min_broad_step'] or gain<SETTINGS['min_plane_gain']:
        return None
    # Test color on both sides of the actual interface, including segments that continue
    # beyond the proposed rectangle, to reject splits extending a single short seam.
    contrasts=[]
    for axis in (0,1):
        a,b=(np.s_[:-1,:],np.s_[1:,:]) if axis==0 else (np.s_[:,:-1],np.s_[:,1:])
        yy,xx=np.nonzero(mask[a]&mask[b]&(side[a]!=side[b]))
        dy,dx=int(axis==0),int(axis==1)
        y0,x0=yy-2*dy,xx-2*dx
        y1,x1=yy+3*dy,xx+3*dx
        use=(y0>=0)&(x0>=0)&(y1<mask.shape[0])&(x1<mask.shape[1])
        y0,x0,y1,x1=y0[use],x0[use],y1[use],x1[use]
        use=mask[y0,x0]&mask[y1,x1]&(side[y0,x0]!=side[y1,x1])
        contrasts.extend(np.max(abs(rgb[y0[use],x0[use]]-rgb[y1[use],x1[use]]),axis=1))
    if len(contrasts)<SETTINGS['min_shared_samples']:
        return None
    agreement=float(np.mean(np.asarray(contrasts)>=SETTINGS['min_edge_contrast']))
    if agreement<.60:
        return None
    return abs(offset)*gain*agreement,dict(fitted_offset=offset,fit_gain=gain,edge_support=agreement)


def _score(mask, side, residual, z, rgb):
    if side is None:
        return None
    minimum=max(SETTINGS['min_area'],int(mask.sum()*SETTINGS['min_fraction']))
    inside=mask & side
    outside=mask & ~side
    if min(inside.sum(), outside.sum()) < minimum:
        return None
    offset=float(np.median(residual[inside])-np.median(residual[outside]))
    if abs(offset)<SETTINGS['min_step']:
        return None
    children=_components(side,mask,minimum)
    if not children:
        side=_connected_sides(mask,side,minimum)
        if side is None:
            return None
        children=_components(side,mask,minimum)
        if not children:
            return None
    edge=_edge_vote(side,mask,rgb,z)
    if edge is not None and edge[0]*offset>0:
        return abs(edge[0])*edge[1],children,dict(relative_offset=offset,edge_offset=edge[0],edge_support=edge[1])
    fitted=_affine_evidence(mask,side,z,rgb)
    if fitted is None:
        return None
    weight,evidence=fitted
    return weight,children,evidence


def _partition(part,mask,box,z,rgb,rects):
    if min(mask.shape)<12 or mask.sum()<2*SETTINGS['min_area']:
        return None
    y,x=np.indices(mask.shape)
    residual=z-_slope(z,mask,0)*x-_slope(z,mask,1)*y
    pixels=rgb[mask]
    centered=pixels-pixels.mean(axis=0)
    _,axes=np.linalg.eigh(centered.T@centered)
    samples=[('photometric',_labels(rgb@axes[:,-1],mask)),('depth',_labels(residual,mask))]
    for rect,normal,xy in rects:
        if normal@part['normal_camera']>=SETTINGS['min_rect_agreement']:
            region=rect[box]
            if (region&mask).sum() < SETTINGS['min_area']:
                continue
            samples.append(('rectangle',region))
            edges=_rectangle_edges(xy,x+box[1].start,y+box[0].start,mask)
            samples.extend(('rectangle edge',edge) for edge in edges)
    best=None
    for kind,side in samples:
        cut=_score(mask,side,residual,z,rgb)
        if cut and (best is None or cut[0]>best[0]):
            weight,children,evidence=cut
            best=weight,children,dict(source=kind,**evidence)
    return best


def refine(image, parts, depth, rectangles):
    """Return disjoint children of a mask only when a proposed division has depth and color support."""
    if not parts:
        return []
    size=tuple(parts[0]['support']['size'])
    if any(tuple(p['support']['size'])!=size for p in parts):
        raise ValueError('Wall splitting requires one shared mask grid')
    if np.ndim(depth)!=2 or not np.isfinite(depth).all():
        raise ValueError('Wall splitting requires a finite relative inverse-depth map')
    z=_image_depth(np.asarray(depth),size)
    small=image.convert('RGB').resize(size,Image.Resampling.BOX)
    rgb=np.asarray(small.filter(ImageFilter.GaussianBlur(1.25)),np.float32)
    gray=np.asarray(small.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    scale=np.asarray(image.size)/size
    rects=list(_rect_sides(rectangles,size,scale))
    output=[]
    for part in parts:
        mask,x,y=_local_mask(np.asarray(part['support']['runs']))
        box=np.s_[y:y+mask.shape[0],x:x+mask.shape[1]]
        best=_partition(part,mask,box,z[box],rgb[box],rects)
        if best is None:
            output.append(part)
            continue
        _,sets,evidence=best
        for i,component in enumerate(sets):
            runs=[[v+y,a+x,b+x] for v,a,b in component]
            values=np.concatenate([z[v,a:b] for v,a,b in runs])
            child=dict(id=f"{part['id']}-wall-{i}",parent_id=part['id'],
                       support=dict(size=list(size),runs=runs),support_area=len(values),
                       normal_camera=part['normal_camera'],relative_inverse_depth=float(np.median(values)),
                       split_by_depth=True,wall_split=evidence,geometry_status='mask only')
            chart=_patch(runs,gray,scale,size,np.asarray(child['normal_camera']))
            if chart is not None:
                child.update(quad=canonical(chart['quad']),geometry_status='mask with sampling chart')
            output.append(child)
    return output
