"""Make a front-facing image chart from one predicted surface and its accepted pixel support."""
import numpy as np

from engine.geometry import UNIT, bilinear, project, transform
from engine.surface_model import canonical, lens


SETTINGS = dict(max_side=640, minimum_edge_px=3, minimum_view_cosine=.16,
                maximum_wall_normal_y=.5, minimum_wall_normal_z=.15)


def _wall_chart(region, size):
    """Construct texture axes on a wall using its measured camera normal and zero camera roll."""
    support=region.get('support')
    normal=np.asarray(region.get('normal_camera'),dtype=float)
    if not support or not len(support['runs']) or normal.shape!=(3,) or not np.isfinite(normal).all():
        return None
    normal=normal/max(1.e-9,np.linalg.norm(normal))
    if abs(normal[1])>SETTINGS['maximum_wall_normal_y'] or abs(normal[2])<SETTINGS['minimum_wall_normal_z']:
        return None
    # Down projects into the camera's yz plane. Its perpendicular in the
    # surface supplies the horizontal axis. A floor normal alone lacks yaw.
    down=np.array([0.,1.,-normal[1]/normal[2]])
    down/=np.linalg.norm(down)
    across=np.cross(normal,down)
    axes=np.stack([across,down],axis=1)
    y,x0,x1=np.asarray(support['runs']).T
    boundary=np.column_stack((np.r_[x0,x1,x0,x1],np.r_[y,y,y+1,y+1]))*np.array(size)/support['size']
    K,origin=lens([],size)
    rays=np.linalg.solve(K,np.column_stack((boundary,np.ones(len(boundary)))).T).T
    incidence=rays@normal
    if np.any(abs(incidence)/np.linalg.norm(rays,axis=1)<SETTINGS['minimum_view_cosine']) or incidence.min()*incidence.max()<=0:
        return None
    # Unit distance fixes an arbitrary plane depth only for rectification.
    # Ratios and texture repeat counts are invariant to that common scale.
    coordinates=(rays/incidence[:,None])@axes
    bounds=coordinates.min(0)+UNIT*np.ptp(coordinates,axis=0)
    corners=(bounds@axes.T + normal)@K.T
    if np.any(corners[:,2]*np.median(incidence)<=0):
        return None
    quad=canonical(corners[:,:2]/corners[:,2,None])
    return quad,'wall normal + '+origin


def _supported_taps(xy, support, size):
    """All four native RGB interpolation taps must belong to the same visible mask."""
    width, height = size
    x, y = np.floor(xy - .5).astype(int).transpose(2, 0, 1)
    valid = (x >= 0) & (x + 1 < width) & (y >= 0) & (y + 1 < height)
    if support is None:
        return valid
    mw, mh = support['size']
    observed = np.zeros((mh, mw), bool)
    for row, start, stop in support['runs']:
        observed[row, start:stop] = True
    for dy in (0, 1):
        my = np.clip(np.floor((y + dy + .5) * mh / height).astype(int), 0, mh-1)
        for dx in (0, 1):
            mx = np.clip(np.floor((x + dx + .5) * mw / width).astype(int), 0, mw-1)
            valid &= observed[my, mx]
    return valid


def _normal_aspect(normal, quad, size):
    """Estimate edge lengths' ratio from a predicted normal with a lens-sensitivity check."""
    n=np.asarray(normal,dtype=float)
    if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)<.5:
        return None
    n=n/np.linalg.norm(n)
    K,_=lens([],size)
    ratios=[]
    for focal_scale in (.9,1.,1.1):
        lens_matrix=K.copy()
        lens_matrix[0,0]*=focal_scale
        lens_matrix[1,1]*=focal_scale
        rays=np.linalg.solve(lens_matrix,np.column_stack((quad,np.ones(4))).T).T
        incidence=rays@n
        if incidence.min()*incidence.max()<=0 or min(abs(incidence)/np.linalg.norm(rays,axis=1))<SETTINGS['minimum_view_cosine']:
            return None
        on_plane=rays/incidence[:,None]
        lengths=np.linalg.norm(np.roll(on_plane,-1,axis=0)-on_plane,axis=1)
        ratios.append((lengths[0]+lengths[2])/(lengths[1]+lengths[3]))
    if max(ratios)/min(ratios)>1.15:
        return None
    return float(ratios[1])


def sample_surface(image, region):
    """Project an existing UV chart to a masked RGB raster.

    Return native_size (screenshot width, height), source_image (RGB PIL), the
    original chart corners, uint8 H×W×3 rgb and bool H×W mask. RGB outside
    the mask is zero. A quad alone requests a rectangular sampling window.
    Chartless walls can use their camera-space normal to straighten the image.
    """
    quad = region.get('quad')
    chart_source='supplied quad'
    if quad is None:
        recovered=_wall_chart(region,image.size)
        if recovered is None:
            return None
        quad,chart_source=recovered
    quad = np.asarray(quad, float)
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    edges = np.roll(quad, -1, axis=0) - quad
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0]*following[:, 1] - edges[:, 1]*following[:, 0]
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) < SETTINGS['minimum_edge_px'] or not (np.all(turns>0) or np.all(turns<0)):
        return None
    dimensions = (lengths[:2]+lengths[2:]) / 2
    width, height = np.maximum(3, np.rint(dimensions * min(1., SETTINGS['max_side']/max(dimensions)))).astype(int)
    try:
        chart = transform(quad)
    except np.linalg.LinAlgError:
        return None
    u, v = (np.arange(width)+.5)/width, (np.arange(height)+.5)/height
    xy = project(chart, np.stack(np.meshgrid(u, v), axis=-1))
    if not np.isfinite(xy).all():
        return None
    valid = _supported_taps(xy, region.get('support'), image.size)
    # Interpolation near the quad edge can also sample an adjacent surface,
    # particularly for a caller that only supplies four reference corners.
    for corner, edge, length in zip(quad, edges, lengths):
        distance = np.abs(edge[0]*(xy[..., 1]-corner[1]) - edge[1]*(xy[..., 0]-corner[0])) / length
        valid &= distance > 1.42
    source = image if image.mode == 'RGB' else image.convert('RGB')
    rgb = np.round(bilinear(np.asarray(source), xy)).astype(np.uint8)
    rgb[~valid] = 0
    return dict(rgb=rgb, mask=valid, quad=quad.tolist(), native_size=source.size, source_image=source,
                chart_source=chart_source,normal_aspect=_normal_aspect(region.get('normal_camera'),quad,size=source.size))
