"""Image geometry shared by visible placement and the independent dataset labeler."""
import numpy as np

SAMPLE_STRIDE = 4


def observed_support(support,shape,size):
    """Resample half-open analysis-mask runs at the sample centers."""
    sw,sh=support['size']
    raw=np.zeros((sh,sw),dtype=bool)
    for y,x0,x1 in support['runs']:
        raw[y,x0:x1]=True
    u=((np.arange(shape[1])+.5)*SAMPLE_STRIDE*sw/size[0]).astype(int).clip(0,sw-1)
    v=((np.arange(shape[0])+.5)*SAMPLE_STRIDE*sh/size[1]).astype(int).clip(0,sh-1)
    return raw[np.ix_(v,u)]


# The frozen visible-geometry archive fingerprints the complete source of rasterize().
def rasterize(quad3, width, height):
    """Return a screen-clipped pixel-center mask, reciprocal depth and full-face area."""
    x, y, w = quad3.T
    if any(np.all(side <= 0) for side in (w, x, y, width*w-x, height*w-y)):
        return None
    # Crossing the camera plane can make a projected polygon unbounded. These fixtures
    # arrange the close views so that such side faces are already outside the frustum.
    if w.min() <= 0:
        raise ValueError("A face crossing the camera plane intersects the scoring viewport")
    q = quad3[:, :2] / w[:, None]
    low = np.maximum(0, np.floor(q.min(0)).astype(int))
    high = np.minimum((width, height), np.ceil(q.max(0)).astype(int))
    if np.any(low >= high):
        return None
    px, py = np.meshgrid(np.arange(low[0], high[0]) + .5, np.arange(low[1], high[1]) + .5)
    polygon = np.ones(px.shape, dtype=bool)
    area = .5 * (q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1))
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        polygon &= np.sign(area) * cross >= 0
    if not polygon.any():
        return None
    a, b, c = np.linalg.solve(quad3[:3], np.ones(3))
    depth = a * px + b * py + c
    complete = np.all(q >= 0) and np.all(q <= (width, height))
    denominator = polygon.sum() if complete else abs(area)
    return (slice(low[1], high[1]), slice(low[0], high[0])), polygon, depth, denominator, q
