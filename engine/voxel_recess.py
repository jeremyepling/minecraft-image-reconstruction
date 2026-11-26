"""Subtract rectangular pockets with photographed evidence for their return edges."""
import numpy as np
from PIL import Image, ImageFilter


def _project(P, xyz):
    uvw = xyz @ P[:, :3].T + P[:, 3]
    return uvw[..., :2] / uvw[..., 2, None]


def _line_strength(gray, start, end):
    """Sustained signed contrast across a projected segment, with a one-pixel tolerance."""
    vector = end - start
    length = np.linalg.norm(vector, axis=-1)
    normal = np.stack((-vector[..., 1], vector[..., 0]), axis=-1)
    normal /= np.maximum(length[..., None], .001)
    along = start[..., None, :] + vector[..., None, :] * np.linspace(.16, .84, 7)[:, None]

    def samples(offset):
        xy = np.rint(along + offset * normal[..., None, :]).astype(int)
        return gray[xy[..., 1].clip(0, gray.shape[0]-1), xy[..., 0].clip(0, gray.shape[1]-1)]

    contrast = [abs((samples(s+1.4) - samples(s-1.4)).mean(axis=-1)) for s in (-.7, 0, .7)]
    return np.maximum.reduce(contrast) * np.clip((length - 1.5) / 2., 0, 1)


def _proposals(gray, hull, P, z):
    """Rank front-plane rectangular rims using line integrals on the voxel grid."""
    nx, ny, _ = hull.shape
    x, y = np.indices((nx+1, ny+1))
    grid = _project(P, np.stack((x, y, np.full_like(x, z)), axis=-1))
    horizontal = _line_strength(gray, grid[:-1], grid[1:])
    vertical = _line_strength(gray, grid[:, :-1], grid[:, 1:])
    hsum = np.pad(horizontal.cumsum(0), ((1, 0), (0, 0)))
    vsum = np.pad(vertical.cumsum(1), ((0, 0), (1, 0)))
    # A pocket must have a front to cut and a neighboring plane to retain.
    visible = hull[:, :, z] & ~hull[:, :, :z].any(axis=2)
    occupied = np.pad(visible.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    out = []
    for w in range(2, min(9, nx)+1):
        for h in range(2, min(9, ny)+1):
            bottom = (hsum[w:, :-h] - hsum[:-w, :-h]) / w
            top = (hsum[w:, h:] - hsum[:-w, h:]) / w
            left = (vsum[:-w, h:] - vsum[:-w, :-h]) / h
            right = (vsum[w:, h:] - vsum[w:, :-h]) / h
            count = occupied[w:, h:] - occupied[:-w, h:] - occupied[w:, :-h] + occupied[:-w, :-h]
            # Permit a weak sill or jamb. Two separate directions must still agree.
            score = (.6 * np.minimum(np.maximum(left, right), np.maximum(top, bottom))
                     + .4 * (bottom+top+left+right)/4)
            score[count < .85*w*h] = 0
            # A jamb and lintel belong to the building on both sides. Reject
            # the outer silhouette, which often dominates every interior edge.
            fx, fy = np.indices(score.shape)
            frame = ((fx > 0) & (fx+w < nx) & (fy+h < ny)
                     & visible[(fx-1).clip(0), fy+h//2]
                     & visible[(fx+w).clip(max=nx-1), fy+h//2]
                     & visible[fx+w//2, (fy+h).clip(max=ny-1)])
            score[~frame] = 0
            for idx in np.argsort(score.ravel(), kind='stable')[-3:]:
                a, b = np.unravel_index(idx, score.shape)
                if score[a,b] > 8:
                    out.append((score[a,b], int(a), int(b), w, h))
    return sorted(out, reverse=True)[:75]


def _cuts(gray, hull, P, camera, z):
    fits = []
    for rim, x, y, w, h in _proposals(gray, hull, P, z):
        corners = np.array([[x,y,z], [x+w,y,z], [x+w,y+h,z], [x,y+h,z]], float)
        front = _project(P, corners)
        for depth in range(1, min(4, hull.shape[2]-z-1)+1):
            if hull[x:x+w,y:y+h,z+depth].mean() < .85:
                continue
            back_xyz = corners + [0,0,depth]
            # Project back vertices onto the opening plane along the real camera rays.
            # Hidden connectors cannot corroborate a cut.
            ratio = (z - camera[2]) / (z + depth - camera[2])
            landing = camera[:2] + ratio * (back_xyz[:,:2] - camera[:2])
            visible = ((landing >= [x,y]) & (landing <= [x+w,y+h])).all(1)
            if not visible.any():
                continue
            back = _project(P, back_xyz)
            return_edges = _line_strength(gray, front, back)
            back_edges = _line_strength(gray, back, np.roll(back, -1, axis=0))
            # At least one connector inside the opening ends at two supported back edges.
            evidence = np.minimum(return_edges, np.minimum(back_edges, np.roll(back_edges, 1)))
            support = evidence[visible].max()
            if support >= 8:
                score = min(rim, support) + .15*support - .9*depth
                # At comparable contrast, a longer observed return carries
                # stronger evidence than a cropped fragment of the same pocket.
                fits.append((score * (w*h)**.08, x,y,z,w,h,depth))
    return fits


def predict(scene):
    hull = np.asarray(scene['hull'], bool)
    result = hull.copy()
    if not hull.any() or hull.shape[2] < 2:
        return result
    P = scene['P']
    camera = -np.linalg.solve(P[:, :3], P[:, 3])
    if camera[2] >= 0:
        return result
    gray = np.asarray(Image.fromarray(scene['image']).convert('L').filter(ImageFilter.GaussianBlur(.65)), float)
    first = hull.argmax(axis=2)[hull.any(axis=2)]
    counts = np.bincount(first, minlength=hull.shape[2])
    fits = []
    for z in np.argsort(-counts, kind='stable')[:3]:
        if counts[z] >= 8:
            fits.extend(_cuts(gray,hull,P,camera,z))
    touched = np.zeros(hull.shape[:2], bool)
    for score,x,y,z,w,h,depth in sorted(fits, reverse=True):
        patch = touched[x:x+w,y:y+h]
        if score < 11 or patch.any():
            continue
        result[x:x+w,y:y+h,z:z+depth] = False
        patch[:] = True
    return result
