"""Single-image calibration using connected, individually outlined floor tiles."""
from collections import defaultdict, deque

import numpy as np
from PIL import Image


SETTINGS = dict(max_width=1000, ground_rows=[.57, .995], threshold_quantiles=[.14, .50],
                min_tiles=12, neighbor_tolerance=.30, consensus_cell_fraction=.12)


def _runs(mask):
    """Extract 4-connected components as horizontal runs."""
    parent, runs, previous = [], [], []

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y, row in enumerate(mask):
        boundaries = np.flatnonzero(np.diff(row, prepend=False, append=False))
        current = []
        k = 0
        for a, b in boundaries.reshape(-1, 2):
            i = len(runs)
            a, b = int(a), int(b)
            runs.append((y, a, b))
            parent.append(i)
            current.append(i)
            while k < len(previous) and runs[previous[k]][2] <= a:
                k += 1
            j = k
            while j < len(previous) and runs[previous[j]][1] < b:
                parent[root(previous[j])] = root(i)
                j += 1
        previous = current
    groups = defaultdict(list)
    for i, run in enumerate(runs):
        groups[root(i)].append(run)
    return groups.values()


def _outline(points):
    """Monotone hull followed by removal of the smallest intervening triangles."""
    points = sorted(set(points))

    def chain(points):
        result = []
        for p in points:
            while len(result) > 1:
                a, b = result[-2:]
                if (b[0]-a[0])*(p[1]-b[1]) > (b[1]-a[1])*(p[0]-b[0]):
                    break
                result.pop()
            result.append(p)
        return result[:-1]

    hull = np.array(chain(points) + chain(reversed(points)), float)
    while len(hull) > 4:
        a, b = hull-np.roll(hull, 1, axis=0), np.roll(hull, -1, axis=0)-hull
        hull = np.delete(hull, np.argmin(np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])), axis=0)
    return hull


def _tiles(image):
    width, height = image.size
    factor = min(1., SETTINGS['max_width']/width)
    size = (round(width*factor), round(height*factor))
    small = image.resize(size, Image.Resampling.BOX).convert('L')
    first, last = [round(size[1]*fraction) for fraction in SETTINGS['ground_rows']]
    gray = np.asarray(small)[first:last]
    low, high = np.quantile(gray, SETTINGS['threshold_quantiles'], axis=1)
    mask = gray > ((low+high)/2)[:, None]
    scale = np.array(image.size) / size
    quads, centers = [], []
    for region in _runs(mask):
        if len(region) < 5 or region[0][0] == 0 or region[-1][0] == len(gray)-1:
            continue
        if any(a == 0 or b == size[0] for _, a, b in region):
            continue
        area = sum(b-a for _, a, b in region)
        if area < 28 or area > mask.size*.05:
            continue
        points = [(x, y) for y, a, b in region for x in (a, b-1)]
        quad = _outline(points)
        if len(quad) != 4:
            continue
        following = np.roll(quad, -1, axis=0)
        polygon_area = abs(np.sum(quad[:, 0]*following[:, 1]-quad[:, 1]*following[:, 0]))/2
        if not .76 < polygon_area/area < 1.12:
            continue
        diagonal = np.array([quad[2]-quad[0], quad[1]-quad[3]]).T
        try:
            t = np.linalg.solve(diagonal, quad[1]-quad[0])[0]
        except np.linalg.LinAlgError:
            continue
        diagonal_center = quad[0]+t*diagonal[:, 0]
        # Averaging all bright pixel centers resists a single ragged corner.
        # Correct the small offset of an area's centroid from the projective
        # center using the fitted polygon (the two coincide in affine views).
        cross = quad[:, 0]*following[:, 1]-quad[:, 1]*following[:, 0]
        poly_centroid = ((quad+following)*cross[:, None]).sum(0)/(3*cross.sum())
        yy, aa, bb = np.array(region).T
        center = np.average(np.stack([(aa+bb-1)/2, yy], axis=1), axis=0, weights=bb-aa)
        centers.append((center+diagonal_center-poly_centroid+[.5, first+.5])*scale)
        quads.append((quad+[.5, first+.5])*scale)
    return np.array(centers), np.array(quads)


def _patches(centers, quads):
    """Index touching tile centers with integer coordinates via shared sides."""
    adjacency = {}
    for i, quad in enumerate(quads):
        sides = (quad+np.roll(quad, -1, axis=0))/2
        basis = np.stack((sides[0]-sides[2], sides[1]-sides[3]), axis=1)
        try:
            delta = (centers-centers[i]) @ np.linalg.inv(basis).T
        except np.linalg.LinAlgError:
            continue
        # A dark separating line makes neighboring centers slightly farther
        # apart than opposite edges of the bright interior.
        for side, direction in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
            distance = np.linalg.norm(delta-1.15*np.asarray(direction), axis=1)
            j = int(distance.argmin())
            if distance[j] < SETTINGS['neighbor_tolerance']:
                adjacency[i, side] = j
    directions = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])
    seen = set()
    for start in range(len(centers)):
        if start in seen:
            continue
        seen.add(start)
        pending = deque([(start, np.zeros(2, int), 0)])
        patch = {}
        while pending:
            i, cell, rotation = pending.popleft()
            patch[i] = cell
            for side in range(4):
                j = adjacency.get((i, side))
                if j is None or j in seen:
                    continue
                back = next((e for e in range(4) if adjacency.get((j, e)) == i), None)
                if back is None:
                    continue
                seen.add(j)
                axis = (side+rotation) % 4
                pending.append((j, cell+directions[axis], (axis+2-back) % 4))
        if len(patch) >= SETTINGS['min_tiles']:
            yield np.array(list(patch)), np.array(list(patch.values()), dtype=float)+.5


def _homography(xz, xy):
    # Use centered source and image coordinates so the DLT remains stable
    # even when the local patch lies far from the origin.
    centers = [xz.mean(0), xy.mean(0)]
    src, dst = xz-centers[0], xy-centers[1]
    u = np.column_stack((src, np.ones(len(xz))))
    zero = np.zeros_like(u)
    rows = np.vstack((np.column_stack((u, zero, -dst[:, 0, None]*u)),
                      np.column_stack((zero, u, -dst[:, 1, None]*u))))
    _, vec = np.linalg.eigh(rows.T @ rows)
    H = vec[:, 0].reshape(3, 3)
    H[:2] += centers[1][:, None]*H[2]
    H[:, 2] -= H[:, :2] @ centers[0]
    return H / H[2, 2]


def _upright_camera(params, center, handedness):
    yaw, pitch, log_f = params[:3]
    right = np.array([1., 0., 0.])
    up = np.array([0., -np.cos(pitch), -np.sin(pitch)])
    forward = np.array([0., -np.sin(pitch), np.cos(pitch)])
    x = np.cos(yaw)*right + np.sin(yaw)*forward
    z = handedness*(-np.sin(yaw)*right + np.cos(yaw)*forward)
    f = np.exp(log_f)
    K = np.array([[f, 0, center[0]], [0, f, center[1]], [0, 0, 1.]])
    return K @ np.column_stack((x, up, z, params[3:]))


def _projection(xz, xy, size):
    H = _homography(xz, xy)
    center = np.array(size)/2
    top = H[:2]-center[:, None]*H[2]
    u, v = top[:, :2].T
    a, b = H[2, :2]
    # Orthogonality AND equal lengths of both world directions constrain f.
    coeff = np.array([2*a*b, a*a-b*b])
    constant = np.array([2*(u@v), u@u-v@v])
    f2 = -np.dot(coeff, constant)/max(1e-25, coeff@coeff)
    if not (.25*size[1])**2 < f2 < (3*size[1])**2:
        return None
    f = np.sqrt(f2)
    K = np.array([[f, 0, center[0]], [0, f, center[1]], [0, 0, 1.]])
    basis = np.linalg.solve(K, H)
    length = np.linalg.norm(basis[:, :2], axis=0).mean()
    basis /= length
    up = np.cross(basis[:, 0], basis[:, 1])
    up /= np.linalg.norm(up)
    if up[1] > 0:
        up = -up
    # The vanilla upright camera has no roll. Orthonormal directions of equal
    # length provide extra leverage on pitch and vertical scale.
    pitch = np.arctan2(-up[2], -up[1])
    front = np.array([0., -np.sin(pitch), np.cos(pitch)])
    yaw = np.arctan2(front@basis[:, 0], basis[0, 0])
    handedness = np.sign(np.cos(yaw)*(front@basis[:, 1])-np.sin(yaw)*basis[0, 1])
    params = np.r_[yaw, pitch, np.log(f), basis[:, 2]]

    def residual(params):
        P = _upright_camera(params, center, handedness)
        dest = xz @ P[:, [0, 2]].T + P[:, 3]
        return (dest[:, :2]/dest[:, 2, None]-xy).ravel()

    for _ in range(8):
        r = residual(params)
        weights = np.minimum(1., 1.5/np.maximum(.01, np.abs(r)))**.5
        J = np.column_stack([(residual(params+delta)-r)/1e-6 for delta in 1e-6*np.eye(6)])
        update = np.linalg.lstsq(J*weights[:, None], -r*weights, rcond=None)[0]
        if np.linalg.norm(update) < 1e-8:
            break
        # Stop at the last useful fit for a poorly conditioned tile patch.
        next_params = params+update
        if abs(next_params[2]-params[2]) > 1 or np.linalg.norm(residual(next_params)*weights) > np.linalg.norm(r*weights):
            break
        params = next_params
    P = _upright_camera(params, center, handedness)
    H = P[:, [0, 2, 3]]
    if not 0 < -np.linalg.solve(P[:, :3], P[:, 3])[1] < 100:
        return None
    if np.any((xz @ H[2, :2]+H[2, 2]) <= 0):
        return None
    fit_error = np.linalg.norm(residual(params).reshape(-1, 2), axis=1)
    # Choose a nearby integer origin, grounding the center of the foreground
    # just below the middle of the screen when that direction sees the floor.
    ray = np.linalg.solve(H, [size[0]/2, .70*size[1], 1])
    if ray[2] > 1e-5:
        P[:, 3] += P[:, [0, 2]] @ np.rint(ray[:2]/ray[2])
    return P, np.exp(params[2]), fit_error


def predict(image, version='26.3'):
    """Treat repeated complete square floor tiles as proposed one-block units."""
    if version != '26.3':
        raise ValueError('Tile camera currently targets Java 26.3')
    uncertain = dict(P=None, scale_status='uncertain', settings=SETTINGS)
    if min(image.size) < 128:
        return dict(uncertain, reason='insufficient resolution to index floor tiles')
    centers, quads = _tiles(image)
    candidates = []
    for ids, cells in _patches(centers, quads):
        xy = centers[ids]
        if np.linalg.matrix_rank(cells-cells.mean(0)) < 2:
            continue
        # Refit after rejecting centers whose polygon was clipped or whose
        # topology was incorrectly linked to a neighboring component.
        keep = np.ones(len(ids), dtype=bool)
        for _ in range(3):
            if keep.sum() < SETTINGS['min_tiles']:
                break
            H = _homography(cells[keep], xy[keep])
            projected = cells @ H[:, :2].T + H[:, 2]
            err = np.linalg.norm(projected[:, :2]/projected[:, 2, None]-xy, axis=1)
            tile_size = np.mean(np.linalg.norm(quads[ids]-np.roll(quads[ids], 1, axis=1), axis=2), axis=1)
            keep = err < np.maximum(1.5, SETTINGS['consensus_cell_fraction']*tile_size)
        if keep.sum() < SETTINGS['min_tiles'] or min(np.ptp(cells[keep], axis=0)) < 2:
            continue
        solution = _projection(cells[keep], xy[keep], image.size)
        if solution:
            P, f, error = solution
            supported = error < np.maximum(1.5, SETTINGS['consensus_cell_fraction']*tile_size[keep])
            if supported.sum() >= max(SETTINGS['min_tiles'], .80*len(error)):
                candidates.append((int(supported.sum()), P, f, float(np.median(error))))
    if not candidates:
        return dict(uncertain, reason='no consistent two-dimensional array of complete square floor tiles',
                    outlined_tiles=len(centers))
    count, P, f, residual = max(candidates, key=lambda row: row[0])
    return dict(P=P.tolist(), scale_status='supported', settings=SETTINGS,
                reason='adjacent square floor tiles support a conditional one-block period',
                evidence=dict(indexed_tiles=count, outlined_tiles=len(centers),
                              focal_px=float(f), median_center_error_px=residual))
