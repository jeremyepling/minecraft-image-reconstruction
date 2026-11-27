"""Find putative horizontal surfaces where a framed opening shows the distant ground."""
import numpy as np

from engine.pillar_seams import native_mask, orientation


def _rim(mask, walls, axis, sign):
    """Fraction of side scanlines meeting visible wall pixels shortly past the edge."""
    if axis == 1:
        mask, walls = mask.T, walls.T
    rows = np.flatnonzero(mask.any(axis=1))
    rows = rows[len(rows)//8:len(rows)-len(rows)//8]
    if not len(rows):
        return 0.
    span = mask[rows]
    edge = span.argmax(axis=1) if sign < 0 else span.shape[1] - 1 - span[:, ::-1].argmax(axis=1)
    # Model masks are sampled below native resolution. Allow a small gap and
    # require a band of vertical surface, excluding isolated noisy rim pixels.
    offsets = np.arange(1, 14)
    xs = edge[:, None] + sign * offsets
    valid = (xs >= 0) & (xs < walls.shape[1])
    sample = walls[rows[:, None], xs.clip(0, walls.shape[1]-1)] & valid
    return float((sample.sum(axis=1) >= 3).mean())


def _enclosure(mask, walls):
    left, right = (_rim(mask, walls, 0, d) for d in (-1, 1))
    head, sill = (_rim(mask, walls, 1, d) for d in (-1, 1))
    if min(left, right) >= .45 and min(head, sill) >= .65:
        return [left, right, head, sill]
    return None


def _ground(inv, eye, x, y):
    ray = np.stack((x+.5, y+.5, np.ones(len(x))), axis=1) @ inv.T
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = -eye[1] / ray[:, 1]
    valid = np.isfinite(depth) & (depth > 2) & (depth < 120)
    texel = np.zeros((len(x), 2), int)
    texel[valid] = np.floor((eye[[0, 2]] + depth[valid, None]*ray[valid][:, [0, 2]])*16).astype(int) % 16
    return texel[:, 0] + 16*texel[:, 1], depth, valid


def _tile(texel, color):
    """Estimate the native game tile from RGB samples outside the structure."""
    order = np.argsort(texel, kind='stable')
    keys = texel[order]
    groups = np.split(order, np.flatnonzero(np.diff(keys))+1)
    tile = np.full((256, 3), np.nan)
    for group in groups:
        if len(group) >= 8:
            tile[texel[group[0]]] = np.median(color[group], axis=0)
    return tile


def _correlation(expected, actual):
    a = expected - expected.mean()
    b = actual - actual.mean()
    return float((a*b).sum() / max(1.e-6, np.linalg.norm(a)*np.linalg.norm(b)))


def _phase_scores(tile, texel, rgb):
    expected = tile[texel]
    if len(texel) < 120 or not np.isfinite(expected).all() or len(np.unique(texel)) < 64:
        return None
    actual = rgb.mean(axis=1)
    gray = tile.mean(axis=1)
    corr = _correlation(gray[texel], actual)
    rivals = []
    for offset in (2, 4, 8):
        rivals.append(_correlation(gray[16*(texel//16) + (texel+offset)%16], actual))
        rivals.append(_correlation(gray[(texel+16*offset)%256], actual))
    gain = np.clip(np.median((actual+1)/(expected.mean(axis=1)+1)), .78, 1.05)
    error = abs(rgb - gain*expected).mean(axis=1)
    return [corr, corr-max(rivals), float(error.mean()), float((error > 8).mean())]


def inspect_openings(image, parts, camera):
    """Return source-only diagnostics for horizontal patches enclosed by vertical surfaces."""
    if not parts or not camera or camera.get('P') is None:
        return []
    projection = np.asarray(camera['P'], dtype=float)
    if (projection.shape != (3, 4) or not np.isfinite(projection).all()
            or np.linalg.matrix_rank(projection[:, :3]) != 3):
        return []
    p, rotation = orientation(camera, image.size)
    inv = np.linalg.inv(p[:, :3])
    eye = -inv @ p[:, 3]
    walls = np.zeros(image.size[::-1], bool)
    union = np.zeros_like(walls)
    horizontals = []
    for part in parts:
        mask = native_mask(part, image.size)
        union |= mask
        normal = part.get('normal_camera')
        if normal is None:
            continue
        normal = rotation.T @ normal
        if abs(normal[1]) < .35:
            walls |= mask
        elif normal[1] > .85 and mask.sum() >= 350:
            horizontals.append((part['id'], mask))
    candidates = []
    for key, mask in horizontals:
        y, x = np.where(mask)
        if np.ptp(x) < 10 or np.ptp(y) < 10:
            continue
        rim = _enclosure(mask, walls)
        if rim is not None:
            candidates.append((key, mask, rim))
    if not candidates:
        return []
    # The two independent ground templates use opposite sides of the entire
    # building silhouette. A framed platform cannot train its own template.
    yy, xx = np.nonzero(union)
    ground_sets = []
    rgb = np.asarray(image.convert('RGB'))
    for columns in (np.arange(0, xx.min(), 2), np.arange(xx.max()+1, image.width, 2)):
        x, y = np.meshgrid(columns, np.arange(0, image.height, 2))
        x, y = x.ravel(), y.ravel()
        uv, depth, good = _ground(inv, eye, x, y)
        ground_sets.append((uv[good], depth[good], rgb[y[good], x[good]]))
    output = []
    for key, mask, rim in candidates:
        y, x = np.nonzero(mask)
        uv, depth, valid = _ground(inv, eye, x, y)
        if valid.mean() < .99:
            continue
        target_depth = np.median(depth[valid])
        record = dict(id=key, rim=rim, samples=len(x), ground_depth=float(target_depth), folds=[], reject=False)
        output.append(record)
        for ground_uv, depths, colors in ground_sets:
            near = (depths > target_depth/1.45) & (depths < target_depth*1.45)
            if near.sum() < 2048:
                break
            tile = _tile(ground_uv[near], colors[near])
            if not np.isfinite(tile).all() or tile.mean(axis=1).std() < 2:
                break
            # Grade the checkerboard halves of the putative opening independently
            # as well as the two disjoint exterior patches used to learn the tile.
            fold = ((x//10 + y//10) % 2).astype(bool)
            for take in (valid & fold, valid & ~fold):
                scores = _phase_scores(tile, uv[take], rgb[y[take], x[take]])
                if scores is not None:
                    record['folds'].append(scores)
        record['reject'] = len(record['folds']) == 4 and all(
            corr >= .72 and margin >= .10 and error <= 4 and outliers <= .08
            for corr, margin, error, outliers in record['folds'])
    return output


def detect(image, parts, camera):
    """Return framed horizontal surface IDs whose pixels match the surrounding pavement."""
    return [record['id'] for record in inspect_openings(image, parts, camera) if record['reject']]
