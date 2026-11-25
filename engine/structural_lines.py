"""Whole-image straight segments from coarse and fine oriented edge support."""
import numpy as np
from PIL import Image, ImageFilter


def _edge_points(image, width, blur):
    ratio = min(1., width / max(image.size))
    size = tuple(max(1, round(n * ratio)) for n in image.size)
    if min(size) < 6:
        return None
    small = image.resize(size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(blur))
    rgb = np.asarray(small, dtype=np.float32)
    gy, gx = np.gradient(rgb, axis=(0, 1))
    # Principal normal of the RGB structure tensor. An isoluminant material
    # boundary can still produce a well-defined edge in the color channels.
    xx = np.mean(gx * gx, axis=2)
    yy = np.mean(gy * gy, axis=2)
    xy = np.mean(gx * gy, axis=2)
    angle = .5 * np.arctan2(2 * xy, xx - yy)
    mag = np.sqrt(.5 * (xx + yy + np.hypot(xx - yy, 2 * xy)))
    normal_bin = np.rint(angle * (4 / np.pi)).astype(int) % 4
    peak = np.zeros(size[::-1], dtype=bool)
    for direction, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        before = np.roll(mag, (dy, dx), axis=(0, 1))
        after = np.roll(mag, (-dy, -dx), axis=(0, 1))
        peak |= (normal_bin == direction) & (mag >= before) & (mag > after)
    peak &= mag >= max(1.5, min(8., np.quantile(mag, .76)))
    peak[[0, -1], :] = False
    peak[:, [0, -1]] = False
    y, x = np.nonzero(peak)
    if len(x) < 6:
        return None
    return (np.column_stack((x + .5, y + .5)), angle[y, x], mag[y, x],
            np.array(image.size) / size, size, rgb)


def _segments(image, scale, width, blur, min_length):
    edges = _edge_points(image, width, blur)
    if edges is None:
        return []
    points, angles, strength, to_original, size, rgb = edges
    minimum = max(12, min_length * max(size))
    diag = int(np.ceil(np.hypot(*size))) + 4
    candidates = []
    bin_width = 1.6
    for degrees in range(0, 180, 2):
        angle = np.deg2rad(degrees)
        normal = np.array([np.cos(angle), np.sin(angle)])
        tangent = np.array([-normal[1], normal[0]])
        aligned = np.abs((angles - angle + np.pi / 2) % np.pi - np.pi / 2) <= np.deg2rad(17)
        pts = points[aligned]
        if len(pts) < minimum * .45:
            continue
        vals = strength[aligned]
        rho = (pts @ normal + diag) / bin_width
        bins = np.rint(rho).astype(int)
        count = np.bincount(bins, minlength=int(2 * diag / bin_width) + 3)
        # A two-degree angle quantization can split a long line over bins.
        # Vote first, then fit the direction to the actual supporting pixels.
        smooth = np.convolve(count, [.45, 1., .45], mode='same')
        peaks = np.flatnonzero((smooth >= np.roll(smooth, -1)) &
                               (smooth > np.roll(smooth, 1)) &
                               (smooth >= minimum * .58))
        order = np.argsort(rho)
        rho, pts, vals = rho[order], pts[order], vals[order]
        for peak in peaks:
            start, stop = np.searchsorted(rho, [peak - 1.35, peak + 1.35])
            local = pts[start:stop]
            along = local @ tangent
            seq = np.argsort(along)
            u = along[seq]
            # Gaps are permitted at junctions or a few weak texture pixels.
            # Separated collinear scene objects remain separate segments.
            for idx in np.split(np.arange(len(u)), np.flatnonzero(np.diff(u) > 4.2) + 1):
                if len(idx) < .52 * minimum or u[idx[-1]] - u[idx[0]] < minimum:
                    continue
                pick = seq[idx]
                patch = local[pick]
                weights = np.minimum(vals[start:stop][pick], 35.)
                center = np.average(patch, axis=0, weights=weights)
                centered = patch - center
                offset = centered @ normal
                position = centered @ tangent
                slope = np.sum(weights * position * offset) / max(1., np.sum(weights * position**2))
                axis = tangent + slope * normal
                axis /= np.linalg.norm(axis)
                residual = np.abs(centered @ np.array([-axis[1], axis[0]]))
                good = residual <= 1.45
                if good.sum() < .5 * minimum:
                    continue
                support = centered[good] @ axis
                ends = np.array([support.min(), support.max()])
                length = ends[1] - ends[0]
                if length < minimum:
                    continue
                coverage = len(np.unique(np.rint(support))) / (length + 1)
                if coverage < .49:
                    continue
                # A random chain of oriented texture pixels often alternates
                # between light-to-dark and dark-to-light. Sample both sides
                # of the entire proposed segment and require consistent signed
                # RGB contrast, including the gaps between Hough supporters.
                samples = center + np.linspace(*ends, min(36, round(length)))[:, None] * axis
                offset_xy = (blur + .85) * np.array([-axis[1], axis[0]])
                positive = np.clip((samples + offset_xy).astype(int), 0, np.array(size) - 1)
                negative = np.clip((samples - offset_xy).astype(int), 0, np.array(size) - 1)
                jump = rgb[positive[:, 1], positive[:, 0]] - rgb[negative[:, 1], negative[:, 0]]
                energy = np.linalg.norm(jump, axis=1)
                coherence = np.linalg.norm(jump.mean(axis=0)) / max(.1, energy.mean())
                if coherence < .70 or np.median(energy) < 6:
                    continue
                contrast = float(np.median(vals[start:stop][pick][good])) * float(coherence)
                endpoints = (center + ends[:, None] * axis) * to_original
                rank = length**.72 * contrast**.65 * min(1., coverage)
                candidates.append((rank, endpoints, contrast, min(1., coverage), scale))
    return candidates


def detect(image):
    """Return coarse lines first and then finer lines, in source image pixels."""
    rgb = image.convert('RGB')
    candidates = [_segments(rgb, 'coarse', 460, 1.75, .055),
                  _segments(rgb, 'fine', 820, 1.05, .017)]
    result = []
    # Store a numeric index for collinearity checks. Skip near-coincident Hough
    # peaks and fine-scale rediscoveries of segments already in the outline.
    records = np.empty((720, 6))
    for batch, limit in zip(candidates, (200, 720)):
        for _, endpoints, strength, coverage, scale in sorted(batch, key=lambda row: -row[0]):
            p, q = endpoints
            diff = q - p
            length = np.linalg.norm(diff)
            axis = diff / length
            normal = np.array([-axis[1], axis[0]])
            old = records[:len(result)]
            near = old[np.abs(old[:, 4:] @ axis) > np.cos(np.deg2rad(4))]
            a, b = near[:, :2] - p, near[:, 2:4] - p
            # Compare perpendicular distance at the midpoint of the overlap.
            # Shared vanishing points allow distinct facade edges to intersect.
            ua, ub = a @ axis, b @ axis
            lower = np.maximum(0., np.minimum(ua, ub))
            upper = np.minimum(length, np.maximum(ua, ub))
            fraction = ((lower + upper) / 2 - ua) / (ub - ua)
            perpendicular = np.abs((a @ normal) + fraction * ((b - a) @ normal))
            tolerance = max(1.7, .0026 * max(image.size))
            if np.any((upper - lower >= .68 * np.minimum(length, np.abs(ub - ua))) &
                      (perpendicular < tolerance)):
                continue
            records[len(result)] = np.r_[p, q, axis]
            result.append(dict(p=p.round(2).tolist(), q=q.round(2).tolist(),
                               scale=scale, strength=round(strength, 3),
                               support=round(float(coverage), 3), length=round(float(length), 2)))
            if len(result) >= limit:
                break
    return result
