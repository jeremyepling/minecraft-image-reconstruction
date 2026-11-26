"""Visible surface masks from a single image, camera normals and relative inverse depth."""
import numpy as np
from PIL import Image, ImageFilter

from engine.camera_lines import _runs
from engine.normal_regions import _axes, _field, _local_mask, _patch, _subject as _rgb_subject, _unit
from engine.surface_model import canonical


SETTINGS = dict(analysis_width=720, max_angle_deg=44., min_area=48,
                foreground_step=.035, depth_step=.025, depth_radius=3, full_frame_depth_step=.16)


def _image_depth(raw, size):
    depth = np.asarray(Image.fromarray(raw.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    low, high = np.percentile(depth, [1., 99.])
    return (depth - low) / max(high - low, 1.e-8)


def _subject(image, vectors, depth, small):
    mean = vectors.mean(axis=(0, 1))
    if np.linalg.norm(mean) > .95 and mean[2] < -.4:
        all_pixels = np.ones(depth.shape, dtype=bool)
        return all_pixels, all_pixels, True

    # The scene can have several separate pieces. Compare to the backdrop on each
    # scanline without selecting just one foreground connected component.
    width = depth.shape[1]
    n = max(2, width // 16)
    sides = np.r_[:n, width-n:width]
    near = depth - np.median(depth[:, sides], axis=1)[:, None]
    background_normal = _unit(np.median(vectors[:, sides], axis=1))
    turn = np.einsum('ijk,ik->ij', vectors, background_normal) < .90
    rgb = np.asarray(small.filter(ImageFilter.GaussianBlur(1.6)), np.float32)
    refs = np.percentile(rgb[:, sides], [10, 50, 90], axis=1)
    color_delta = np.minimum.reduce([np.abs(rgb - c[:, None]).mean(axis=2) for c in refs])
    subject = ((near > SETTINGS['foreground_step']) & ((color_delta > 6) | (near > .13))
               | (turn & (color_delta > 10) & (near > -.035)))
    silhouette, _ = _rgb_subject(image, vectors)
    # The broad RGB silhouette suppresses distant clouds and seams in the floor.
    # Separate objects can still qualify through joint photometric and depth evidence.
    seeds = (near > .10) & (color_delta > 16)
    if np.ptp(depth) < 1.e-6:
        seeds = turn & (color_delta > 16)
    rescue = Image.fromarray(seeds).filter(ImageFilter.MaxFilter(9))
    # At least part of the region must be distinct from the distant backdrop.
    # Include a photographic/normal fallback when relative depth is unavailable.
    evidence = ((near > .018) & (color_delta > 6)) | seeds
    return subject & (silhouette | np.asarray(rescue)), evidence, False


def _depth_parts(runs, depth, full):
    """Split only if a persistent depth transition disconnects two substantial cores."""
    local, left, top = _local_mask(np.asarray(runs))
    height, width = local.shape
    radius = SETTINGS['depth_radius']
    if min(width, height) < 2 * radius + 1:
        return [runs]
    z = depth[top:top + height, left:left + width]
    y, x = np.indices(z.shape)
    dx = np.diff(z, axis=1)[local[:, :-1] & local[:, 1:]]
    dy = np.diff(z, axis=0)[local[:-1] & local[1:]]
    if not dx.size or not dy.size:
        return [runs]
    residual = z - np.median(dx) * x - np.median(dy) * y
    step = SETTINGS['full_frame_depth_step'] if full else SETTINGS['depth_step']
    barriers = np.zeros_like(local)
    for axis in (0, 1):
        values = residual if axis == 0 else residual.T
        mask = local if axis == 0 else local.T
        blocked = barriers if axis == 0 else barriers.T
        inside = mask[:-2*radius] & mask[2*radius:]
        jump = abs(values[:-2*radius] - values[2*radius:])
        # A straight sloping plane has nearly the same slope across the window.
        preceding = abs(values[radius:-radius] - values[:-2*radius])
        following = abs(values[2*radius:] - values[radius:-radius])
        strongest = np.maximum(preceding, following)
        weakest = np.minimum(preceding, following)
        blocked[radius:-radius] |= inside & (jump > step) & (strongest > 1.6*weakest)
    if not barriers.any():
        return [runs]
    minimum = max(SETTINGS['min_area'], int(local.sum() * .025))
    cores = [c for c in _runs(local & ~barriers) if sum(b-a for _, a, b in c) >= minimum]
    if len(cores) < 2:
        return [runs]
    labels = np.zeros(local.shape, dtype=np.int32)
    medians = []
    for index, core in enumerate(cores, 1):
        for v, a, b in core:
            labels[v,a:b] = index
        medians.append(np.median(residual[labels == index]))
    if np.ptp(medians) < step:
        return [runs]
    # Reassign every removed boundary pixel from its nearest connected core.
    # At equal distance choose the core with the closer relative plane offset.
    values = np.asarray([np.nan, *medians])
    while (local & (labels == 0)).any():
        padded = np.pad(labels, 1)
        neighbors = np.stack((padded[:-2,1:-1], padded[2:,1:-1], padded[1:-1,:-2], padded[1:-1,2:]))
        reach = local & (labels == 0) & (neighbors.max(0) > 0)
        if not reach.any():
            return [runs]
        options = neighbors[:, reach]
        error = np.where(options > 0, abs(values[options] - residual[reach]), np.inf)
        labels[reach] = np.take_along_axis(options, error.argmin(0)[None], axis=0)[0]
    parts = []
    for index in range(1, len(cores)+1):
        for component in _runs(labels == index):
            parts.append([(v+top, a+left, b+left) for v,a,b in component])
    return parts


def regions(image, normals, depth, version='26.3'):
    """Return disjoint half-open pixel masks. Optional native quads are sampling charts only."""
    if version != '26.3':
        raise ValueError('The mask-first experiment uses Java 26.3.')
    normals, depth = np.asarray(normals), np.asarray(depth)
    if normals.ndim != 3 or normals.shape[-1] != 3 or depth.ndim != 2 or not (
            np.isfinite(normals).all() and np.isfinite(depth).all()):
        raise ValueError('Expected finite numerical camera normals and relative inverse depth.')
    factor = min(1., SETTINGS['analysis_width'] / max(image.size))
    size = tuple(max(1, round(n*factor)) for n in image.size)
    if min(size) < 10:
        return []
    vectors = _field(normals, size)
    z = _image_depth(depth, size)
    small = image.convert('RGB').resize(size, Image.Resampling.BOX)
    gray = np.asarray(small.convert('L').filter(ImageFilter.GaussianBlur(.9)), dtype=np.float32)[..., None]
    subject, evidence, full = _subject(image, vectors, z, small)
    if not subject.any():
        return []
    axes = _unit(vectors.mean((0,1)))[None,:] if full else _axes(vectors, subject)
    directions = np.concatenate((axes, -axes))
    cosine = vectors @ directions.T
    labels = cosine.argmax(axis=2)
    labels[~subject | (cosine.max(axis=2) < np.cos(np.deg2rad(SETTINGS['max_angle_deg'])))] = -1
    scale = np.array(image.size) / size
    result = []
    for direction, normal in enumerate(directions):
        for runs in _runs(labels == direction):
            if sum(b-a for _,a,b in runs) < SETTINGS['min_area']:
                continue
            components = _depth_parts(runs, z, full)
            for part in components:
                area = sum(b-a for _,a,b in part)
                if sum(evidence[y,a:b].sum() for y,a,b in part) < .20 * area:
                    continue
                support = dict(size=list(size), runs=[list(run) for run in part])
                samples = np.concatenate([z[y,a:b] for y,a,b in part])
                region = dict(support=support, support_area=area, normal_camera=normal.tolist(),
                              relative_inverse_depth=float(np.median(samples)),
                              split_by_depth=len(components)>1, geometry_status='mask only')
                chart = _patch(part, gray, scale, size, normal)
                if chart is not None:
                    region.update(quad=canonical(chart['quad']), geometry_status='mask with sampling chart')
                result.append(region)
    result.sort(key=lambda p:-p['support_area'])
    for index, region in enumerate(result):
        region['id'] = f'mask-{index}'
    return result
