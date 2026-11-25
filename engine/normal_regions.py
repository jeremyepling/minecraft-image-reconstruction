"""Extract visible full-block surface patches from one RGB image and its camera normals."""
import numpy as np
from PIL import Image, ImageFilter

from engine.camera_lines import _outline, _runs
from engine.geometry import bilinear
from engine.plane_candidates import foreground
from engine.surface_edges import _components, _cross, _fit_boundary, _intersections, _order


SETTINGS = dict(analysis_width=540, max_angle_deg=30., min_area=28,
                min_quad_purity=.58, boundary_search_px=4.,
                rgb_subdivision_blur=2.8, min_subdivision_subject_fraction=.018,
                min_subdivision_contrast=15., min_turn_coverage=.4)


def _unit(vectors):
    return vectors / np.maximum(1.e-8, np.linalg.norm(vectors, axis=-1, keepdims=True))


def _field(vectors, size):
    return _unit(np.stack([
        np.asarray(Image.fromarray(vectors[..., j].astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
        for j in range(3)], axis=-1))


def _subject(image, normals):
    # A close crop can be one uninterrupted facade all the way to the border.
    shape = normals.shape[:2]
    mean = normals.mean((0, 1))
    if np.linalg.norm(mean) > .95 and mean[2] < -.4:
        return np.ones(shape, bool), True
    _, raw = foreground(image)
    expanded = Image.fromarray(np.uint8(raw)*255).resize(shape[::-1], Image.Resampling.NEAREST)
    return np.asarray(expanded.filter(ImageFilter.MaxFilter(9))) > 0, False


def _axes(normals, mask):
    # Java's upright camera has no roll. Floor/roof pixels constrain pitch
    # without needing their depth, unit spacing, or location in the world.
    samples = normals[::3, ::3].reshape(-1, 3)
    level = samples[(abs(samples[:, 1]) > .80) & (abs(samples[:, 0]) < .25)]
    up = np.array([0., -1., 0.])
    if len(level) >= 20:
        up[2] = np.median(level[:, 2] / level[:, 1]) * up[1]
        up = _unit(up)
    right = np.array([1., 0., 0.])
    forward = np.cross(up, right)
    walls = normals[::2, ::2][mask[::2, ::2]]
    walls = walls[abs(walls @ up) < .55]
    if not len(walls):
        return np.stack((right, up, forward))
    # North/east/sign swaps all have the same angle modulo a quarter-turn.
    angles = np.arctan2(walls @ forward, walls @ right)
    residues = angles % (np.pi/2)
    hist, bins = np.histogram(residues, bins=45, range=(0, np.pi/2))
    smoothed = sum(np.roll(hist, k) for k in range(-3, 4))
    peak = bins[smoothed.argmax()] + (bins[1]-bins[0])/2
    residual = (residues - peak + np.pi/4) % (np.pi/2) - np.pi/4
    angle = peak + np.median(residual[abs(residual) < np.radians(14)])
    horizontal = np.cos(angle)*right + np.sin(angle)*forward
    return np.stack((horizontal, up, np.cross(horizontal, up)))


def _rgb_edges(gray, quad):
    """Align each measured normal boundary to a coherent nearby RGB edge."""
    segments = np.roll(quad, -1, axis=0)-quad
    length = np.linalg.norm(segments, axis=1)
    directions = segments / np.maximum(1.e-8, length[:, None])
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    t = np.linspace(.08, .92, 28)
    shifts = np.arange(-SETTINGS["boundary_search_px"], SETTINGS["boundary_search_px"]+.1, .5)
    lines = []
    for a, segment, n in zip(quad, segments, normals):
        pts = a + segment[None, :] * t[:, None]
        samples = pts[None, :, :] + shifts[:, None, None]*n
        delta = (bilinear(gray, samples+1.1*n) - bilinear(gray, samples-1.1*n))[..., 0]
        weight = np.abs(np.clip(delta, -35., 35.).mean(1)) - 1.3*np.abs(shifts)
        shift = shifts[weight.argmax()] if weight.max() > 4 else 0.
        lines.append([*n, -((a+shift*n) @ n)])
    aligned = _intersections(np.asarray(lines))
    if aligned is None or np.max(np.linalg.norm(aligned-quad, axis=1)) > 9:
        return quad
    return aligned


def _local_mask(rows):
    y, a, b = rows.T
    left, top = a.min(), y.min()
    mask = np.zeros((y.max()-top+1, b.max()-left), bool)
    for y, a, b in rows:
        mask[y-top, a-left:b-left] = True
    return mask, int(left), int(top)


def _parts(runs, colors, rgb, minimum):
    rows = np.asarray(runs)
    local, x0, y0 = _local_mask(rows)
    region = colors[y0:y0+local.shape[0], x0:x0+local.shape[1]]
    image = rgb[y0:y0+local.shape[0], x0:x0+local.shape[1]]
    parent_color = np.median(image[local], axis=0)
    ids, counts = np.unique(region[local], return_counts=True)
    for index, pixels in zip(ids, counts):
        if pixels < minimum or pixels > .86 * local.sum():
            continue
        subset = local & (region == index)
        contrast = float(np.max(abs(np.median(image[subset], axis=0) - parent_color)))
        for split in _runs(subset):
            if sum(b-a for _,a,b in split) >= minimum:
                yield [(y+y0, a+x0, b+x0) for y,a,b in split], contrast


def _remainder(runs, children):
    local, x0, y0 = _local_mask(np.asarray(runs))
    for child in children:
        for y, a, b in child["support"]["runs"]:
            local[y-y0, a-x0:b-x0] = False
    for component in _runs(local):
        yield [(y+y0, a+x0, b+x0) for y,a,b in component]


def _outside_turns(quad, labels, k):
    # A color subdivision needs transitions away from the parent normal
    # along multiple edges, otherwise a contrasting brick becomes a surface.
    edges = np.roll(quad, -1, axis=0)-quad
    lengths = np.linalg.norm(edges, axis=1)
    outward = np.column_stack((edges[:,1], -edges[:,0]))/lengths[:,None]
    samples = quad[:,None,:]+np.linspace(.10,.90,19)[None,:,None]*edges[:,None,:]
    bands = []
    for distance in (2., 4., 7.):
        exterior = np.clip((samples+distance*outward[:,None,:]).astype(int),
                           0, np.array(labels.shape[::-1])-1)
        bands.append(labels[exterior[...,1],exterior[...,0]] != k)
    return np.any(bands, axis=0).mean(1)


def _patch(runs, gray, scale, size, normal):
    rows = np.asarray(runs)
    y, a, b = rows.T
    area = int((b-a).sum())
    if area < SETTINGS["min_area"]:
        return None
    local, x0, y0 = _local_mask(rows)
    ex_y, ex_x = np.where(np.diff(np.pad(local, ((0,0),(1,1))), axis=1))
    ey_y, ey_x = np.where(np.diff(np.pad(local, ((1,1),(0,0))), axis=0))
    boundary = np.vstack((np.column_stack((ex_x, ex_y+.5)), np.column_stack((ey_x+.5, ey_y))))+[x0,y0]
    quad = _outline(map(tuple, boundary))
    if len(quad) != 4:
        return None
    fit = _fit_boundary(boundary, _order(quad))
    if fit is None:
        return None
    quad, coverage = fit
    qarea = abs(_cross(quad, np.roll(quad, -1, axis=0)).sum())/2
    purity = min(1., area / max(1., qarea))
    if purity < SETTINGS["min_quad_purity"]:
        return None
    quad = np.clip(_rgb_edges(gray, quad), [0,0], size)
    edges = np.roll(quad, -1, axis=0)-quad
    if np.min(_cross(edges, np.roll(edges, -1, axis=0))) <= 0 or min(np.linalg.norm(edges, axis=1)) < 2.2:
        return None
    return dict(quad=np.round(quad*scale, 2).tolist(), confidence=round(float(purity*np.mean(coverage)), 4),
                normal_camera=normal.tolist(), support_area=area,
                support=dict(size=list(size), runs=rows.tolist()))


def detect(image, normals, version="26.3"):
    """Return native-pixel envelopes with independent pixel support.

    Each support has size [analysis_width, analysis_height] and half-open
    [row, x0, x1] runs. A native pixel center maps to an analysis cell by
    multiplying separately by the width and height ratios and flooring.
    A quad can enclose openings absent from the reported support.
    """
    if version != "26.3":
        raise ValueError("The normal-region estimator targets Java 26.3.")
    normals = np.asarray(normals)
    if normals.ndim != 3 or normals.shape[2] != 3 or not np.isfinite(normals).all():
        raise ValueError("Expected a finite H×W×3 camera-normal field")
    factor = min(1., SETTINGS["analysis_width"]/max(image.size))
    size = tuple(max(1, round(n*factor)) for n in image.size)
    if min(size) < 10:
        return []
    image = image.convert("RGB")
    vectors = _field(normals, size)
    mask, full = _subject(image, vectors)
    if full:
        directions = _unit(vectors.mean((0,1)))[None,:]
    else:
        basis = _axes(vectors, mask)
        directions = np.concatenate((basis, -basis))
    cosine = vectors @ directions.T
    labels = cosine.argmax(2)
    usable = mask & (cosine.max(2) > np.cos(np.radians(SETTINGS["max_angle_deg"])))
    labels[~usable] = -1
    small = image.resize(size, Image.Resampling.BOX)
    gray = np.asarray(small.convert("L").filter(ImageFilter.GaussianBlur(.9)), dtype=np.float32)[..., None]
    rgb = np.asarray(small.filter(ImageFilter.GaussianBlur(SETTINGS["rgb_subdivision_blur"])))
    colors = _components(rgb) if not full else None
    min_child = max(2*SETTINGS["min_area"], SETTINGS["min_subdivision_subject_fraction"] * mask.sum())
    scale = np.array(image.size) / size
    patches = []
    for k, normal in enumerate(directions):
        for runs in _runs(labels == k):
            # Reject pieces of the terrain which merely entered the silhouette's margin.
            if not full and any(y in (0, size[1]-1) or a == 0 or b == size[0] for y, a, b in runs):
                continue
            part = _patch(runs, gray, scale, size, normal)
            area = sum(b-a for _,a,b in runs)
            children = []
            if not full and area >= 2*min_child and (not part or part["confidence"] < .9):
                for split, contrast in _parts(runs, colors, rgb, min_child):
                    child = _patch(split, gray, scale, size, normal)
                    if child is None:
                        continue
                    turns = float(_outside_turns(np.array(child["quad"])/scale, labels, k).mean())
                    if turns < SETTINGS["min_turn_coverage"]:
                        continue
                    dominant = child["support_area"] > .35*area and turns > .75
                    if contrast < SETTINGS["min_subdivision_contrast"] and not dominant:
                        continue
                    child.update(source="RGB subdivision", color_contrast=contrast,
                                 outside_turn_coverage=round(turns, 4))
                    children.append(child)
            if children:
                patches.extend(children)
                for remaining in _remainder(runs, children):
                    # Small residual fragments can be the blurred rim of a strong RGB edge.
                    if sum(b-a for _,a,b in remaining) < min_child:
                        continue
                    rest = _patch(remaining, gray, scale, size, normal)
                    if rest:
                        rest["source"] = "remaining normal component"
                        patches.append(rest)
            elif part is not None:
                part["source"] = "normal component"
                patches.append(part)
    patches.sort(key=lambda p: -p["support_area"])
    for i, p in enumerate(patches):
        p["id"] = f"normal-region-{i}"
    return patches
