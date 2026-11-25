"""Rank visible masonry by its repeated spatial pattern and its local lighting."""
from typing import NamedTuple

import numpy as np

from engine.geometry import bilinear, project
from engine.courses import course_evidence
from engine.lighting import linear, srgb


LUMA = np.array([.213, .715, .072])


class PatternBank(NamedTuple):
    names: np.ndarray
    colors: np.ndarray
    folded: np.ndarray
    masonry: np.ndarray
    native: np.ndarray


def pattern_templates(tiles, names):
    """Include competing patterns: staggered planks can resemble brickwork."""
    # Each canonical tile is periodic. Preserve the orientation and phase of its joints.
    small = (tiles @ LUMA).reshape(len(tiles), 8, 2, 8, 2).mean(axis=(2, 4))
    smooth = (small + np.roll(small, 2, 1) + np.roll(small, -2, 1)
              + np.roll(small, 2, 2) + np.roll(small, -2, 2)) / 5
    detail = small - smooth
    norm = np.linalg.norm(detail, axis=(1, 2))
    valid = norm > 1
    if not valid.any():
        return None
    patterns = detail[valid] / norm[valid, None, None]
    names = names[valid]
    masonry = np.array([name == 'bricks' or (name.endswith('_bricks') and 'chiseled' not in name)
                        for name in names])
    if not masonry.any():
        return None
    native = tiles[valid] @ LUMA
    native -= native.mean(axis=(1, 2), keepdims=True)
    native /= np.maximum(1., np.linalg.norm(native, axis=(1, 2), keepdims=True))
    return PatternBank(names, tiles[valid].mean(axis=(1, 2))/255, np.fft.rfft2(patterns).conj(),
                       masonry, np.fft.rfft2(native).conj())


def competing_face(patch, bank):
    """Recognize a complete nonbrick tile even when the crop starts mid-pattern."""
    names, colors, _, masonry, native = bank
    tone = patch @ LUMA
    tone -= tone.mean()
    if tone.std() < 2 or masonry.all():
        return None
    query = np.fft.rfft2(tone/np.linalg.norm(tone))
    response = np.fft.irfft2(query*native, s=(16, 16), axes=(-2, -1)).max(axis=(1, 2))
    best = response[~masonry].max()
    if best < .80 or best < response[masonry].max() + .08:
        return None
    # Several woods can share one grain pattern. Use color to break a close tie.
    labels = np.flatnonzero(~masonry & (response >= best-.05))
    observed = patch.mean(axis=(0, 1))/255
    gain = np.clip((observed @ LUMA + .015)/(colors[labels] @ LUMA + .015), .4, 1.85)
    color_error = np.linalg.norm(observed - gain[:, None]*colors[labels], axis=1)
    costs = 1-response[labels] + color_error + .06*np.abs(np.log(gain))
    order = np.argsort(costs, kind='stable')[:5]
    label = labels[order[0]]
    return names[labels[order]].tolist(), costs[order].tolist(), float(response[label])


def region_ranking(rgb, transform, evidence, bank, light_gains):
    """Fold several observed courses onto a texture tile. Search scale and phase."""
    names, colors, templates, masonry, _ = bank
    left, top, right, bottom = evidence['patch_uv']
    tile_width = evidence['brick_width_cells']
    tile_height = 2 * evidence['course_height_cells']
    width = int(np.clip(np.ceil(16*(right-left)/tile_width), 8, 128))
    height = int(np.clip(np.ceil(16*(bottom-top)/tile_height), 8, 128))
    y, x = np.mgrid[:height, :width]
    u = (x + .5) * ((right-left)/width)
    v = (y + .5) * ((bottom-top)/height)
    pixels = bilinear(rgb, project(transform, np.stack((left+u, top+v), axis=-1)))
    light = pixels @ LUMA
    padded = np.pad(light, 2, mode='edge')
    smooth = (padded[2:-2, :-4] + padded[2:-2, 4:] + padded[:-4, 2:-2]
              + padded[4:, 2:-2] + light) / 5
    detail = (light - smooth).ravel()
    scale = max(np.sqrt(np.mean(detail**2))*8, 1)
    correlation = np.full(len(names), -1.)
    for sx in (.8, .9, 1., 1.1, 1.2, 1.6, 2.):
        tx = (u * 8/(tile_width*sx)).astype(int) % 8
        for sy in (.94, 1., 1.06):
            ty = (v * 8/(tile_height*sy)).astype(int) % 8
            bins = (8*ty + tx).ravel()
            count = np.bincount(bins, minlength=64)
            if np.count_nonzero(count) < 48:
                continue
            folded = np.bincount(bins, weights=detail, minlength=64) / np.maximum(count, 1)
            tile = (folded - folded.mean()).reshape(8, 8) / scale
            response = np.fft.irfft2(np.fft.rfft2(tile)*templates, s=(8, 8), axes=(-2, -1))
            correlation = np.maximum(correlation, response.max(axis=(1, 2)))
    if correlation.max() < .18:
        return None

    # Fit one bounded light/color adjustment across the complete patch. It can
    # explain warm-lit gray, while a large real difference in hue still costs.
    observed = srgb(linear(pixels/255)*light_gains).mean(axis=(0, 1))
    gain = np.clip((observed @ LUMA + .015)/(colors @ LUMA + .015), .4, 1.85)
    delta = observed - colors * gain[:, None]
    tint = np.clip(delta - (delta @ LUMA)[:, None], -.035, .035)
    residual = np.linalg.norm(delta - tint, axis=1)
    costs = (1 - np.clip(correlation, -1, 1) + 2*residual
             + .12*np.abs(np.log(gain)) + .22*np.linalg.norm(tint, axis=1))
    order = np.argsort(costs, kind='stable')
    if not masonry[order[0]] or correlation[order[0]] < .18:
        return None
    order = order[masonry[order]]
    return names[order], costs[order], correlation[order]


def course_matches(rgb, plane, cells, bank, face_rows, agreement, light_gains):
    """Use the same measured region for its member cells; retain reliable face detail."""
    if bank is None or not plane or not {'grid_to_image','quad','columns','rows'} <= plane.keys():
        return
    transform = np.asarray(plane['grid_to_image'])
    regions = {}
    aligned_plane = np.median(agreement) >= .8
    for index, evidence in course_evidence(rgb, plane, cells).items():
        key = (tuple(evidence['patch_uv']), evidence['brick_width_cells'], evidence['course_height_cells'])
        if key not in regions:
            regions[key] = region_ranking(rgb, transform, evidence, bank, light_gains)
        ranking = regions[key]
        if ranking is None:
            continue
        names, costs, correlation = ranking
        face_name = face_rows[index][0][0]
        # An independently aligned full face remains valuable at changes of material.
        if (agreement[index] >= .8 or aligned_plane) and face_name not in names:
            continue
        if agreement[index] >= .5 and face_name in names:
            costs = costs.copy()
            costs[names == face_name] -= .12 * agreement[index]
            order = np.argsort(costs, kind='stable')
            names, costs, correlation = names[order], costs[order], correlation[order]
        margin = (costs[1]-costs[0])/max(costs[1], 1e-6) if len(costs)>1 else 0.
        yield index, names[:5].tolist(), costs[:5].tolist(), float(margin), dict(
            evidence, atlas_correlation=round(float(correlation[0]), 4))


def context_matches(rgb, plane, cells, bank, patches, face_rows, agreement, light_gains):
    """Yield only supported changes to the plane's existing material ranking."""
    if bank is None:
        return
    for j, names, costs, margin, region in course_matches(rgb, plane, cells, bank, face_rows, agreement, light_gains):
        details = dict(material_method='masonry', masonry_evidence=region)
        competitor = competing_face(patches[j], bank)
        if competitor is not None:
            names, costs, correlation = competitor
            margin = (costs[1]-costs[0])/max(costs[1], 1e-6) if len(costs)>1 else 0.
            details = dict(material_method='shifted face', atlas_correlation=correlation)
        yield j, names, costs, margin, details
